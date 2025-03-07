"""
Copyright Statement:

Author: Ratta Chindasilpa
Author's email: raththar@hotmail.com

This code is part of the master's thesis of Ratta Chindasilpa, "Deep Reinforcement Learning for Adaptive Control of Heater Position and Heating Power in a Smart Greenhouse," 
developed at Wageningen University and Research.
"""

# Import necessary libraries
from greenhouse_geometry import GreenhouseGeometry
import numpy as np
import pandas as pd
import openstudio
import subprocess
import os
import tensorflow
import gymnasium as gym
import gymnasium.spaces as spaces
from gymnasium.spaces import Discrete, MultiDiscrete
from tensorflow.keras.models import load_model # type: ignore

# Gymnasium Environment
class GreenhouseEnv(gym.Env):
    """
    A Gymnasium environment for greenhouse simulation
    """ 
    # Greenhouse environment initialization
    def __init__(self, 
                 start_month = 2, 
                 end_month = 2,
                 start_day = 28,
                 end_day = 28,
                 start_hour = 1,
                 num_width = 8,
                 num_length = 8,
                 num_heater = 4,
                 init_action = [0, 0, 0, 0, 18.0, 18.0, 18.0, 18.0],
                 init_position = [18, 21, 42, 45],
                 init_reward = 0,
                 model_path = "energyplus_data/model_files/greenhouse_triangle.osm",
                 energyplus_exe_path = r"C:\EnergyPlusV24-2-0\energyplus.exe",
                 epw_file_path = "datasets/Rotterdam_the_Hague_2023.epw",
                 output_dir = "energyplus_data",
                 output_prefix = "updated_sim_",
                 modified_osm_path = r"energypluse_data/model_files/modified_model.osm",
                 modified_idf_path = r"energypluse_data/model_files/modified_model.idf"
                 ):
        super().__init__()

        # Define parameters
        self.start_month = start_month
        self.end_month = end_month
        self.start_day = start_day
        self.end_day = end_day
        self.start_hour = start_hour
        self.num_width = num_width
        self.num_length = num_length
        self.num_heater = num_heater
        self.num_action = 5 # Stay, Up, Down, Left, Right
        self.num_energy_lvl = 3 # Low (0), Medium (1), High (2)
        self.grid_size = num_width * num_length # Total thermal zones
        self.init_action = init_action
        self.init_position = init_position
        self.init_reward = init_reward
        self.model_path = model_path
        self.energyplus_exe_path = energyplus_exe_path
        self.epw_file_path = epw_file_path
        self.output_dir = output_dir
        self.output_prefix = output_prefix
        self.modified_osm_path = modified_osm_path
        self.modified_idf_path = modified_idf_path

        # Initialize state
        self.position = None
        self.temp = None
        self.elec = None
        self.observation = None
        self.reward = 0
        self.terminated = False   # the episode termination

        # Initialize the data frames
        self.action_df = None
        self.position_df = None
        self.temp_df = None
        self.elec_df = None
        self.reward_df = None

        # Initialize mini-step
        self.mini_step = 0
        self.mini_step_track = 0

        # Initialize the OpenStudio greenhouse model
        try:
            model_file_path = openstudio.path(self.model_path)
            self.greenhouse_model = openstudio.model.Model.load(model_file_path).get()
            # print(f"Model successfully loaded from {self.model_path}")
        except Exception as e:
            raise RuntimeError(f"Failed to load the OpenStudio model from {self.model_path}. Error: {e}")
        
        # Set up the observation spaces
        # Observation space with predicted weather data
        temperature_low = [0.0] * self.grid_size
        temperature_high = [30.0] * self.grid_size

        # Predicted weather data
        pred_step = 6
        pred_weather_data_low = [0.0] * self.grid_size * pred_step
        pred_weather_data_high = [30.0] * self.grid_size * pred_step

        # Heater position ranges
        heater_low = [0] * self.num_heater
        heater_high = [self.num_width * self.num_length - 1] * self.num_heater

        # Time components
        month_low = [1]
        month_high = [12 + 400]
        day_low = [1]
        day_high = [30 + 400]
        hour_low = [1]
        hour_high = [24 + 400]

        # Observation space with future weather data
        self.observation_space = spaces.Box(
            low=np.array(temperature_low + 
                         pred_weather_data_low +
                         heater_low + 
                         month_low + 
                         day_low + 
                         hour_low, 
                         dtype=np.float32),
            high=np.array(temperature_high + 
                          pred_weather_data_high +
                          heater_high + 
                          month_high + 
                          day_high + 
                          hour_high, 
                          dtype=np.float32),
            dtype=np.float32  # Use float32 for compatibility
        )

        # Set up the action spaces for PPO model
        # MultiDiscrete action
        self.num_action_per_heater = self.num_action * self.num_energy_lvl  # 5 movements * 3 energy levels = 15
        self.action_space = MultiDiscrete([self.num_action_per_heater] * self.num_heater)

    # Reset Method
    def reset(self, seed=None, options=None):
        """
        Resets the environment to its initial state.
        """
        super().reset(seed=seed)

        # Reset the episode time
        self.month = self.start_month
        self.day = self.start_day
        self.hour = self.start_hour
        print(f"Reset the episode time: Month:{self.month}, Day:{self.day}, Hour:{self.hour}")

        # Reset mini-step
        self.mini_step = 0
        self.mini_step_track = 0
        
        # Reset the reward
        self.reward = 0

        # Reset the episode termination
        self.terminated = False

        # Reset the action DataFrame with initial actions
        self.action_df = self.init_action_df()

        # Reset the position DataFrame with initial heater positions
        self.position_df = self.init_position_df()

        # Reset the reward DataFrame with initial reward
        self.reward_df = self.init_reward_df()

        # Reset the temperature state to the first time step
        # Create the initial greenhouse model
        self.greenhouse_model = self.create_greenhouse(start_month=self.month, 
                                                       end_month=self.month,
                                                       start_day=self.day,
                                                       end_day=self.day)
        # Add HVAC system (heaters) to the greenhouse based on initial position
        self.add_hvac(position=self.init_position)
        # Assign the initial heater schedule
        self.assign_schedule(position=self.init_position, 
                             start_month=self.month, 
                             end_month=self.month, 
                             start_day=self.day, 
                             end_day=self.day, 
                             hour=self.hour)
        # Run the EnergyPlus simulation
        self.run_energyplus()
        # Reset the temperature data frame by extracting simulation outputs
        self.temp_df = self.init_temp_df()
        # Reset the electricity rate data frame by extracting simulation outputs
        self.elec_df = self.init_elec_df()
        
        # Reset the observation to the first time step
        # Get the first temperature state
        self.temp = self.get_temp(month=self.month, 
                                  day=self.day, 
                                  hour=self.hour)
        # Get the first electric rate state
        self.elec = self.get_elec(month=self.month,
                                  day=self.day,
                                  hour=self.hour)
        # Get the first heater position
        self.position = self.get_position(month=self.month, 
                                          day=self.day, 
                                          hour=self.hour)
        
        # Process the observation
        # Get the predicted weather data
        pred_weather = self.predict_next_temp(model_path='gru_model.keras',
                                              month=self.month,
                                              day=self.day,
                                              hour=self.hour)
        
        # Get the first timestep observation with predicted weather data
        self.observation = self.get_obs_with_pred_weather(temp_list=self.temp,
                                                          pred_weather=pred_weather,
                                                          position_list=self.position,
                                                          month=self.month,
                                                          day=self.day,
                                                          hour=self.hour)

        # print("--------\n", f"Observation: {self.observation}", "\n--------")

        # Save DataFrames to CSV for record-keeping
        try:
            # Save each DataFrame
            self.temp_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}temp.csv"), index=False)
            self.elec_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}elec.csv"), index=False)
            self.position_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}position.csv"), index=False)
            self.action_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}action.csv"), index=False)
            self.reward_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}reward.csv"), index=False)
            print(f"DataFrames saved to CSV successfully.", "\n--------")
        except Exception as e:
            print(f"Error saving DataFrames to CSV: {e}")

        # Create an empty info dictionary
        info = {}

        # Return the observation
        return self.observation, info
    
    # Step method
    def step(self, action):
        """
        Executes one time step within the environment based on the action.
        Manages mini-steps for finer control before running full simulations.

        Parameters:
        - action (list): Actions for each heater in the environment.

        Returns:
        - observation (np.ndarray): The updated observation after the step.
        - reward (float): The reward for the action taken.
        - terminated (bool): Whether the episode has ended naturally.
        - truncated (bool): Whether the episode was artificially terminated.
        - info (dict): Additional information about the step.
        """
        # Show the current date time
        # print(f"Month: {self.month}, Day: {self.day}, Hour: {self.hour}")

        # MultiDiscrete action
        decoded_action = [(a % self.num_action, a // self.num_action) for a in action]
        # print(f"Decoded Action: {decoded_action}")

        # Separate movement and energy level action
        movement_action = [heater_action[0] for heater_action in decoded_action]
        energy_lvl = [heater_action[1] for heater_action in decoded_action]

        # Convert the energy levels to the energy action (thermostat temperature)
        energy_action = self.convert_energy_lvl_to_thermostat(energy_levels=energy_lvl)

        # Check if still within mini-steps
        if self.mini_step < 4:
            # Increment mini-step counter
            self.mini_step += 1
            # print(f"Mini-step: {self.mini_step}")
            self.mini_step_track += 100
            # print(f"Mini-step track: {self.mini_step_track}")

            # Update the action data frame
            print(f"Movement action (mini-step): {movement_action}, Energy action (mini-step): {energy_action}")
            full_action = movement_action + energy_action
            # print(f"Mini-step full action: {full_action}")

            # Show the current heater positions
            current_position = self.position
            # print(f"Current heater positions: {current_position}")
            
            # Convert the actions to the thermal zone number and then update the heaters positions
            updated_position = self.convert_action_to_position(position=current_position, 
                                                               action=movement_action)
            # print(f"Updated heater positions: {updated_position}")

            # Update self.position to reflect the new positions
            self.position = updated_position
            # print(f"New heater positions: {self.position}")

            # Process the observation
            # Get the mini-step observation
            self.observation = self.get_mini_step_obs(temp_list=self.temp, 
                                                      position_list=self.position, 
                                                      month=self.month + self.mini_step_track, 
                                                      day=self.day + self.mini_step_track, 
                                                      hour=self.hour + self.mini_step_track)
            # print(f"Mini-step obs: {self.observation}")

            # Reward process
            # Calculate the average temperature pooling reward
            try:
                # Get the current temperature
                if self.hour == 1 and self.day == 1:
                    # First hour of a new month case
                    current_temp = self.get_temp(month=self.month-1, 
                                                day=30, 
                                                hour=24)
                elif self.hour == 1 and self.day != self.start_day:
                    # First hour of a new day case
                    current_temp = self.get_temp(month=self.month, 
                                                day=self.day-1, 
                                                hour=24)
                else:
                    # General case: hours 2-24  
                    current_temp = self.get_temp(month=self.month, 
                                                 day=self.day, 
                                                 hour=self.hour-1)

                # print(f"Current temperature: {current_temp}")
                avg_pooling_reward = self.calculate_avg_temp_pooling_reward(current_positions=current_position, 
                                                                            updated_positions=updated_position,
                                                                            temperatures=current_temp)
                
            except:
                # Set the default average temperature pooling reward
                avg_pooling_reward = 0
            
            # Calculate the mini-step reward (only avg_pooling_reward because no full step reward yet)
            mini_step_reward = avg_pooling_reward
            # print(f"Mini-step total reward: {mini_step_reward:.2f}")

            # print(f"Mini-step reward: {self.reward:.2f}")

            # Set the terminated
            self.terminated = False
            
            # Truncated is typically False unless it has specific truncation conditions
            truncated = False

            # Create an info dictionary (can be empty)
            info = {}

            return (self.observation,   # observation
                    mini_step_reward,   # reward
                    self.terminated,    # terminated
                    truncated,          # truncate
                    info                # info
                    )
        
        # If all mini-steps are completed, reset mini_step and run full step logic
        else:
            # Reset mini-step counter
            # print(f"Mini-step: {self.mini_step+1}")
            self.mini_step = 0
            self.mini_step_track = 0
            # print(f"Reset mini-step: {self.mini_step}")
            # print(f"Reset mini-step track: {self.mini_step_track}")

            # Update the hour
            self.hour += 1
            # Show the current date time
            # print(f"Month: {self.month}, Day: {self.day}, Hour: {self.hour}")

            # Update the action data frame
            print(f"Movement action (main-step): {movement_action}, Energy action (main-step): {energy_action}")
            full_action = movement_action + energy_action
            # print(f"Mini-step full action: {full_action}")
            self.action_df = self.update_action_df(month=self.month, 
                                                day=self.day, 
                                                hour=self.hour, 
                                                action=full_action)

            # Show the current heater positions
            current_position = self.position
            # print(f"Current heater positions: {current_position}")
            
            # Convert the actions to the thermal zone number and then update the heaters positions
            updated_position = self.convert_action_to_position(position=current_position, 
                                                            action=movement_action)
            # print(f"Updated heater positions: {updated_position}")

            # Update self.position to reflect the new positions
            self.position = updated_position
            # print(f"New heater positions: {self.position}")

            # Calculate the average temperature pooling reward
            # Get the current temperature
            if self.hour == 1 and self.day == 1:
                # First hour of a new month case
                current_temp = self.get_temp(month=self.month-1, 
                                            day=30, 
                                            hour=24)
            elif self.hour == 1 and self.day != self.start_day:
                # First hour of a new day case
                current_temp = self.get_temp(month=self.month, 
                                            day=self.day-1, 
                                            hour=24)
            else:
                # General case: hours 2-24  
                current_temp = self.get_temp(month=self.month, 
                                day=self.day, 
                                hour=self.hour-1)

            # print(f"Current temperature: {current_temp}")
            avg_pooling_reward = self.calculate_avg_temp_pooling_reward(current_positions=current_position, 
                                                                        updated_positions=updated_position,
                                                                        temperatures=current_temp)
            
            # Update the position data frame
            self.position_df = self.update_position_df(month=self.month, 
                                                    day=self.day, 
                                                    hour=self.hour, 
                                                    position=updated_position)
            
            # Check the time case
            if self.hour == 1 and self.day == 1:
                # First hour of a new month case
                # Create the greenhouse model
                self.greenhouse_model = self.create_greenhouse(start_month=self.month, 
                                                            end_month=self.month, 
                                                            start_day=self.day, 
                                                            end_day=self.day)
                # Process the current position
                # Add HVAC with controlling thermostat to thermal zones
                self.add_hvac_with_control(position=updated_position, 
                                        energy_action=energy_action)
                # Assign the schedule to HVAC systems
                self.assign_schedule(position=updated_position, 
                                    start_month=self.month, 
                                    end_month=self.month, 
                                    start_day=self.day, 
                                    end_day=self.day, 
                                    hour=self.hour)
                
                # Run the EnergyPlus simulation
                self.run_energyplus()

                # Update the current temperature state
                self.temp_df = self.extract_update_temp(month=self.month,
                                                        day=self.day,
                                                        hour=self.hour)
                
                # Update the current electricity rate state
                self.elec_df = self.extract_update_elec(month=self.month,
                                                        day=self.day,
                                                        hour=self.hour)

                # Calculate the reward
                # Temperature deviation reward
                # Get the previous temperature state
                previous_temp = self.get_temp(month=self.month-1,
                                            day=30,
                                            hour=24)
                # Get the current temperature state
                self.temp_df = self.extract_update_temp(month=self.month,
                                                        day=self.day,
                                                        hour=self.hour)
                current_temp = self.get_temp(month=self.month,
                                            day=self.day,
                                            hour=self.hour)
                # Calculate the temperature deviation reward
                temp_dev_reward = self.get_temp_dev_reward(previous_temperature=previous_temp,
                                                        current_temperature=current_temp,
                                                        weight=10)

            elif self.hour == 1 and self.day != self.start_day:
                # First hour of a new day case
                # Create the greenhouse model
                self.greenhouse_model = self.create_greenhouse(start_month=self.month, 
                                                        end_month=self.month,
                                                        start_day=self.day,
                                                        end_day=self.day)
                # Process the current position
                # Add HVAC with controlling thermostat to thermal zones
                self.add_hvac_with_control(position=updated_position, 
                                        energy_action=energy_action)
                # Assign the schedule to HVAC systems
                self.assign_schedule(position=updated_position, 
                                start_month=self.month, 
                                end_month=self.month, 
                                start_day=self.day, 
                                end_day=self.day, 
                                hour=self.hour)
                
                # Run the EnergyPlus simulation
                self.run_energyplus()

                # Update the current temperature state
                self.temp_df = self.extract_update_temp(month=self.month,
                                                        day=self.day,
                                                        hour=self.hour)
                
                # Update the current electricity rate state
                self.elec_df = self.extract_update_elec(month=self.month,
                                                        day=self.day,
                                                        hour=self.hour)

                # Calculate the reward
                # Temperature deviation reward
                # Get the previous temperature state
                previous_temp = self.get_temp(month=self.month,
                                            day=self.day-1,
                                            hour=24)
                # Get the current temperature state
                self.temp_df = self.extract_update_temp(month=self.month,
                                                        day=self.day,
                                                        hour=self.hour)
                current_temp = self.get_temp(month=self.month,
                                            day=self.day,
                                            hour=self.hour)
                # Calculate the temperature deviation reward
                temp_dev_reward = self.get_temp_dev_reward(previous_temperature=previous_temp,
                                                        current_temperature=current_temp,
                                                        weight=10)
                
            else:
                # General case: hours 2-24
                # Create the greenhouse model
                self.greenhouse_model = self.create_greenhouse(start_month=self.month, 
                                                        end_month=self.month,
                                                        start_day=self.day,
                                                        end_day=self.day)
                # Process the current position
                # # Add HVAC to thermal zones
                # self.add_hvac(updated_position)
                # Add HVAC with controlling thermostat to thermal zones
                self.add_hvac_with_control(position=updated_position, 
                                        energy_action=energy_action)
                # Assign the schedule to HVAC systems
                self.assign_schedule(position=updated_position, 
                                start_month=self.month, 
                                end_month=self.month, 
                                start_day=self.day, 
                                end_day=self.day, 
                                hour=self.hour)

                # Process the previous position
                # Check and process previous position for continuity if different
                previous_position = self.get_position(month=self.month,
                                                    day=self.day,
                                                    hour=self.hour-1)
                # Remove common elements from previous_position
                previous_position = [position for position in previous_position if position not in updated_position]
                # print(f"Previous position: {previous_position}")
                if previous_position:
                    # Add HVAC to previous position thermal zones
                    self.add_hvac(previous_position)
                    # Assign the schedule to previous position HVAC systems
                    self.assign_schedule(position=previous_position,
                                        start_month=self.month,
                                        end_month=self.month,
                                        start_day=self.day,
                                        end_day=self.day,
                                        hour=self.hour-1)
                
                # Run the EnergyPlus simulation
                self.run_energyplus()

                # Update the current temperature state
                self.temp_df = self.extract_update_temp(month=self.month,
                                                        day=self.day,
                                                        hour=self.hour)
                
                # Update the current electricity rate state
                self.elec_df = self.extract_update_elec(month=self.month,
                                                        day=self.day,
                                                        hour=self.hour)

                # Calculate the reward
                # Temperature deviation reward
                # Get the previous temperature state
                previous_temp = self.get_temp(month=self.month,
                                            day=self.day,
                                            hour=self.hour-1)
                # Get the current temperature state
                self.temp_df = self.extract_update_temp(month=self.month,
                                                        day=self.day,
                                                        hour=self.hour)
                current_temp = self.get_temp(month=self.month,
                                            day=self.day,
                                            hour=self.hour)
                # Calculate the temperature deviation reward
                temp_dev_reward = self.get_temp_dev_reward(previous_temperature=previous_temp,
                                                        current_temperature=current_temp,
                                                        weight=10)
            
            # Calculate the movement reward
            movment_reward = self.get_movement_reward(updated_position)

            # Calculate the total reward
            self.reward = temp_dev_reward + avg_pooling_reward + movment_reward

            # Update the reward data frame
            self.reward_df = self.update_reward_df(month=self.month,
                                                day=self.day,
                                                hour=self.hour,
                                                reward=self.reward)
            # print(f"Temperature deviation reward: {temp_dev_reward:.2f}")
            # print(f"Average temperature pooling reward: {avg_pooling_reward:.2f}")
            # print(f"Movement reward: {movment_reward:.2f}")
            # print(f"Total reward: {self.reward:.2f}\n")

            # Combine temperature state, heater positions, month, day, hour into a single observation
            # Get the temperature state
            self.temp = self.get_temp(month=self.month, 
                                    day=self.day, 
                                    hour=self.hour)
            
            # Get the heater position
            updated_position = self.get_position(month=self.month, 
                                                 day=self.day, 
                                                 hour=self.hour)
            
            # Process the observation
            # Get the predicted weather data
            pred_weather = self.predict_next_temp(model_path='gru_model.keras', 
                                                  month=self.month, 
                                                  day=self.day, 
                                                  hour=self.hour)
            
            # Get the first timestep observation with predicted weather data
            self.observation = self.get_obs_with_pred_weather(temp_list=self.temp, 
                                                              pred_weather=pred_weather, 
                                                              position_list=self.position, 
                                                              month=self.month, 
                                                              day=self.day, 
                                                              hour=self.hour)
            # print(f"Main-step obs: {self.observation}")
            
            # Check if the episode has ended. Terminate the episode if the end time is reached
            if self.month == self.end_month and self.day == self.end_day and self.hour==24:
                print(f"----------------Episode Terminated----------------")
                self.terminated = True
                # Save DataFrames to CSV for record-keeping
                try:
                    # Save each DataFrame
                    self.temp_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}temp.csv"), index=False)
                    self.elec_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}elec.csv"), index=False)
                    self.position_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}position.csv"), index=False)
                    self.action_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}action.csv"), index=False)
                    self.reward_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}reward.csv"), index=False)
                    print(f"DataFrames saved to CSV successfully.", "\n--------")
                except Exception as e:
                    print(f"Error saving DataFrames to CSV: {e}")

            # Truncated is typically False unless it has specific truncation conditions
            truncated = False

            # Create an info dictionary (can be empty)
            info = {}
            
            # Update the episode time
            # Update new month
            if self.hour == 24 and self.day == 30:
                self.month += 1
                self.day = 1
                self.hour = 0
            # Update new month for February
            if self.hour == 24 and self.day == 28 and self.month == 2:
                self.month += 1
                self.day = 1
                self.hour = 0
            # Update new day
            if self.hour == 24:
                self.day += 1
                self.hour = 0

            return (self.observation,   # observation
                    self.reward,        # reward
                    self.terminated,    # terminated
                    truncated,          # truncated
                    info                # info
                    )
    
    def init_action_df(self) -> pd.DataFrame:
        """
        Initializes a Pandas DataFrame to store actions throughout the simulation time steps.

        Returns:
        - pd.DataFrame: A DataFrame with columns Month, Day, Hour, and Action.
        """
        action_data = {
            "Month": [self.start_month],
            "Day": [self.start_day],
            "Hour": [self.start_hour],
            "Action": [self.init_action]  # Store the initial action as a list
        }

        self.action_df = pd.DataFrame(action_data)
        return self.action_df

    
    def init_position_df(self) -> pd.DataFrame:
        """
        Initializes a Pandas DataFrame to store position state throughout the simulation time steps.

        Returns:
        - pd.DataFrame: A DataFrame with columns Month, Day, Hour, and Position.
        """
        position_data = {
            "Month": [self.start_month],
            "Day": [self.start_day],
            "Hour": [self.start_hour],
            "Position": [self.init_position]  # Store the initial position as a list
        }

        self.position_df = pd.DataFrame(position_data)
        return self.position_df
    
    def init_reward_df(self) -> pd.DataFrame:
        """
        Initializes a Pandas DataFrame to store reward state throughout the simulation time steps.

        Returns:
        - pd.DataFrame: A DataFrame with columns Month, Day, Hour, and Reward.
        """
        reward_data = {
            "Month": [self.start_month],
            "Day": [self.start_day],
            "Hour": [self.start_hour],
            "Reward": [self.init_reward]  # Store the initial reward
        }

        self.reward_df = pd.DataFrame(reward_data)
        return self.reward_df
    
    def create_greenhouse(self, start_month: int, end_month: int, start_day: int, end_day: int):
        """
        Creates a greenhouse model using the specified parameters.

        Parameters:
        - start_month (int): Starting month of the simulation.
        - end_month (int): Ending month of the simulation.
        - start_day (int): Starting day of the simulation.
        - end_day (int): Ending day of the simulation.

        Returns:
        - model (openstudio.model.Model): The created OpenStudio model object.
        """
        # Create an instance of GreenhouseGeometry with the specified parameters
        greenhouse = GreenhouseGeometry(
            wall_thickness=0.2,
            window_thickness=0.3,
            roof_type="triangle",
            wall_height=4,
            wall_width=4,
            wall_length=4,
            slope=23,
            num_segments=2,
            frame_width=0.05,
            shade_distance_to_roof=3,
            time_step=60,
            number_width=self.num_width,
            number_length=self.num_length,
            max_indoor_temp=60,
            min_indoor_temp=0,
            max_outdoor_temp=60,
            min_outdoor_temp=0,
            max_delta_temp=-5,
            max_wind_speed=30,
            start_month=start_month,
            start_day=start_day,
            end_month=end_month,
            end_day=end_day,
        )

        # Create greenhouse structures
        greenhouse.create_houses()

        # Load or create the OpenStudio model
        try:
            model_file_path = openstudio.path(self.model_path)
            self.greenhouse_model = openstudio.model.Model.load(model_file_path).get()
        except Exception as e:
            print(f"Error loading OpenStudio model from path: {self.model_path}. Exception: {e}")
            raise

        # print("Greenhouse model successfully created.")
        return self.greenhouse_model
    
    # Function to add HVAC systems
    def add_hvac(self, position: list):
        """
        Adds HVAC systems to thermal zones based on the provided heater position list.

        Parameters:
        - position (list): List of thermal zone numbers where HVAC systems will be added.
        """
        # Convert each thermal zone number to its corresponding thermal zone name
        for zone_number in position:
            thermal_zone_name = self.get_thermal_zone_name(zone_number)
            # Add HVAC system to the thermal zone
            try:
                self.add_hvac_to_zone(thermal_zone_name)
                # print(f"HVAC system added to {thermal_zone_name}")
            except ValueError as e:
                print(f"Error adding HVAC system to {thermal_zone_name}: {e}")

    # Function that convert a thermal zone number to the corresponding thermal zone name for a given grid size.
    def get_thermal_zone_name(self, zone_number):
        """
        Converts a thermal zone number to the corresponding thermal zone name.

        Parameters:
        - zone_number (int): The thermal zone number.

        Returns:
        - str: The thermal zone name, e.g., "Greenhouse_Zone_0_0".
        """
        # Check if the zone_number is within the valid range for the grid size
        if not (0 <= zone_number < self.num_width * self.num_length):
            raise ValueError("Zone number must be within the grid size range.")
        # Calculate the row and column based on the specified grid dimensions
        row = zone_number // self.num_width
        col = zone_number % self.num_width

        # Format the thermal zone name
        thermal_zone_name = f"Greenhouse_Zone_{row}_{col}"
    
        return thermal_zone_name
    
    # Function to add HVAC systems to thermal zone name
    def add_hvac_to_zone(self, thermal_zone_name):
        """
        Adds an HVAC system to a specific thermal zone with default heating and cooling settings.

        Parameters:
            model (openstudio.model.Model): The OpenStudio model.
            thermal_zone_name (str): Name of the thermal zone to add HVAC to.

        Raises:
            ValueError: If the specified thermal zone is not found.
        """
        # Retrieve the thermal zone by name
        thermal_zone = self.greenhouse_model.getThermalZoneByName(thermal_zone_name)
        if not thermal_zone.is_initialized():
            raise ValueError(f"Thermal zone '{thermal_zone_name}' not found")
        thermal_zone = thermal_zone.get()

        # Set up zone sizing for heating and cooling parameters
        sizing_zone = thermal_zone.sizingZone()
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(15.0)
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(40.0)
        sizing_zone.setZoneCoolingDesignSupplyAirHumidityRatio(0.008)
        sizing_zone.setZoneHeatingDesignSupplyAirHumidityRatio(0.008)
        sizing_zone.setCoolingDesignAirFlowRate(100)    # m3/s, adjust as needed
        sizing_zone.setHeatingDesignAirFlowRate(100)    # m3/s, adjust as needed
        
        # Create a thermostat and assign it to the thermal zone
        thermostat = openstudio.model.ThermostatSetpointDualSetpoint(self.greenhouse_model)
        thermal_zone.setThermostatSetpointDualSetpoint(thermostat)

        # Create heating and cooling schedules
        heating_sch = self.create_schedule("Heating Schedule", 18.0)
        cooling_sch = self.create_schedule("Cooling Schedule", 19.0)

        # Set the heating and cooling schedule to thermostat
        thermostat.setHeatingSchedule(heating_sch)
        thermostat.setCoolingSchedule(cooling_sch)

        # Create an Air Loop HVAC system for the zone
        air_loop = openstudio.model.AirLoopHVAC(self.greenhouse_model)
        air_loop.setName(f"Air Loop for {thermal_zone_name}")

        # Set up system sizing for the air loop
        sizing_system = air_loop.sizingSystem()
        sizing_system.setTypeofLoadtoSizeOn("Sensible")
        sizing_system.setDesignOutdoorAirFlowRate(0.1)
        sizing_system.setCentralCoolingDesignSupplyAirTemperature(13.0)
        sizing_system.setCentralHeatingDesignSupplyAirTemperature(40.0)
        sizing_system.setSizingOption("Coincident")
        sizing_system.setAllOutdoorAirinCooling(False)
        sizing_system.setAllOutdoorAirinHeating(False)

        # Create and add HVAC components
        # Create the availability schedule (On/Off) with default availability schedule (off)
        hvac_default_availability_schedule = self.create_default_on_off_schedule("HVAC Availability Schedule")

        # Create a supply fan
        fan = self.create_fan()
        fan.setAvailabilitySchedule(hvac_default_availability_schedule)

        # Create a cooling coil
        cooling_coil = self.create_cooling_coil()
        cooling_coil.setAvailabilitySchedule(hvac_default_availability_schedule)

        # Create a heating coil
        heating_coil = self.create_heating_coil()
        heating_coil.setAvailabilitySchedule(hvac_default_availability_schedule)

        # Add components to the air loop
        supply_outlet_node = air_loop.supplyOutletNode()
        heating_coil.addToNode(supply_outlet_node)
        cooling_coil.addToNode(supply_outlet_node)
        fan.addToNode(supply_outlet_node)

        # Add setpoint managers for heating and cooling
        self.add_setpoint_managers(supply_outlet_node, heating_sch, cooling_sch)

        # Create and configure an air terminal for the zone
        air_terminal = openstudio.model.AirTerminalSingleDuctVAVNoReheat(self.greenhouse_model, self.greenhouse_model.alwaysOnDiscreteSchedule())
        air_terminal.setConstantMinimumAirFlowFraction(0.3)

        # Connect the air loop to the zone
        air_loop.addBranchForZone(thermal_zone, air_terminal.to_StraightComponent())

        # Add sizing periods
        openstudio.model.DesignDay(self.greenhouse_model).setRainIndicator(False)
        openstudio.model.DesignDay(self.greenhouse_model).setSnowIndicator(False)

        # Set the simulation control
        sim_control = self.greenhouse_model.getSimulationControl()
        sim_control.setDoZoneSizingCalculation(True)
        sim_control.setDoSystemSizingCalculation(True)
        sim_control.setRunSimulationforSizingPeriods(False)
        sim_control.setRunSimulationforWeatherFileRunPeriods(True)

    # Helper functions
    # Creates a schedule with a constant temperature
    def create_schedule(self, name: str, temperature: float):
        """
        Creates a schedule with a constant temperature.

        Parameters:
        - name (str): Name of the schedule.
        - temperature (float): Constant temperature for the schedule.
        """
        schedule = openstudio.model.ScheduleRuleset(self.greenhouse_model)
        schedule.setName(name)
        day_schedule = schedule.defaultDaySchedule()
        day_schedule.addValue(openstudio.Time(0, 24, 0, 0), float(temperature))
        return schedule
    
    # Creates a variable volumn fan
    def create_fan(self):
        """
        Creates a variable volume fan.
        """
        fan = openstudio.model.FanVariableVolume(self.greenhouse_model)
        fan.setPressureRise(75.0)   # Pascal
        return fan
    
    # Creates a cooling coil
    def create_cooling_coil(self):
        """
        Creates a cooling coil.
        """
        cooling_coil = openstudio.model.CoilCoolingDXSingleSpeed(self.greenhouse_model)
        cooling_coil.setRatedCOP(3.0)
        return cooling_coil
    
    # Creates a heating coil
    def create_heating_coil(self):
        """
        Creates a heating coil.
        """
        heating_coil = openstudio.model.CoilHeatingElectric(self.greenhouse_model)
        heating_coil.setEfficiency(0.99)
        return heating_coil
    
    # Adds setpoint managers to a node
    def add_setpoint_managers(self, node, heating_sch, cooling_sch):
        """
        Adds setpoint managers to a node.
        """
        heating_setpoint_manager = openstudio.model.SetpointManagerScheduled(self.greenhouse_model, heating_sch)
        cooling_setpoint_manager = openstudio.model.SetpointManagerScheduled(self.greenhouse_model, cooling_sch)
        heating_setpoint_manager.addToNode(node)
        cooling_setpoint_manager.addToNode(node)

    # Create a default on/off schedule with all times set to 'off'
    def create_default_on_off_schedule(self, name):
        """
        Creates a default on/off schedule with all times set to 'off'.
        """
        schedule = openstudio.model.ScheduleRuleset(self.greenhouse_model)
        schedule.setName(name)
        day_schedule = schedule.defaultDaySchedule()
        day_schedule.addValue(openstudio.Time(0, 1, 0, 0), 0)
        day_schedule.addValue(openstudio.Time(0, 24, 0, 0), 0)
        return schedule
    
    def assign_schedule(self, position: list, start_month: int, end_month: int, start_day: int, end_day: int, hour: int):
        """
        Assigns a schedule to HVAC systems for the specified thermal zones in the heater position list.

        Parameters:
        - position (list): List of thermal zone numbers where schedules will be assigned.
        - start_month (int): Starting month of the schedule.
        - end_month (int): Ending month of the schedule.
        - start_day (int): Starting day of the schedule.
        - end_day (int): Ending day of the schedule.
        - hour (int): The hour during which the HVAC system will be active.
        """
        for zone_number in position:
            # Convert thermal zone number to thermal zone name
            thermal_zone_name = self.get_thermal_zone_name(zone_number)

            # Get the thermal zone
            thermal_zone = self.greenhouse_model.getThermalZoneByName(thermal_zone_name)
            if not thermal_zone.is_initialized():
                raise ValueError(f"Thermal zone '{thermal_zone_name}' not found")
            thermal_zone = thermal_zone.get()

            # Create a new schedule ruleset
            schedule = openstudio.model.ScheduleRuleset(self.greenhouse_model)
            schedule.setName(f"HVAC_Schedule_{thermal_zone_name}")

            # Create a schedule rule
            schedule_rule = openstudio.model.ScheduleRule(schedule)
            schedule_rule.setApplyMonday(True)
            schedule_rule.setApplyTuesday(True)
            schedule_rule.setApplyWednesday(True)
            schedule_rule.setApplyThursday(True)
            schedule_rule.setApplyFriday(True)
            schedule_rule.setApplySaturday(True)
            schedule_rule.setApplySunday(True)

            # Set the date range for this rule
            start_month_obj = openstudio.MonthOfYear(start_month)
            end_month_obj = openstudio.MonthOfYear(end_month)
            start_date = openstudio.Date(start_month_obj, start_day)
            end_date = openstudio.Date(end_month_obj, end_day)
            schedule_rule.setStartDate(start_date)
            schedule_rule.setEndDate(end_date)

            # Get the day schedule for this rule
            day_schedule = schedule_rule.daySchedule()

            # Set hourly values for this day
            day_schedule.addValue(openstudio.Time(0, hour - 1, 0, 0), 0)  # Off before the hour
            day_schedule.addValue(openstudio.Time(0, hour, 0, 0), 1)  # On during the hour

            # Update the schedule for the HVAC system
            air_loops = thermal_zone.airLoopHVACs()
            if len(air_loops) > 0:
                air_loop = air_loops[0]

                # Update availability schedule for the main air loop
                air_loop.setAvailabilitySchedule(schedule)

                # Update availability schedules for all components in the air loop
                supply_components = air_loop.supplyComponents()
                for component in supply_components:
                    # Handle Fan Variable Volume
                    fan = component.to_FanVariableVolume()
                    if fan.is_initialized():
                        fan.get().setAvailabilitySchedule(schedule)

                    # Handle Cooling Coil DX Single Speed
                    cooling_coil = component.to_CoilCoolingDXSingleSpeed()
                    if cooling_coil.is_initialized():
                        cooling_coil.get().setAvailabilitySchedule(schedule)

                    # Handle Heating Coil Electric
                    heating_coil = component.to_CoilHeatingElectric()
                    if heating_coil.is_initialized():
                        heating_coil.get().setAvailabilitySchedule(schedule)

                # print(f"Schedule successfully assigned to {thermal_zone_name}")
            else:
                print(f"No air loop found for thermal zone '{thermal_zone_name}'.")

    def run_energyplus(self):
        """
        Runs an EnergyPlus simulation with the current OpenStudio model.
        """
        # Save the modified model in OSM format
        osm_path = openstudio.path(self.modified_osm_path)
        self.greenhouse_model.save(osm_path, True)

        # Convert the model to IDF and save it
        forward_translator = openstudio.energyplus.ForwardTranslator()
        workspace = forward_translator.translateModel(self.greenhouse_model)
        idf_path = openstudio.path(self.modified_idf_path)
        workspace.save(idf_path, True)

        # print("Model modification and IDF conversion complete.")

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)

        # Construct the EnergyPlus command
        command = [
            self.energyplus_exe_path,                # EnergyPlus executable
            "--weather", self.epw_file_path,         # Path to the weather file (.epw)
            "--output-directory", self.output_dir,   # Directory to store output
            "--readvars",                            # Instruct EnergyPlus to create variables
            "--output-prefix", self.output_prefix,   # Prefix for output files
            self.modified_idf_path                   # The modified IDF file path
        ]

        # Run EnergyPlus and handle errors
        try:
            subprocess.run(command, check=True, shell=True)
            print("EnergyPlus simulation completed successfully.")
        except subprocess.CalledProcessError as e:
            print(f"EnergyPlus simulation failed with error: {e}")
        except FileNotFoundError as e:
            print(f"EnergyPlus executable not found: {e}")

    def init_temp_df(self) -> pd.DataFrame:
        """
        Initializes the temperature state DataFrame by reading the simulation CSV file and extracting temperature state data.

        Returns:
        - pd.DataFrame: The initialized temperature state DataFrame containing temperature states.
        """
        # Construct the full path to the simulation output file
        simulation_file = os.path.join(self.output_dir, f"{self.output_prefix}out.csv")

        # Read the simulation CSV file
        try:
            data = pd.read_csv(simulation_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Simulation file not found at path: {simulation_file}")

        # Clean the Date/Time column by removing all whitespace
        data['Date/Time'] = data['Date/Time'].str.replace(" ", "", regex=False)

        # Format the Date/Time string to match the cleaned CSV format (no spaces)
        date_str = f"{self.start_month:02}/{self.start_day:02}{self.hour:02}:00:00"
        # print(f"Formatted date string for matching: {date_str}")

        # Filter the DataFrame for the specific hour's data
        time_step_data = data[data['Date/Time'] == date_str]

        # Check if we successfully filtered down to a single row
        if time_step_data.empty:
            raise ValueError(f"No data found for the specified time: {date_str}")

        # Determine the number of columns to extract based on the grid size
        num_columns = self.num_width * self.num_length
        start_col_index = 2  # Assuming temperature columns start from column 'C'

        # Extract temperature state data starting from column C
        temp_state = time_step_data.iloc[0, start_col_index:start_col_index + num_columns].tolist()

        # Create a row for the CSV file with Date/Time and the extracted temperature state
        temp_state_row = [date_str] + temp_state

        # Create headers for the DataFrame
        headers = ["Date/Time"] + data.columns[start_col_index:start_col_index + num_columns].tolist()

        # Create and return the temperature state DataFrame
        self.temp_df = pd.DataFrame([temp_state_row], columns=headers)
        # print(f"Initialized state DataFrame with {len(self.temp_df.columns) - 1} temperature states.")
        return self.temp_df
    
    def get_temp(self, month: int, day: int, hour: int) -> list:
        """
        Extracts and returns the list of temperature states at a specific time step from the temp_df.

        Parameters:
        - month (int): Month for the specific time step.
        - day (int): Day for the specific time step.
        - hour (int): Hour for the specific time step.

        Returns:
        - list: List of temperature states for the specified time step.

        Raises:
        - ValueError: If no matching row is found for the specified time step or temp_df is not initialized.
        """
        if self.temp_df is None:
            raise ValueError("Temperature DataFrame (temp_df) is not initialized.")

        # Format the Date/Time string to match the DataFrame's 'Date/Time' column
        date_str = f"{month:02}/{day:02}{hour:02}:00:00"
        # print(f"Looking for state at Date/Time: {date_str}")

        # Filter the DataFrame for the specific Date/Time
        filtered_row = self.temp_df[self.temp_df["Date/Time"] == date_str]

        # Check if a matching row exists
        if filtered_row.empty:
            raise ValueError(f"No state found for the specified time step: {date_str}")

        # Extract the row and convert it to a list, excluding the 'Date/Time' column
        temp_list = filtered_row.iloc[0, 1:].tolist()

        return temp_list
    
    def get_position(self, month: int, day: int, hour: int) -> list:
        """
        Retrieves the position list for a specific month, day, and hour from the position_df DataFrame.

        Parameters:
        - month (int): The specific month to filter the position.
        - day (int): The specific day to filter the position.
        - hour (int): The specific hour to filter the position.

        Returns:
        - list: The position list corresponding to the specified timestep.

        Raises:
        - ValueError: If no position is found for the specified month, day, and hour.
        """
        if self.position_df is None:
            raise ValueError("Position DataFrame (position_df) is not initialized.")

        # Filter the DataFrame for the specified time step
        position = self.position_df[
            (self.position_df["Month"] == month) &
            (self.position_df["Day"] == day) &
            (self.position_df["Hour"] == hour)
        ]

        # Ensure that a position exists for the specified time step
        if position.empty:
            raise ValueError(f"No position found for Month {month}, Day {day}, Hour {hour}.")

        # Extract the position list
        position_list = position.iloc[0]["Position"]

        return position_list

    def get_observation(self, temp_list: list, position_list: list, month: int, day: int, hour: int) -> np.ndarray:
        """
        Create a flat observation space representation.

        Parameters:
        - temp_list (list): A list of temperature values.
        - position_list (list): A list of heater positions.
        - month (int): The current month (1-12).
        - day (int): The current day of the month (1-31).
        - hour (int): The current hour (0-23).

        Returns:
        - np.ndarray: A flat NumPy array representing the observation space.
        """
        if temp_list is None or position_list is None:
            raise ValueError("Temperature list or position list is not initialized.")

        # Flatten and combine all features into a single array
        self.observation = np.concatenate([
            np.array(temp_list, dtype=np.float32),    # Convert temperatures to a NumPy array
            np.array(position_list, dtype=np.float32), # Convert positions to a NumPy array
            np.array([month, day, hour], dtype=np.float32)  # Add month, day, hour as an array
        ])
        return self.observation
    
    def get_observation_plus_future(self, 
                                    temp_list: list, 
                                    weather_future_1: list,
                                    weather_future_2: list,
                                    weather_future_3: list,
                                    weather_future_4: list,
                                    weather_future_5: list,
                                    weather_future_6: list,
                                    position_list: list, 
                                    month: int, day: int, 
                                    hour: int) -> np.ndarray:
        """
        Create a flat observation space representation.

        Parameters:
        - temp_list (list): A list of temperature values.
        - weater_future_1 (list): A list of weather data at time+1
        - weater_future_2 (list): A list of weather data at time+2
        - weater_future_3 (list): A list of weather data at time+3
        - weater_future_4 (list): A list of weather data at time+4
        - weater_future_5 (list): A list of weather data at time+5
        - weater_future_6 (list): A list of weather data at time+6
        - position_list (list): A list of heater positions.
        - month (int): The current month (1-12).
        - day (int): The current day of the month (1-31).
        - hour (int): The current hour (0-23).

        Returns:
        - np.ndarray: A flat NumPy array representing the observation space.
        """
        if temp_list is None or position_list is None:
            raise ValueError("Temperature list or position list is not initialized.")

        # Flatten and combine all features into a single array
        self.observation = np.concatenate([
            np.array(temp_list, dtype=np.float32),    # Convert temperatures to a NumPy array
            np.array(weather_future_1, dtype=np.float32),
            np.array(weather_future_2, dtype=np.float32),
            np.array(weather_future_3, dtype=np.float32),
            np.array(weather_future_4, dtype=np.float32),
            np.array(weather_future_5, dtype=np.float32),
            np.array(weather_future_6, dtype=np.float32),
            np.array(position_list, dtype=np.float32), # Convert positions to a NumPy array
            np.array([month, day, hour], dtype=np.float32)  # Add month, day, hour as an array
        ])
        return self.observation
    
    def get_obs_with_pred_weather(self, 
                                  temp_list: list, 
                                  pred_weather: list, 
                                  position_list: list, 
                                  month: int, 
                                  day: int, 
                                  hour: int) -> np.ndarray:
        """
        Create a flat observation with predicted weather space representation.

        Parameters:
        - temp_list (list): A list of temperature values.
        - pred_weather (list): A list of predicted weather data from a trained GRU model.
        - position_list (list): A list of heater positions.
        - month (int): The current month (1-12).
        - day (int): The current day of the month (1-31).
        - hour (int): The current hour (0-23).

        Returns:
        - np.ndarray: A flat NumPy array representing the observation space.
        """
        if temp_list is None or position_list is None:
            raise ValueError("Temperature list or position list is not initialized.")
        
        # Flatten and combine all features into a single array
        self.observation = np.concatenate([
            np.array(temp_list, dtype=np.float32),    # Convert temperatures to a NumPy array
            np.array(pred_weather, dtype=np.float32),
            np.array(position_list, dtype=np.float32), # Convert positions to a NumPy array
            np.array([month, day, hour], dtype=np.float32)  # Add month, day, hour as an array
        ])
        return self.observation
    
    def get_mini_step_obs(self, 
                          temp_list: list, 
                          position_list: list, 
                          month: int, 
                          day: int, 
                          hour: int) -> np.ndarray:
        """
        Create a flat observation space representation for mini-step.

        Parameters:
        - temp_list (list): A list of temperature values.
        - position_list (list): A list of heater positions.
        - month (int): The current month (1-12).
        - day (int): The current day of the month (1-31).
        - hour (int): The current hour (0-23).

        Returns:
        - np.ndarray: A flat NumPy array representing the mini-step observation space.
        """
        if temp_list is None or position_list is None:
            raise ValueError("Temperature list or position list is not initialized.")
        
        # Create the default weather list
        dummy_temp = 0
        pred_step = 6
        default_weather = [dummy_temp] * self.grid_size * pred_step
        
        # Flatten and combine all features into a single array
        self.observation = np.concatenate([
            np.array(temp_list, dtype=np.float32),          # Convert temperatures to a NumPy array
            np.array(default_weather, dtype=np.float32),
            np.array(position_list, dtype=np.float32),      # Convert positions to a NumPy array
            np.array([month, day, hour], dtype=np.float32)  # Add month, day, hour as an array
        ])
        return self.observation
    
    def update_action_df(self, month: int, day: int, hour: int, action: list) -> pd.DataFrame:
        """
        Updates the action list in the action_df DataFrame at a specific time.

        Parameters:
        - month (int): The month for the action to be updated.
        - day (int): The day for the action to be updated.
        - hour (int): The hour for the action to be updated.
        - action (list): The new action list to update in the DataFrame.

        Returns:
        - pd.DataFrame: Updated action_df DataFrame.
        """
        if self.action_df is None:
            raise ValueError("Action DataFrame (action_df) is not initialized.")

        # Check if the specified time (month, day, hour) already exists in the DataFrame
        existing_row = self.action_df[
            (self.action_df["Month"] == month) & 
            (self.action_df["Day"] == day) & 
            (self.action_df["Hour"] == hour)
        ]

        if not existing_row.empty:
            # Update the existing row with the new action (store as a list)
            self.action_df.at[existing_row.index[0], "Action"] = action
            # print(f"Updated action for Month {month}, Day {day}, Hour {hour}: {action}")
        else:
            # Add a new row with the specified time and action
            new_row = {"Month": month, "Day": day, "Hour": hour, "Action": action}
            self.action_df = pd.concat([self.action_df, pd.DataFrame([new_row])], ignore_index=True)
            # print(f"Added new action for Month {month}, Day {day}, Hour {hour}: {action}")

        # Return the updated DataFrame
        return self.action_df
    
    def convert_action_to_position(self, position: list, action: list) -> list:
        """
        Converts the action to new positions for the heaters based on the grid layout.

        Parameters:
        - position (list): A list of integers representing the current positions of heaters.
                           Each position is represented as a flattened grid index.
        - action (list): A list of integers representing movement actions for each heater.

        Returns:
        - list: A list of integers representing the updated positions of heaters
                after applying the actions, with boundary and collision handling.
        """
        # Action to (row_change, col_change) mapping
        action_to_delta = {
            0: (0, 0),   # Stay
            1: (1, 0),   # Move Up
            2: (-1, 0),  # Move Down
            3: (0, -1),  # Move Left
            4: (0, 1)    # Move Right
        }

        # Priority fallback actions when collision occurs
        fallback_actions = [0, 1, 2, 3, 4]  # Stay, Up, Down, Left, Right

        if not position or not action:
            return position

        def try_move(pos: int, act: int) -> int:
            """Helper function to calculate new position given an action."""
            row = pos // self.num_width
            col = pos % self.num_width
            delta_row, delta_col = action_to_delta.get(act, (0, 0))

            # Compute new row and column within grid boundaries
            new_row = max(0, min(self.num_length - 1, row + delta_row))
            new_col = max(0, min(self.num_width - 1, col + delta_col))

            return new_row * self.num_width + new_col

        proposed_positions = []
        for current_pos, act in zip(position, action):
            proposed_positions.append(try_move(current_pos, act))

        final_positions = position[:]  # Copy the original positions
        seen_positions = {}  # Dictionary to track occupied positions

        for i, proposed_pos in enumerate(proposed_positions):
            if proposed_pos in seen_positions:
                # Collision detected: Try fallback moves
                resolved = False
                for fallback_act in fallback_actions:
                    new_position = try_move(position[i], fallback_act)
                    if new_position not in seen_positions:
                        proposed_pos = new_position
                        resolved = True
                        break

                # If still unresolved (extremely rare), stay in place
                if not resolved:
                    proposed_pos = position[i]

            final_positions[i] = proposed_pos
            seen_positions[proposed_pos] = i  # Mark the position as occupied

        return final_positions
    
    def update_position_df(self, month: int, day: int, hour: int, position: list) -> pd.DataFrame:
        """
        Updates the position list in the position_df DataFrame at a specific time.

        Parameters:
        - month (int): The month for the position to be updated.
        - day (int): The day for the position to be updated.
        - hour (int): The hour for the position to be updated.
        - position (list): The new position list to update in the DataFrame.

        Returns:
        - pd.DataFrame: Updated position_df DataFrame.
        """
        if self.position_df is None:
            raise ValueError("Position DataFrame (position_df) is not initialized.")
        
        # Check if the specified time (month, day, hour) already exists in the DataFrame
        existing_row = self.position_df[
            (self.position_df["Month"] == month) & 
            (self.position_df["Day"] == day) & 
            (self.position_df["Hour"] == hour)
        ]

        if not existing_row.empty:
            # Update the existing row with the new position (store as a list)
            self.position_df.at[existing_row.index[0], "Position"] = position
            # print(f"Updated position for Month {month}, Day {day}, Hour {hour}: {position}")
        else:
            # Add a new row with the specified time and position
            new_row = {"Month": month, "Day": day, "Hour": hour, "Position": position}
            self.position_df = pd.concat([self.position_df, pd.DataFrame([new_row])], ignore_index=True)
            # print(f"Added new position for Month {month}, Day {day}, Hour {hour}: {position}")
        
        # Return the updated DataFrame
        return self.position_df
    
    def extract_update_temp(self, month: int, day: int, hour: int) -> pd.DataFrame:
        """
        Extracts the temperature state data for a specific time step from the simulation CSV file
        and updates or adds the temperature state list in the temp_df DataFrame.

        Parameters:
        - month (int): The month of the specific time step.
        - day (int): The day of the specific time step.
        - hour (int): The hour of the specific time step.

        Returns:
        - pd.DataFrame: Updated temp_df DataFrame with the new temperature state for the specified time step.

        Raises:
        - FileNotFoundError: If the simulation CSV file is not found.
        - ValueError: If no data is found for the specified time step.
        """
        if self.temp_df is None:
            raise ValueError("Temperature DataFrame (temp_df) is not initialized.")
        
        # Construct the full path to the simulation output file
        simulation_file = os.path.join(self.output_dir, f"{self.output_prefix}out.csv")

        # Read the simulation CSV file
        try:
            data = pd.read_csv(simulation_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Simulation file not found at path: {simulation_file}")

        # Clean the Date/Time column by removing all whitespace
        data['Date/Time'] = data['Date/Time'].str.replace(" ", "", regex=False)

        # Format the Date/Time string to match the cleaned CSV format
        date_str = f"{month:02}/{day:02}{hour:02}:00:00"

        # Filter the DataFrame for the specific time step
        time_step_data = data[data['Date/Time'] == date_str]

        # Check if we successfully filtered down to a single row
        if time_step_data.empty:
            raise ValueError(f"No data found for the specified time: {date_str}")

        # Determine the number of columns to extract based on the grid size
        num_columns = self.num_width * self.num_length
        start_col_index = 2  # Assuming temperature columns start from column 'C'

        # Extract temperature state data starting from column 'C'
        temp_state = time_step_data.iloc[0, start_col_index:start_col_index + num_columns].tolist()

        # Check if the specified time (Date/Time) already exists in the DataFrame
        existing_row = self.temp_df[self.temp_df["Date/Time"] == date_str]

        if not existing_row.empty:
            # Update the existing row with the new temperature state
            for idx, value in enumerate(temp_state):
                self.temp_df.loc[existing_row.index, self.temp_df.columns[idx + 1]] = value
            # print(f"Updated temperature state for Date/Time {date_str}")
        else:
            # Add a new row with the specified time and temperature state
            new_row = {"Date/Time": date_str, **{col: val for col, val in zip(self.temp_df.columns[1:], temp_state)}}
            self.temp_df = pd.concat([self.temp_df, pd.DataFrame([new_row])], ignore_index=True)
            # print(f"Added new temperature state for Date/Time {date_str}")

        # Return the updated DataFrame
        return self.temp_df
    
    def get_temp_dev_reward(self, previous_temperature: list, current_temperature: list, weight: int = 10) -> int:
        """
        Calculate the temperature deviation reward.

        This method measures the reduction in temperature variability within the greenhouse 
        as a result of heater actions. A positive reward indicates a reduction in variability.

        Parameters:
        - previous_temperature (list): Temperatures before heater action.
        - current_temperature (list): Temperatures after heater action.
        - weight (int): Scaling factor for the reward (default is 10).

        Returns:
        - float: The calculated temperature deviation reward.
        """
        if not previous_temperature or not current_temperature:
            raise ValueError("Temperature lists must not be empty.")

        if len(previous_temperature) != len(current_temperature):
            raise ValueError("Previous and current temperature lists must have the same length.")

        # Calculate the standard deviation before heater action
        sigma_before = np.std(previous_temperature)
        # Calculate the standard deviation after the heater action
        sigma_after = np.std(current_temperature)

        # Calculate the temperature deviation reward
        temp_dev_reward = weight * (sigma_before - sigma_after)

        # print(f"Std. Dev Before: {sigma_before}, Std. Dev After: {sigma_after}, Reward: {temp_dev_reward}")

        # Return the reward as an integer
        return temp_dev_reward
    
    def update_reward_df(self, month: int, day: int, hour: int, reward: float) -> pd.DataFrame:
        """
        Updates the reward list in the reward_df DataFrame at a specific time.

        Parameters:
        - month (int): The month for the reward to be updated.
        - day (int): The day for the reward to be updated.
        - hour (int): The hour for the reward to be updated.
        - reward (float): The new reward value to update in the DataFrame.

        Returns:
        - pd.DataFrame: Updated reward_df DataFrame.
        """
        if self.reward_df is None:
            raise ValueError("Reward DataFrame (reward_df) is not initialized.")
        
        # Check if the specified time (month, day, hour) already exists in the DataFrame
        existing_row = self.reward_df[
            (self.reward_df["Month"] == month) & 
            (self.reward_df["Day"] == day) & 
            (self.reward_df["Hour"] == hour)
        ]

        if not existing_row.empty:
            # Update the existing row with the new reward
            self.reward_df.at[existing_row.index[0], "Reward"] = reward
            # print(f"Updated reward for Month {month}, Day {day}, Hour {hour}: {reward}")
        else:
            # Add a new row with the specified time and reward
            new_row = {"Month": month, "Day": day, "Hour": hour, "Reward": reward}
            self.reward_df = pd.concat([self.reward_df, pd.DataFrame([new_row])], ignore_index=True)
            # print(f"Added new reward for Month {month}, Day {day}, Hour {hour}: {reward:.4f}")

        # Return the updated DataFrame
        return self.reward_df
    
    def get_movement_reward(self, current_position: list) -> int:
        """
        Compute the movement reward based on heater positions.
        
        If there is a collision (i.e. at least one duplicate in current_position),
        the reward is -1 times the number of collisions. Here, the number of collisions
        is defined as the total number of heaters minus the number of unique positions.
        
        If there is no collision, the reward is +1 times the number of heaters.
        
        Parameters:
            current_position (list): A list of heater positions (e.g., [18, 21, 42, 45]).
            
        Returns:
            int: The computed movement reward.
        """
        total_heaters = len(current_position)
        unique_positions = set(current_position)
        
        # Number of collisions is the extra heaters at positions already occupied.
        collisions = total_heaters - len(unique_positions)
        
        if collisions > 0:
            movement_reward = -1 * collisions
        else:
            movement_reward = 1 * total_heaters
            
        return movement_reward

    def move_to_lowest(self):
        """
        Selects the thermal zone positions with the lowest temperature based on the number of heaters.

        Returns:
        - list: A list of indices representing the thermal zone positions with the lowest temperature.
        """
        # Convert state list to a numpy array for efficient processing
        state_array = np.array(self.temp)

        # Use np.argpartition to get the indices of the lowest temperatures
        lowest_indices = np.argpartition(state_array, self.num_heater)[:self.num_heater]

        # Retrieve indices of the thermal zones with the lowest temperatures
        selected_zones = lowest_indices.tolist()

        # # Debug: Print selected zones and their temperatures
        # for index in selected_zones:
        #     print(f"The Lowest Thermal Zone {index}: Temperature = {state_array[index]}")

        return selected_zones
    
    def step_move_to_lowest(self):
        """
        Executes one time step to the lowest temperature within the environment.

        Returns:
        - observation (np.ndarray): The updated observation after the step.
        - reward (float): The reward for the action taken.
        - terminated (bool): Whether the episode has ended naturally.
        - truncated (bool): Whether the episode was artificially terminated.
        - info (dict): Additional information about the step.
        """
        # Update the hour
        self.hour += 1
        # print(f"Month: {self.month}, Day: {self.day}, Hour: {self.hour}")

        # Show the current heater positions
        current_position = self.position
        # print(f"Current heater positions: {current_position}")
        
        # Convert the actions to the thermal zone number
        updated_position = self.move_to_lowest()
        # print(f"Updated heater positions: {updated_position}")
        
        # Update the position data frame
        self.position_df = self.update_position_df(month=self.month, 
                                                   day=self.day, 
                                                   hour=self.hour, 
                                                   position=updated_position)
        
        # Check the time case
        if self.hour == 1 and self.day == 1:
            # First hour of a new month case
            # Create the greenhouse model
            self.greenhouse_model = self.create_greenhouse(start_month=self.month, 
                                                           end_month=self.month, 
                                                           start_day=self.day, 
                                                           end_day=self.day)
            # Process the current position
            # Add HVAC to thermal zones
            self.add_hvac(updated_position)
            # Assign the schedule to HVAC systems
            self.assign_schedule(position=updated_position, 
                                 start_month=self.month, 
                                 end_month=self.month, 
                                 start_day=self.day, 
                                 end_day=self.day, 
                                 hour=self.hour)
            
            # Run the EnergyPlus simulation
            self.run_energyplus()

            # Update the current temperature state
            self.temp_df = self.extract_update_temp(month=self.month,
                                                    day=self.day,
                                                    hour=self.hour)
            
            # Update the current electricity rate state
            self.elec_df = self.extract_update_elec(month=self.month,
                                                    day=self.day,
                                                    hour=self.hour)

            # Calculate the reward
            # Temperature deviation reward
            # Get the previous temperature state
            previous_temp = self.get_temp(month=self.month-1,
                                          day=30,
                                          hour=24)
            # Get the current temperature state
            self.temp_df = self.extract_update_temp(month=self.month,
                                                    day=self.day,
                                                    hour=self.hour)
            current_temp = self.get_temp(month=self.month,
                                         day=self.day,
                                         hour=self.hour)
            # Calculate the temperature deviation reward
            temp_dev_reward = self.get_temp_dev_reward(previous_temperature=previous_temp,
                                                       current_temperature=current_temp,
                                                       weight=10)

        elif self.hour == 1 and self.day != self.start_day:
            # First hour of a new day case
            # Create the greenhouse model
            self.greenhouse_model = self.create_greenhouse(start_month=self.month, 
                                                       end_month=self.month,
                                                       start_day=self.day,
                                                       end_day=self.day)
            # Process the current position
            # Add HVAC to thermal zones
            self.add_hvac(updated_position)
            # Assign the schedule to HVAC systems
            self.assign_schedule(position=updated_position, 
                             start_month=self.month, 
                             end_month=self.month, 
                             start_day=self.day, 
                             end_day=self.day, 
                             hour=self.hour)
            
            # Run the EnergyPlus simulation
            self.run_energyplus()

            # Update the current temperature state
            self.temp_df = self.extract_update_temp(month=self.month,
                                                    day=self.day,
                                                    hour=self.hour)
            
            # Update the current electricity rate state
            self.elec_df = self.extract_update_elec(month=self.month,
                                                    day=self.day,
                                                    hour=self.hour)

            # Calculate the reward
            # Temperature deviation reward
            # Get the previous temperature state
            previous_temp = self.get_temp(month=self.month,
                                          day=self.day-1,
                                          hour=24)
            # Get the current temperature state
            self.temp_df = self.extract_update_temp(month=self.month,
                                                    day=self.day,
                                                    hour=self.hour)
            current_temp = self.get_temp(month=self.month,
                                         day=self.day,
                                         hour=self.hour)
            # Calculate the temperature deviation reward
            temp_dev_reward = self.get_temp_dev_reward(previous_temperature=previous_temp,
                                                       current_temperature=current_temp,
                                                       weight=10)
            
        else:
            # General case: hours 2-24
            # Create the greenhouse model
            self.greenhouse_model = self.create_greenhouse(start_month=self.month, 
                                                       end_month=self.month,
                                                       start_day=self.day,
                                                       end_day=self.day)
            # Process the current position
            # Add HVAC to thermal zones
            self.add_hvac(updated_position)
            # Assign the schedule to HVAC systems
            self.assign_schedule(position=updated_position, 
                             start_month=self.month, 
                             end_month=self.month, 
                             start_day=self.day, 
                             end_day=self.day, 
                             hour=self.hour)
            
            # Run the EnergyPlus simulation
            self.run_energyplus()

            # Update the current temperature state
            self.temp_df = self.extract_update_temp(month=self.month,
                                                    day=self.day,
                                                    hour=self.hour)
            
            # Update the current electricity rate state
            self.elec_df = self.extract_update_elec(month=self.month,
                                                    day=self.day,
                                                    hour=self.hour)

            # Calculate the reward
            # Temperature deviation reward
            # Get the previous temperature state
            previous_temp = self.get_temp(month=self.month,
                                          day=self.day,
                                          hour=self.hour-1)
            # Get the current temperature state
            self.temp_df = self.extract_update_temp(month=self.month,
                                                    day=self.day,
                                                    hour=self.hour)
            current_temp = self.get_temp(month=self.month,
                                         day=self.day,
                                         hour=self.hour)
            # Calculate the temperature deviation reward
            temp_dev_reward = self.get_temp_dev_reward(previous_temperature=previous_temp,
                                                       current_temperature=current_temp,
                                                       weight=10)
            

        # Calculate the total reward
        self.reward = 0
        # Update the reward data frame
        self.reward_df = self.update_reward_df(month=self.month,
                                               day=self.day,
                                               hour=self.hour,
                                               reward=self.reward)
        # print(f"Total reward: {self.reward}\n")

        # Combine temperature state, heater positions, month, day, hour into a single observation
        # Get the temperature state
        self.temp = self.get_temp(month=self.month, 
                                  day=self.day, 
                                  hour=self.hour)
        # Get the heater position
        updated_position = self.get_position(month=self.month, 
                                          day=self.day, 
                                          hour=self.hour)
        self.observation = self.get_observation(temp_list=self.temp, 
                                                position_list=updated_position, 
                                                month=self.month, 
                                                day=self.day, 
                                                hour=self.hour)
        
        # Check if the episode has ended. Terminate the episode if the end time is reached
        if self.month == self.end_month and self.day == self.end_day and self.hour==24:
            self.terminated = True
            # Save DataFrames to CSV for record-keeping
            try:
                # Save each DataFrame
                self.temp_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}temp.csv"), index=False)
                self.elec_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}elec.csv"), index=False)
                self.position_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}position.csv"), index=False)
                self.action_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}action.csv"), index=False)
                self.reward_df.to_csv(os.path.join(self.output_dir, f"{self.output_prefix}reward.csv"), index=False)
                print(f"DataFrames saved to CSV successfully.", "\n--------")
            except Exception as e:
                print(f"Error saving DataFrames to CSV: {e}")

        # # Determine if the episode is terminated
        # terminated = self.month == self.end_month and self.day == self.end_day and self.hour == 24

        # Truncated is typically False unless it has specific truncation conditions
        truncated = False

        # Create an info dictionary (can be empty)
        info = {}
        
        # Update the episode time
        # Update new month
        if self.hour == 24 and self.day == 30:
            self.month += 1
            self.day = 1
            self.hour = 0
        # Update new month for February
        if self.hour == 24 and self.day == 28 and self.month == 2:
            self.month += 1
            self.day = 1
            self.hour = 0
        # Update new day
        if self.hour == 24:
            self.day += 1
            self.hour = 0

        return (self.observation,   # observation
                self.reward,        # reward
                self.terminated,    # terminated
                truncated,          # truncated
                info                # info
                )
    
    def calculate_avg_temp_pooling_reward(self, current_positions, updated_positions, temperatures):
        """
        Calculate the average temperature pooling reward dynamically for heaters.

        Parameters:
        - current_positions (list): List of current heater positions (1D index).
        - updated_positions (list): List of updated heater positions (1D index).
        - temperatures (list): List of temperatures for the entire grid.

        Returns:
        - avg_temp_pooling_reward (int): Total reward from heater movements.
        """
        # Convert the temperature list to a 2D grid
        grid = np.array(temperatures).reshape((self.num_length, self.num_width))
        # print(f"Grid: {grid}")
        avg_temp_pooling_reward = 0

        for heater_number, (current, updated) in enumerate(zip(current_positions, updated_positions), start=1):
            # Calculate all mu values dynamically
            mu_heater = self.calculate_avg_for_neighbors(current, grid)
            mu_upward = self.calculate_mu_upward(current, grid)
            mu_downward = self.calculate_mu_downward(current, grid)
            mu_leftward = self.calculate_mu_leftward(current, grid)
            mu_rightward = self.calculate_mu_rightward(current, grid)
            # print(f"AVG Up:{mu_upward:.2f} | Down:{mu_downward:.2f} | Left:{mu_leftward:.2f} | Right:{mu_rightward:.2f} | Heater:{mu_heater:.2f}")

            # Determine the action based on current and updated positions
            current_y, current_x = divmod(current, self.num_width)
            # print(f"Current X: {current_x}, Current Y: {current_y}")
            updated_y, updated_x = divmod(updated, self.num_width)
            # print(f"Update X: {updated_x}, Update Y: {updated_y}")

            if updated_x == current_x and updated_y == current_y:
                action = 0  # Stay
            elif updated_y == current_y + 1 and updated_x == current_x:
                action = 1  # Move upward
            elif updated_y == current_y - 1 and updated_x == current_x:
                action = 2  # Move downward
            elif updated_y == current_y and updated_x == current_x - 1:
                action = 3  # Move leftward
            elif updated_y == current_y and updated_x == current_x + 1:
                action = 4  # Move rightward
            else:
                # Invalid move, skip this iteration
                print(f"Heater {heater_number}: Invalid move from {current} to {updated}. No reward assigned.")
                continue

            # Reward logic based on mu comparisons and actions
            if ((mu_upward < mu_downward and action == 1) or
                (mu_downward < mu_upward and action == 2) or
                (mu_leftward < mu_rightward and action == 3) or
                (mu_rightward < mu_leftward and action == 4) or
                (mu_heater == min(mu_heater, mu_upward, mu_downward, mu_leftward, mu_rightward) and action == 0)):
                avg_temp_pooling_reward += 1  # Correct move
            else:
                avg_temp_pooling_reward -= 1  # Incorrect move

            # # Print debug information
            # print(f"Heater {heater_number}: Action {action}, Current Position {current}, "
            #     f"Updated Position {updated}, Reward {avg_temp_pooling_reward}")

        return avg_temp_pooling_reward

    def calculate_mu_upward(self, position, grid):
        """
        Calculate mu_upward: The average of the averages of all rows below the current position.
        This function computes the average temperature for rows below the current heater's position,
        considering all columns in those rows.

        Parameters:
        - position (int): Current heater position (1D index).
        - grid (2D array): Grid representation of the temperatures.
        - width (int): Number of columns in the grid.
        - length (int): Number of rows in the grid.

        Returns:
        - mu_upward (float): the computed upward average.
        """
        y, x = divmod(position, self.num_width)
        if y < self.num_length - 1:  # Ensure there is at least one row below
            rows_below = grid[y+1:, :]  # All rows below the current row
            row_averages = [np.mean(row) for row in rows_below]
            return np.mean(row_averages)
        return float('inf')  # No rows below

    def calculate_mu_downward(self, position, grid):
        """
        Calculate mu_downward: The average of the averages of all rows above the current position.
        This function computes the average temperature for rows above the current heater's position,
        considering all columns in those rows.

        Parameters:
        - position (int): Current heater position (1D index).
        - grid (2D array): Grid representation of the temperatures.
        - width (int): Number of columns in the grid.
        - length (int): Number of rows in the grid.

        Returns:
        - mu_downward (float): The computed downward average.
        """
        y, x = divmod(position, self.num_width)
        if y > 0:  # Ensure there is at least one row above
            rows_above = grid[:y, :]  # All rows above the current row
            row_averages = [np.mean(row) for row in rows_above]
            return np.mean(row_averages)
        return float('inf')  # No rows above

    def calculate_mu_leftward(self, position, grid):
        """
        Calculate mu_leftward: The average of the averages of the columns to the left of the current position.
        This function computes the average temperature for all columns to the left of the current heater's position.

        Parameters:
        - position (int): Current heater position (1D index).
        - grid (2D array): Grid representation of the temperatures.
        - width (int): Number of columns in the grid.
        - length (int): Number of rows in the grid.

        Returns:
        - mu_leftward (float): The computed leftward average.
        """
        y, x = divmod(position, self.num_width)
        if x > 0:  # Ensure there is at least one column to the left
            column_averages = [np.mean(grid[:, col]) for col in range(x)]  # Columns to the left
            return np.mean(column_averages)
        return float('inf')  # No columns to the left

    def calculate_mu_rightward(self, position, grid):
        """
        Calculate mu_rightward: The average of the averages of the columns to the right of the current position.
        This function computes the average temperature for all columns to the right of the current heater's position.

        Parameters:
        - position (int): Current heater position (1D index).
        - grid (2D array): Grid representation of the temperatures.
        - width (int): Number of columns in the grid.
        - length (int): Number of rows in the grid.

        Returns:
        - mu_rightward (float): The computed mu_rightward average.
        """
        y, x = divmod(position, self.num_width)
        if x < self.num_width - 1:  # Ensure there is at least one column to the right
            column_averages = [np.mean(grid[:, col]) for col in range(x+1, self.num_width)]  # Columns to the right
            return np.mean(column_averages)
        return float('inf')  # No columns to the right

    def calculate_avg_for_neighbors(self, position, grid):
        """
        Calculate the average temperature for all neighbors of the given position, including the heater's position itself.
        Neighbors include adjacent cells in diagonal, horizontal, and vertical directions.

        Parameters:
        - position (int): Current heater position (1D index).
        - grid (2D array): Grid representation of the temperatures.
        - width (int): Number of columns in the grid.
        - length (int): Number of rows in the grid.

        Returns:
        - mu_heater (float): Average temperature of the heater's neighbors, including its own position.
        """
        y, x = divmod(position, self.num_width)
        neighbors = []
        directions = [
            (0, 0),     # Include the heater's position itself
            (1, 0),     # Up
            (-1, 0),    # Down
            (0, -1),    # Left
            (0, 1),     # Right
            (-1, -1), (-1, 1), (1, -1), (1, 1)  # Diagonals
        ]
        for dy, dx in directions:
            ny, nx = y + dy, x + dx
            if 0 <= ny < self.num_length and 0 <= nx < self.num_width:
                neighbors.append(grid[ny, nx])
        # print(f"Neighbors: {neighbors}")
        return np.mean(neighbors) if neighbors else grid[y, x]
    
    def get_future_weather_data(self, current_month, current_day, current_hour, future_step):
        """
        This function retrieves future weather data based on the input parameters.

        Parameters:
        - current_month (int): The current month to match.
        - current_day (int): The current day to match.
        - current_hour (int): The current hour to match.
        - future_step (int): The number of steps into the future to retrieve data.

        Returns:
        - future_weather_data (list): The future weather data as a list.
        """
        # Weather file path
        weather_data_file_path = r"C:\Users\ratht\OneDrive\Ratta Documents\Wageningen University & Research\WUR - MSc Agricultural Biosystems Engineering\INF80436 - MSc Thesis Information Technology\MSc Thesis project\5x5_weather_data_Rotterdam_2023.csv"

        # Read the weather data CSV file
        weather_data = pd.read_csv(weather_data_file_path)
        
        # Match the row with current_month, current_day, and current_hour
        matched_row = weather_data[
            (weather_data['month'] == current_month) &
            (weather_data['day'] == current_day) &
            (weather_data['hour'] == current_hour)
        ]
        
        if matched_row.empty:
            raise ValueError("No matching row found for the specified month, day, and hour.")
        
        # Get the index of the matched row
        matched_index = matched_row.index[0]
        
        # Calculate the index of the future row
        future_index = matched_index + future_step
        
        # Ensure the future_index is within bounds
        if future_index >= len(weather_data):
            raise IndexError("Future step exceeds the available data range.")
        
        # Retrieve the future weather data starting from column 'E'
        future_weather_data = weather_data.iloc[future_index, 4:].tolist()
        
        return future_weather_data
    
    def init_elec_df(self) -> pd.DataFrame:
        """
        Initializes the electricity rate state data frame by reading the simulation CSV file and extracting electricity rate state data.

        Returns:
        - pd.DataFrame: The initialized electricity rate state data frame containing electricity rate states.
        """
        # Construct the full path to the simulation output file
        simulation_file = os.path.join(self.output_dir, f"{self.output_prefix}out.csv")
        
        # Read the simulation CSV file
        try:
            data = pd.read_csv(simulation_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Simulation file not found at path: {simulation_file}")
        
        # Clean the Date/Time column by removing all whitespace
        data['Date/Time'] = data['Date/Time'].str.replace(" ", "", regex=False)
        
        # Format the Date/Time string to match the cleaned CSV format (no spaces)
        date_str = f"{self.start_month:02}/{self.start_day:02}{self.hour:02}:00:00"
        # print(f"Formatted date string for matching: {date_str}")
        
        # Filter the DataFrame for the specific hour's data
        time_step_data = data[data['Date/Time'] == date_str]
        
        # Check if we successfully filtered down to a single row
        if time_step_data.empty:
            raise ValueError(f"No data found for the specified time: {date_str}")
        
        # Determine the number of columns to extract based on the grid size
        # Electricity rate columns start after "Date/Time", "Roof temperature", and zone temperatures.
        start_col_index = 2  # "Date/Time" and "Roof temperature"
        temp_columns = self.num_width * self.num_length
        elec_start_index = start_col_index + temp_columns
        
        # Extract electricity rate state data starting from the calculated column index
        elec_state = time_step_data.iloc[0, elec_start_index:].tolist()
        
        # Create a row for the CSV file with Date/Time and the extracted electricity rate state
        elec_state_row = [date_str] + elec_state
        
        # Create headers for the DataFrame
        headers = ["Date/Time"] + data.columns[elec_start_index:].tolist()
        
        # Create and return the electricity rate state DataFrame
        self.elec_df = pd.DataFrame([elec_state_row], columns=headers)
        # print(f"Initialized electricity rate DataFrame with {len(self.elec_df.columns) - 1} rates.")
        return self.elec_df
    
    def extract_update_elec(self, 
                            month: int, 
                            day: int, 
                            hour: int) -> pd.DataFrame:
        """
        Extracts electricity rate data for a specific time step from the simulation CSV file
        and updates or adds the filtered electricity rate data to the elec_df DataFrame.

        Parameters:
        - month (int): The month of the specific time step.
        - day (int): The day of the specific time step.
        - hour (int): The hour of the specific time step.

        Returns:
        - pd.DataFrame: Updated elec_df DataFrame with the new electricity rate state for the specified time step.

        Raises:
        - FileNotFoundError: If the simulation CSV file is not found.
        - ValueError: If no data is found for the specified time step.
        """
        # Construct the full path to the simulation output file
        simulation_file = os.path.join(self.output_dir, f"{self.output_prefix}out.csv")
        
        # Read the simulation CSV file
        try:
            data = pd.read_csv(simulation_file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Simulation file not found at path: {simulation_file}")
        
        # Clean the Date/Time column by removing all whitespace
        data['Date/Time'] = data['Date/Time'].str.replace(" ", "", regex=False)
        
        # Format the Date/Time string to match the cleaned CSV format
        date_str = f"{month:02}/{day:02}{hour:02}:00:00"
        
        # Filter the DataFrame for the specific time step
        time_step_data = data[data['Date/Time'] == date_str]
        
        # Check if we successfully filtered down to a single row
        if time_step_data.empty:
            raise ValueError(f"No data found for the specified time: {date_str}")
        
        # Determine the start column index for electricity rate data
        # Start after "Date/Time", "Roof temperature", and zone temperatures
        start_col_index = 2  # Columns "Date/Time" and "Roof temperature"
        temp_columns = self.num_width * self.num_length
        elec_start_index = start_col_index + temp_columns
        
        # Extract electricity rate state data starting from the calculated column index
        elec_state = time_step_data.iloc[0, elec_start_index:].tolist()
        
        # Filter the electricity rates: select only the 4 rates larger than 0
        filtered_elec_state = [rate for rate in elec_state if rate > 0]
        
        # Check if the specified time (Date/Time) already exists in the DataFrame
        existing_row = self.elec_df[self.elec_df["Date/Time"] == date_str]
        
        if not existing_row.empty:
            # Update the existing row with the new electricity rate state
            for idx, value in enumerate(filtered_elec_state):
                self.elec_df.loc[existing_row.index, self.elec_df.columns[idx + 1]] = value
            # print(f"Updated electricity rate state for Date/Time {date_str}")
        else:
            # Add a new row with the specified time and electricity rate state
            new_row = {"Date/Time": date_str, **{col: val for col, val in zip(self.elec_df.columns[1:], filtered_elec_state)}}
            self.elec_df = pd.concat([self.elec_df, pd.DataFrame([new_row])], ignore_index=True)
            # print(f"Added new electricity rate state for Date/Time {date_str}")
        
        # Return the updated DataFrame
        return self.elec_df
    
    def get_elec(self, 
                 month: int, 
                 day: int, 
                 hour: int) -> list:
        """
        Extracts and returns the list of electricity rate states at a specific time step from the elec_df.

        Parameters:
        - month (int): Month for the specific time step.
        - day (int): Day for the specific time step.
        - hour (int): Hour for the specific time step.

        Returns:
        - list: List of electricity rate states for the specified time step.

        Raises:
        - ValueError: If no matching row is found for the specified time step.
        """
        # Format the Date/Time string to match the DataFrame's 'Date/Time' column
        date_str = f"{month:02}/{day:02}{hour:02}:00:00"
        
        # Filter the DataFrame for the specific Date/Time
        filtered_row = self.elec_df[self.elec_df["Date/Time"] == date_str]
        
        # Check if a matching row exists
        if filtered_row.empty:
            raise ValueError(f"No electricity rate state found for the specified time step: {date_str}")
        
        # Extract the row and convert it to a list, excluding the 'Date/Time' column
        elec_list = filtered_row.iloc[0, 1:].tolist()

        return elec_list
    
    def convert_energy_lvl_to_thermostat(self, energy_levels: list) -> list:
        """
        Maps energy levels to thermostat temperatures.

        Parameters:
            energy_levels (list of int): A list of energy levels where:
                0 = Low energy level
                1 = Medium energy level
                2 = High energy level

        Returns:
            list of int: A list of thermostat temperatures corresponding to the energy levels.
                Mapping:
                - Energy level 0 -> Thermostat temperature 17.0
                - Energy level 1 -> Thermostat temperature 18.0
                - Energy level 2 -> Thermostat temperature 19.0
        
        Example:
            Input: [0, 1, 2]
            Output: [17.0, 18.0, 21.0]
        """
        # Define a list mapping where the index (level - 1) gives the temperature
        energy_to_temp_mapping = [17.0, 18.0, 19.0]  # 0 -> 17.0, 1 -> 18.0, 2 -> 19.0

        # Map energy levels to thermostat temperatures using direct indexing
        thermostat_temps = [energy_to_temp_mapping[level] for level in energy_levels]

        # Return the resulting list of thermostat temperatures
        return thermostat_temps
    
    # Function to add HVAC systems with controlling
    def add_hvac_with_control(self, position: list, energy_action: list):
        """
        Adds HVAC systems to thermal zones based on the provided heater position list and energy actions.

        Parameters:
        - position (list): List of thermal zone numbers where HVAC systems will be added.
        - energy_action (list): List of thermostat temperatures for each zone's heating and cooling schedules.
        """
        # if len(position) != len(energy_action):
        #     raise ValueError("The length of position and energy_action must match.")
        
        # Convert each thermal zone number to its corresponding thermal zone name
        for zone_number, energy_temp in zip(position, energy_action):
            thermal_zone_name = self.get_thermal_zone_name(zone_number)
            # Add HVAC system to the thermal zone
            try:
                self.add_hvac_with_control_to_zone(thermal_zone_name, energy_temp)
                # print(f"HVAC system added to {thermal_zone_name} with heating thermostat temp {energy_temp}°C")
            except ValueError as e:
                print(f"Error adding HVAC system to {thermal_zone_name}: {e}")

    # Function to add HVAC systems with controlling to thermal zone name
    def add_hvac_with_control_to_zone(self, thermal_zone_name: str, energy_temp: int):
        """
        Adds an HVAC system to a specific thermal zone with heating and cooling settings based on energy_temp.

        Parameters:
        - thermal_zone_name (str): Name of the thermal zone to add HVAC to.
        - energy_temp (float): Heating thermostat temperature. Cooling will be energy_temp + 1.
        
        Raises:
            ValueError: If the specified thermal zone is not found.
        """
        # Retrieve the thermal zone by name
        thermal_zone = self.greenhouse_model.getThermalZoneByName(thermal_zone_name)
        if not thermal_zone.is_initialized():
            raise ValueError(f"Thermal zone '{thermal_zone_name}' not found")
        thermal_zone = thermal_zone.get()

        # Set up zone sizing for heating and cooling parameters
        sizing_zone = thermal_zone.sizingZone()
        sizing_zone.setZoneCoolingDesignSupplyAirTemperature(15.0)
        sizing_zone.setZoneHeatingDesignSupplyAirTemperature(40.0)
        sizing_zone.setZoneCoolingDesignSupplyAirHumidityRatio(0.008)
        sizing_zone.setZoneHeatingDesignSupplyAirHumidityRatio(0.008)
        sizing_zone.setCoolingDesignAirFlowRate(100)    # m3/s, adjust as needed
        sizing_zone.setHeatingDesignAirFlowRate(100)    # m3/s, adjust as needed

        # Create a thermostat and assign it to the thermal zone
        thermostat = openstudio.model.ThermostatSetpointDualSetpoint(self.greenhouse_model)
        thermal_zone.setThermostatSetpointDualSetpoint(thermostat)

        # Create heating and cooling schedules based on energy_temp
        heating_sch = self.create_schedule("Heating Schedule", float(energy_temp))
        cooling_sch = self.create_schedule("Cooling Schedule", float(energy_temp + 1))

        # Set the heating and cooling schedule to thermostat
        thermostat.setHeatingSchedule(heating_sch)
        thermostat.setCoolingSchedule(cooling_sch)

        # Create an Air Loop HVAC system for the zone
        air_loop = openstudio.model.AirLoopHVAC(self.greenhouse_model)
        air_loop.setName(f"Air Loop for {thermal_zone_name}")

        # Set up system sizing for the air loop
        sizing_system = air_loop.sizingSystem()
        sizing_system.setTypeofLoadtoSizeOn("Sensible")
        sizing_system.setDesignOutdoorAirFlowRate(0.1)
        sizing_system.setCentralCoolingDesignSupplyAirTemperature(13.0)
        sizing_system.setCentralHeatingDesignSupplyAirTemperature(40.0)
        sizing_system.setSizingOption("Coincident")
        sizing_system.setAllOutdoorAirinCooling(False)
        sizing_system.setAllOutdoorAirinHeating(False)

        # Create and add HVAC components
        # Create the availability schedule (On/Off) with default availability schedule (off)
        hvac_default_availability_schedule = self.create_default_on_off_schedule("HVAC Availability Schedule")

        # Create a supply fan
        fan = self.create_fan()
        fan.setAvailabilitySchedule(hvac_default_availability_schedule)

        # Create a cooling coil
        cooling_coil = self.create_cooling_coil()
        cooling_coil.setAvailabilitySchedule(hvac_default_availability_schedule)

        # Create a heating coil
        heating_coil = self.create_heating_coil()
        heating_coil.setAvailabilitySchedule(hvac_default_availability_schedule)

        # Add components to the air loop
        supply_outlet_node = air_loop.supplyOutletNode()
        heating_coil.addToNode(supply_outlet_node)
        cooling_coil.addToNode(supply_outlet_node)
        fan.addToNode(supply_outlet_node)

        # Add setpoint managers for heating and cooling
        self.add_setpoint_managers(supply_outlet_node, heating_sch, cooling_sch)

        # Create and configure an air terminal for the zone
        air_terminal = openstudio.model.AirTerminalSingleDuctVAVNoReheat(self.greenhouse_model, self.greenhouse_model.alwaysOnDiscreteSchedule())
        air_terminal.setConstantMinimumAirFlowFraction(0.3)

        # Connect the air loop to the zone
        air_loop.addBranchForZone(thermal_zone, air_terminal.to_StraightComponent())

        # Add sizing periods
        openstudio.model.DesignDay(self.greenhouse_model).setRainIndicator(False)
        openstudio.model.DesignDay(self.greenhouse_model).setSnowIndicator(False)

        # Set the simulation control
        sim_control = self.greenhouse_model.getSimulationControl()
        sim_control.setDoZoneSizingCalculation(True)
        sim_control.setDoSystemSizingCalculation(True)
        sim_control.setRunSimulationforSizingPeriods(False)
        sim_control.setRunSimulationforWeatherFileRunPeriods(True)

    def predict_next_temp(self,
                          model_path: str='gru_model.keras', 
                          month: int=1, 
                          day: int=1, 
                          hour: int=1)-> list:
        """
        Predict the next 6 steps temperature using a trained GRU model.

        Parameters:
            model_path (str): Path to the trained GRU model (.keras).
            month (int): Current month (1-12).
            day (int): Current day (1-31).
            hour (int): Current hour (1-24).

        Returns:
            list: A flattened list of next 6 steps temperature predictions.
        """
        # Load the trained GRU model
        model = load_model(model_path)

        # Prepare the input data
        # Input format: [month, day, hour], shaped as (1, 1, 3) for GRU input
        input_data = np.array([[month, day, hour]])  # Shape (1, 3)
        input_data = input_data.reshape(1, 1, 3)    # Reshape to (1, 1, features)

        # Make predictions
        predictions = model.predict(input_data)  # Shape (1, 3, 64)

        # Flatten the predictions for 6 steps
        predicted_temp = predictions.flatten().tolist()  # Flatten and convert to list
        # print(f"Length of prediction: {len(predicted_temp)}")

        return predicted_temp
    
    # Precompute action mapping: Discrete -> MultiDiscrete
    def decode_action(self, action_int):
        """
        Decode a flat action integer into a list of per-heater actions.
        
        Each heater has 39 possible actions (13 movement actions * 3 energy levels),
        so we perform a base-39 decomposition.
        
        Parameters:
            action_int (int): The flat action integer.
            
        Returns:
            list: A list of per-heater action integers.
        """
        per_heater_actions = []
        for _ in range(self.num_heater):
            per_heater_actions.append(action_int % self.num_action_per_heater)
            action_int //= self.num_action_per_heater
        # The actions are extracted in reverse order, so reverse them:
        per_heater_actions.reverse()
        return per_heater_actions
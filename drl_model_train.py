"""
Copyright Statement:

Author: Ratta Chindasilpa
Author's email: raththar@hotmail.com

This code is part of the master's thesis of Ratta Chindasilpa, "Deep Reinforcement Learning for Adaptive Control of Heater Position and Heating Power in a Smart Greenhouse," 
developed at Wageningen University and Research.
"""

# Import necessary libraries
from greenhouse_geometry import GreenhouseGeometry
from greenhouse_environment import GreenhouseEnv
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv

# Initialize the greenhouse geometry
greenhouse = GreenhouseGeometry(wall_thickness=0.2,         # Wall thickness (in meters) 
                                window_thickness=0.3,       # Window thickness (in meters) 
                                roof_type="triangle",       # Type of roof (e.g., "triangle", "flat", etc.) 
                                wall_height=4,              # Wall height (in meters) wall_width=4,  # Wall width (in meters) 
                                wall_length=4,              # Wall length (in meters) slope=23,  # Roof slope (in degrees) 
                                num_segments=2,             # Number of segments used for geometry 
                                frame_width=0.05,           # Frame width (in meters) 
                                shade_distance_to_roof=3,   # Distance from shade to the roof (in meters) 
                                time_step=60,               # Time step for the simulation (60/time_step equals to the number of interval in one hour) 
                                number_width=8,             # Number of grid sections along width 
                                number_length=8,            # Number of grid sections along length 
                                max_indoor_temp=60,         # Maximum allowable indoor temperature 
                                min_indoor_temp=-10,        # Minimum allowable indoor temperature 
                                max_outdoor_temp=60,        # Maximum allowable outdoor temperature 
                                min_outdoor_temp=-10,       # Minimum allowable outdoor temperature 
                                max_delta_temp=-5,          # Maximum temperature difference allowed between indoor and outdoor 
                                max_wind_speed=30,          # Maximum wind speed allowed (in m/s) 
                                start_month=2,              # Starting month of the simulation period 
                                start_day=28,               # Starting day of the simulation period 
                                end_month=2,                # Ending month of the simulation period 
                                end_day=28                  # Ending day of the simulation period 
                                )

# Create the greenhouse geometry
greenhouse.create_houses()

# Initialize the greenhouse environment
env = GreenhouseEnv(start_month = 2, 
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
                    modified_osm_path = "energyplus_data/model_files/modified_model.osm",    
                    modified_idf_path = "energyplus_data/model_files/modified_model.idf"
                    )

# Wrap the Environment for PPO
env = Monitor(env, filename="reward")
env = DummyVecEnv([lambda: env])

# Initialize the PPO Agent
model = PPO(
    policy="MlpPolicy",         # Use a Multi-Layer Perceptron policy for both the actor and the critic
    env=env,                    # Greenhouse environment
    learning_rate=3e-4,         # Learning rate
    n_steps=115,                # Number of steps to run for each environment per update (rollout length)
    batch_size=115,             # Batch size for each gradient update
    n_epochs=10,                # Number of passes (epochs) over the data per update
    gamma=0.99,                 # Discount factor for future rewards
    gae_lambda=0.95,            # Lambda for Generalized Advantage Estimation (GAE)
    clip_range=0.2,             # Clipping parameter for PPO’s objective to ensure small policy updates
    ent_coef=0.0,               # Entropy coefficient (encourages exploration)
    vf_coef=0.5,                # Coefficient for the value function loss term
    max_grad_norm=0.5,          # Maximum gradient norm (for gradient clipping)
    verbose=1                   # Verbosity level (1 prints training information)
)

# Train the PPO agent
total_timesteps = 115 * 300  # 115 steps per episode * 300 episodes
model.learn(total_timesteps=total_timesteps, progress_bar=True)

# Save the trained model
model.save("ppo_model")
print("Model saved successfully as 'ppo_model.zip'")
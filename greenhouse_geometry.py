"""
Copyright Statement:

Author: Ratta Chindasilpa
Author's email: raththar@hotmail.com

This code is part of the master's thesis of Ratta Chindasilpa, "Deep Reinforcement Learning for Adaptive Control of Heater Position and Heating Power in a Smart Greenhouse", 
developed at Wageningen University and Research.
This code is referenced and developed based on the original code from the project "GreenLightPlus" by Daidai Qiu.
"""

# Import necessary libraries
import openstudio
import numpy as np
import os
import csv

# Define the GreenhouseGeometry class
class GreenhouseGeometry:
    def __init__(
        self,
        wall_thickness=0.2,  # Wall thickness (in meters)
        window_thickness=0.3,  # Window thickness (in meters)
        roof_type="triangle",  # Type of roof (e.g., "triangle", "flat", etc.)
        wall_height=4,  # Wall height (in meters)
        wall_width=4,  # Wall width (in meters)
        wall_length=4,  # Wall length (in meters)
        slope=23,  # Roof slope (in degrees)
        num_segments=2,  # Number of segments used for geometry
        frame_width=0.05,  # Frame width (in meters)
        shade_distance_to_roof=3,  # Distance from shade to the roof (in meters)
        time_step=60,  # Time step for the simulation (60/time_step equals to the number of interval in one hour)
        number_width=8,  # Number of grid sections along width
        number_length=8,  # Number of grid sections along length
        max_indoor_temp=60,  # Maximum allowable indoor temperature
        min_indoor_temp=-10,  # Minimum allowable indoor temperature
        max_outdoor_temp=60,  # Maximum allowable outdoor temperature
        min_outdoor_temp=-10,  # Minimum allowable outdoor temperature
        max_delta_temp=-5,  # Maximum temperature difference allowed between indoor and outdoor
        max_wind_speed=30,  # Maximum wind speed allowed (in m/s)
        start_month=1,  # Starting month of the simulation period
        start_day=1,  # Starting day of the simulation period
        end_month=12,  # Ending month of the simulation period
        end_day=31  # Ending day of the simulation period
    ):

        ###############################################################
        # Instantiate variables
        ###############################################################

        # Assign roof type, wall dimensions, and other geometry-related parameters
        self.roof_type = roof_type
        self.wall_height = wall_height
        self.wall_width = wall_width
        self.wall_length = wall_length
        self.number_width = number_width
        self.number_length = number_length
        self.slope = slope

        # Store the number of geometric segments
        self.num_segments = num_segments

        # Calculate the roof height based on the wall width and slope
        self.roof_height_relative_to_wall = self.calculate_roof_height_relative_to_wall(self.wall_width, self.slope)

        # Store the frame width and time step for the simulation
        self.frame_width = frame_width
        self.time_step = time_step

        # Calculate the total floor area of the greenhouse
        self.floor_area = self.wall_width * self.wall_length * self.number_length * self.number_width

        ###############################################################
        # Calculate the area of ​​the roof windows
        ###############################################################

        # Calculate the roof ventilation area, split between left and right sides
        self.roof_ventilation_area_left = 0.1169 * self.floor_area / 2
        self.roof_ventilation_area_right = 0.1169 * self.floor_area / 2

        # Store the distance from the shading system to the roof
        self.shade_distance_to_roof = shade_distance_to_roof
        self.roof_volume = 0  # Initial roof volume (can be calculated later)

        # Set the temperature and wind speed limits for the simulation
        self.max_indoor_temp = max_indoor_temp
        self.min_indoor_temp = min_indoor_temp
        self.max_outdoor_temp = max_outdoor_temp
        self.min_outdoor_temp = min_outdoor_temp
        self.max_delta_temp = max_delta_temp
        self.max_wind_speed = max_wind_speed

        # Store the start and end dates of the simulation
        self.start_month = start_month
        self.start_day = start_day
        self.end_month = end_month
        self.end_day = end_day

        # Calculate the surface area of the sloped roof
        self.calculate_surface_area_slope(self.wall_length * self.number_length,
                                        self.wall_width * self.number_length, 
                                        self.roof_height_relative_to_wall)

        # # Print out some basic calculated values
        print(f"Roof height relative to wall: {self.roof_height_relative_to_wall:.2f} m")
        print(f"Floor area: {self.floor_area} m²")
        # print(f"Roof ventilation area left: {self.roof_ventilation_area_left:.2f} m²")
        # print(f"Roof ventilation area right: {self.roof_ventilation_area_right:.2f} m²")

        # Based on the type of roof, assign a method to calculate the z-value for geometry
        if self.roof_type == "half_circle":
            self.roof_height_relative_to_wall = self.wall_width / 2  # Set roof height for a half-circle roof
            self.z_value = self.z_value_half_circle  # Use the method for half-circle geometry
        elif self.roof_type == "triangle":
            self.z_value = self.z_value_triangle  # Use the method for triangular roof geometry
        elif self.roof_type == "flat_arch":
            self.z_value = self.z_value_flat_arch  # Use the method for flat arch geometry
        elif self.roof_type == "gothic_arch":
            self.z_value = self.z_value_gothic_arch  # Use the method for gothic arch geometry
        elif self.roof_type == "sawtooth":
            self.z_value = self.z_value_sawtooth  # Use the method for sawtooth roof geometry
        elif self.roof_type == "sawtooth_arch":
            self.z_value = self.z_value_sawtooth_arch  # Use the method for sawtooth arch geometry

        # Create a new OpenStudio model (this will be used for energy simulation)
        self.model = openstudio.model.Model()

        ###############################################################
        # Create a run period
        ###############################################################

        # Set up the run period for the simulation, including start and end dates
        run_period = self.model.getRunPeriod()
        run_period.setBeginMonth(self.start_month)  # Set the starting month
        run_period.setBeginDayOfMonth(self.start_day)  # Set the starting day
        run_period.setEndMonth(self.end_month)  # Set the ending month
        run_period.setEndDayOfMonth(self.end_day)  # Set the ending day

        # Configure the run period to use settings from a weather file
        run_period.setUseWeatherFileHolidays(True)  # Use holidays from the weather file
        run_period.setUseWeatherFileDaylightSavings(True)  # Use daylight savings from the weather file
        run_period.setApplyWeekendHolidayRule(True)  # Apply weekend and holiday rules
        run_period.setUseWeatherFileRainInd(True)  # Use rainfall data from the weather file
        run_period.setUseWeatherFileSnowInd(True)  # Use snowfall data from the weather file

        ###############################################################
        # Setting output variables for the simulation
        ###############################################################
        
        timestep = self.model.getTimestep()
        timestep.setNumberOfTimestepsPerHour(int(60/self.time_step))

        # Add output variables for zone air temperature and other conditions
        variables = [
            "Zone Air Temperature", 
            # "Zone Mean Air Temperature", 
            # "Zone Air Relative Humidity", 
            # "Zone Outdoor Air Drybulb Temperature", 
            # "Zone Outdoor Air Wetbulb Temperature", 
            # "Zone Outdoor Air Wind Speed", 
            # "Zone Ventilation Mass Flow Rate", 
            # "Zone Ventilation Mass", 
            # "Zone Ventilation Air Change Rate", 
            # "Zone Air Humidity Ratio", 
            # "Site Outdoor Air Drybulb Temperature",
            # "Heating Coil Heating Energy",          # [J] Total air heating energy provided by the coil
            # "Heating Coil Heating Rate",            # [W] Rate of air heating
            # "Heating Coil Electricity Energy",      # [J] Total electric energy used by the heating coil
            "Heating Coil Electricity Rate",        # [W] Power usage of the electric heating coil
            # "Cooling Coil Electricity Energy",      # [J] Total electric energy used by the cooling coil
            # "Cooling Coil Electricity Rate",        # [W] Power usage of the electric cooling coil
            # "Cooling Coil Total Cooling Energy",    # [J] Total air cooling energy provided by the coil
            # "Cooling Coil Total Cooling Rate"       # [W] Rate of air cooling
        ]
        
        # Add the output variables to the model
        for var in variables:
            output_var = openstudio.model.OutputVariable(var, self.model)
            output_var.setKeyValue("*")  # Apply to all zones
            output_var.setReportingFrequency("Timestep")

        ###############################################################
        # Create spaces and thermal zones for the greenhouse
        ###############################################################

        # Initialize lists to hold multiple spaces and thermal zones
        self.spaces = []
        self.space_types = []
        self.thermal_zones = []

        # Loop through the number of grid sections (number_width * number_length) and create individual spaces and zones
        for w in range(self.number_width):
            for l in range(self.number_length):
                space_name = f"Greenhouse_Space_{w}_{l}"
                space_type_name = f"Greenhouse_Type_{w}_{l}"
                thermal_zone_name = f"Greenhouse_Zone_{w}_{l}"

                # Create space and space type for each grid section
                space, space_type = self.create_space_and_type(space_name, space_type_name)
                self.spaces.append(space)
                self.space_types.append(space_type)

                # Create a thermal zone for each space
                thermal_zone = self.create_thermal_zone(thermal_zone_name, space)
                self.thermal_zones.append(thermal_zone)

        # Create the roof space and its space type (remains the same for the roof)
        self.space_roof, self.space_type_roof = self.create_space_and_type(
            "Greenhouse Space Roof", "Greenhouse Type Roof")

        # Create the thermal zone for the roof of the greenhouse
        self.thermal_zone_roof = self.create_thermal_zone(
            "Greenhouse Zone Roof", self.space_roof)

        ###############################################################
        # Create default construction set for materials and building components
        ###############################################################

        # Create a new default construction set for the greenhouse
        self.default_construction_set = openstudio.model.DefaultConstructionSet(self.model)
        self.default_construction_set.setName("Greenhouse Construction Set")

        # Apply the default construction set to all spaces and space types
        for space, space_type in zip(self.spaces, self.space_types):
            space.setDefaultConstructionSet(self.default_construction_set)
            space_type.setDefaultConstructionSet(self.default_construction_set)

        # Apply the default construction set to the roof space and space type
        self.space_roof.setDefaultConstructionSet(self.default_construction_set)
        self.space_type_roof.setDefaultConstructionSet(self.default_construction_set)

        ###############################################################
        # Define materials for floor, walls, windows, and shading systems
        ###############################################################

        # Define the material properties for the greenhouse floor
        floor_properties = {
            "thermal_conductivity": 1.3,  # Thermal conductivity of the floor material (e.g., concrete)
            "density": 2300,  # Density of the material (in kg/m³)
            "specific_heat": 0.8  # Specific heat capacity (in J/kgK)
        }

        # Create a floor material object
        self.floor_material = self.create_material(
            "Floor Material", "StandardOpaqueMaterial", floor_properties)

        # Define the material properties for the greenhouse wall frame (e.g., steel)
        frame_properties = {
            "thermal_conductivity": 50,  # Thermal conductivity of steel (in W/mK)
            "density": 7850,  # Density of steel (in kg/m³)
            "specific_heat": 500,  # Specific heat capacity of steel (in J/kgK)
            "thickness": 0.2  # Thickness of the steel frame (in meters)
        }

        # Create a wall material object (steel frame)
        self.frame_material = self.create_material(
            "Wall Material", "StandardOpaqueMaterial", frame_properties)

        # Define the material properties for the shading system
        shading_properties = {
            "thermal_conductivity": 0.04,  # Thermal conductivity of the shading material (low to prevent heat transfer)
            "thickness": 0.01,  # Thickness of the shading material (in meters)
            "density": 1.5,  # Density of the shading material (in kg/m³)
            "specific_heat": 1000,  # Specific heat capacity of the shading material (in J/kgK)
            "thermal_absorptance": 0.6,  # Thermal absorptance (fraction of thermal radiation absorbed)
            "solar_absorptance": 0.6,  # Solar absorptance (fraction of solar radiation absorbed)
            "visible_absorptance": 0.6  # Visible light absorptance (fraction of visible light absorbed)
        }

        # Create a shading material object
        self.shading_material = self.create_material(
            "Shading Material", "StandardOpaqueMaterial", shading_properties)

        # Define the material properties for the greenhouse windows (e.g., glass)
        window_properties = {
            "u_factor": 5.8,  # U-factor (thermal transmittance) of the window
            "solar_transmittance": 0.99,  # Fraction of solar radiation transmitted through the glass
            "visible_transmittance": 0.8,  # Fraction of visible light transmitted through the glass
            "thickness": 4.0  # Thickness of the window (in meters)
        }

        # Create a window material object
        self.window_material = self.create_material(
            "Window Material", "SimpleGlazing", window_properties)

        ###############################################################
        # Create construction components using the materials defined above
        ###############################################################

        # Create a construction component for the floor
        self.floor_construction = self.create_construction(
            "Floor Construction", [self.floor_material])

        # Create a construction component for the walls (steel frame)
        self.frame_construction = self.create_construction(
            "Wall Construction", [self.frame_material])

        # Create construction components for the windows (main and roof)
        self.window_construction_main = self.create_construction(
            "Window Construction Main", [self.window_material])
        self.window_construction_roof = self.create_construction(
            "Window Construction Roof", [self.window_material])

        # Create construction components for roof windows (left and right)
        self.window_construction_roof_left = self.create_construction(
            "Window Construction Roof Left", [self.window_material])
        self.window_construction_roof_right = self.create_construction(
            "Window Construction Roof Right", [self.window_material])

        # Create a construction component for the shading system
        self.shading_construction = self.create_construction(
            "Shading Construction", [self.shading_material])

        ###############################################################
        # Add Air Mixing Between Adjacent Zones
        ###############################################################
        
        # Call the function that implements air mixing between zones
        self.add_air_mixing_between_zones()


    ###############################################################
    # Add the air mixing method to handle air exchange between adjacent zones
    ###############################################################

    def add_air_mixing_between_zones(self):
        """
        Add air mixing between adjacent thermal zones to simulate heat transfer via air exchange
        in both horizontal (left-right) and vertical (up-down) directions.
        """
        number_zones = len(self.thermal_zones)  # Total number of thermal zones
        mixing_flow_rate = 0.1  # Airflow rate (m^3/s) - adjust as needed

        # 1. Horizontal air mixing (left-right) within the same row of zones
        for w in range(self.number_width):
            for l in range(self.number_length):
                current_zone = self.thermal_zones[w * self.number_length + l]  # Current zone

                # Horizontal air mixing with the next zone to the right
                if l < self.number_length - 1:  # Check if there is a zone to the right
                    right_zone = self.thermal_zones[w * self.number_length + (l + 1)]  # Adjacent right zone

                    # Create air mixing from current zone to right zone
                    zone_mixing_1 = openstudio.model.ZoneMixing(current_zone)
                    zone_mixing_1.setName(f"Air Mixing from Zone ({w}, {l}) to ({w}, {l+1})")
                    zone_mixing_1.setDesignFlowRate(mixing_flow_rate)
                    zone_mixing_1.setSourceZone(right_zone)

                    # Reverse mixing (from right zone to current zone)
                    zone_mixing_2 = openstudio.model.ZoneMixing(right_zone)
                    zone_mixing_2.setName(f"Air Mixing from Zone ({w}, {l+1}) to ({w}, {l})")
                    zone_mixing_2.setDesignFlowRate(mixing_flow_rate)
                    zone_mixing_2.setSourceZone(current_zone)

                # 2. Vertical air mixing (up-down) between rows
                if w < self.number_width - 1:  # Check if there is a zone below (next row)
                    below_zone = self.thermal_zones[(w + 1) * self.number_length + l]  # Zone below

                    # Create air mixing from current zone to below zone
                    zone_mixing_3 = openstudio.model.ZoneMixing(current_zone)
                    zone_mixing_3.setName(f"Air Mixing from Zone ({w}, {l}) to ({w+1}, {l})")
                    zone_mixing_3.setDesignFlowRate(mixing_flow_rate)
                    zone_mixing_3.setSourceZone(below_zone)

                    # Reverse mixing (from below zone to current zone)
                    zone_mixing_4 = openstudio.model.ZoneMixing(below_zone)
                    zone_mixing_4.setName(f"Air Mixing from Zone ({w+1}, {l}) to ({w}, {l})")
                    zone_mixing_4.setDesignFlowRate(mixing_flow_rate)
                    zone_mixing_4.setSourceZone(current_zone)
        
    def calculate_roof_height_relative_to_wall(self, roof_width, angle_degrees):
        """
        Calculate the height of the roof relative to the wall based on roof width and slope angle.

        This function calculates the height of the roof above the wall using trigonometry. 
        Given the width of the roof and the slope angle (in degrees), the function converts 
        the slope angle to radians and computes the vertical height of the roof from the top 
        of the wall to the roof peak.

        Args:
            roof_width (float): The total width of the roof (in meters).
            angle_degrees (float): The slope angle of the roof in degrees.

        Returns:
            float: The height of the roof relative to the wall (in meters).
        """
        # Convert the slope (angle in degrees) to radians
        angle_radians = np.radians(angle_degrees)

        # Calculate the height of the roof relative to the wall using trigonometry.
        # The formula used is: tan(angle) * (roof_width / 2)
        # This represents the height from the wall to the roof's peak.
        roof_height_relative_to_wall = np.tan(angle_radians) * (roof_width / 2)

        # Return the calculated roof height
        return roof_height_relative_to_wall
    
    def calculate_trapezoidal_prism_volume(self, top_base, bottom_base, height_trapezoid, height_prism):
        """
        Calculate the volume of a trapezoidal prism.

        This function calculates the volume of a trapezoidal prism by first calculating the area 
        of the trapezoid and then multiplying it by the height of the prism (depth). The trapezoid's 
        area is calculated using the formula: 
        (top_base + bottom_base) * height_trapezoid / 2.

        Args:
            top_base (float): The length of the top base of the trapezoid (in meters).
            bottom_base (float): The length of the bottom base of the trapezoid (in meters).
            height_trapezoid (float): The height of the trapezoid, which is the perpendicular distance 
                                    between the top and bottom bases (in meters).
            height_prism (float): The height of the trapezoidal prism, which represents the depth 
                                (or length) of the prism (in meters).

        Returns:
            float: The volume of the trapezoidal prism (in cubic meters).
        """
        # Calculate the area of the trapezoid using the formula: 
        # (top_base + bottom_base) * height_trapezoid / 2
        area_trapezoid = (top_base + bottom_base) * height_trapezoid / 2

        # Calculate the volume of the prism by multiplying the area of the trapezoid by the height (depth) of the prism
        volume_prism = area_trapezoid * height_prism

        # Return the calculated volume of the trapezoidal prism
        return volume_prism
    
    def calculate_window_area(self):
        """
        Calculate the area of the roof windows.

        This function calculates the dimensions and area of the windows on the sloped roof of the greenhouse.
        It computes the width and length of each window and the total window area using the wall dimensions,
        roof height, number of windows, and frame width.

        Args:
            None. (Uses class attributes such as wall_width, wall_length, num_segments, roof_height_relative_to_wall, frame_width)

        Returns:
            float: The area of a single roof window in square meters.
        """
        
        # Print the width, length, and height of the walls and roof for debugging purposes
        print(f"Wall width: {self.wall_width:.2f} m")
        print(f"Wall length: {self.wall_length:.2f} m")
        print(f"Roof height relative to wall: {self.roof_height_relative_to_wall:.2f} m")

        # Calculate the length of the sloped roof using Pythagoras' theorem
        # sqrt((half of wall width)^2 + (roof height)^2) gives the sloped length
        roof_slope_length = np.sqrt((self.wall_width / 2) ** 2 + self.roof_height_relative_to_wall ** 2)

        # Calculate the width of each window based on the number of segments (windows) and frame width
        # Adjust window width to account for the frame width
        window_width = (roof_slope_length / self.num_segments * 2) - (2 * self.frame_width)

        # Calculate the length of each window based on the wall length and frame width
        window_length = self.wall_length - (2 * self.frame_width)

        # Calculate the area of a single window
        window_area = window_width * window_length

        # # Print the calculated window area, width, and length for debugging
        # print(f"Single roof window area: {window_area} m²")
        # print(f"Single roof window width: {window_width} m")
        # print(f"Single roof window length: {window_length} m")
        # print(f"Sloped roof length: {roof_slope_length} m")

        # Return the calculated window area
        return window_area

    def create_default_construction_set(self, set_name):
        """
        Create a new default construction set.

        This function creates a new default construction set in the OpenStudio model. 
        A construction set contains default construction elements such as walls, floors, 
        windows, and roofs that will be used in the building model. The set is named based 
        on the input provided and is returned as an object.

        Args:
            set_name (str): The name of the default construction set.

        Returns:
            openstudio.model.DefaultConstructionSet: The newly created construction set object.
        """
        
        # Create a new DefaultConstructionSet object in the OpenStudio model
        construction_set = openstudio.model.DefaultConstructionSet(self.model)

        # Set the name of the construction set based on the input parameter
        construction_set.setName(set_name)

        # Return the newly created construction set object
        return construction_set
    
    def create_space_and_type(self, space_name, space_type_name):
        """
        Create a new space and space type, and associate the space type with the space.

        This function creates a new space and a corresponding space type in the OpenStudio model.
        The space type is then assigned to the newly created space. Spaces represent specific 
        areas within the building, while space types define the function and characteristics 
        of the spaces (e.g., office, greenhouse, etc.).

        Args:
            space_name (str): The name of the space.
            space_type_name (str): The name of the space type.

        Returns:
            tuple: A tuple containing the newly created space object and space type object.
        """

        # Create a new space in the OpenStudio model
        space = openstudio.model.Space(self.model)
        # Set the name of the space based on the provided space_name
        space.setName(space_name)

        # Create a new space type in the OpenStudio model
        space_type = openstudio.model.SpaceType(self.model)
        # Set the name of the space type based on the provided space_type_name
        space_type.setName(space_type_name)

        # Associate the space type with the space
        space.setSpaceType(space_type)

        # Return both the space and space type objects
        return space, space_type
    
    def create_thermal_zone(self, zone_name, space):
        """
        Create a new thermal zone and associate it with the specified space.

        This function creates a new thermal zone in the OpenStudio model and associates 
        it with the provided space. Thermal zones represent areas in the building that 
        are controlled by HVAC systems and share the same heating, cooling, and ventilation 
        settings.

        Args:
            zone_name (str): The name of the thermal zone.
            space (openstudio.model.Space): The space object to associate with the thermal zone.

        Returns:
            openstudio.model.ThermalZone: The newly created thermal zone object.
        """
        
        # Create a new ThermalZone object in the OpenStudio model
        thermal_zone = openstudio.model.ThermalZone(self.model)
        # Set the name of the thermal zone based on the provided zone_name
        thermal_zone.setName(zone_name)

        # Associate the space with the thermal zone
        space.setThermalZone(thermal_zone)

        # Return the created thermal zone object
        return thermal_zone
    
    def create_construction(self, construction_name, materials):
        """
        Create a new construction assembly.

        This function creates a new construction assembly in the OpenStudio model, which defines 
        the materials used to build building components like walls, roofs, and floors. A construction 
        consists of one or more layers of materials, with each layer representing a different part of 
        the building's structure.

        Args:
            construction_name (str): The name of the construction assembly.
            materials (list[openstudio.model.Material]): A list of material objects representing 
                                                        the layers in the construction.

        Returns:
            openstudio.model.Construction: The newly created construction object.
        """
        
        # Create a new Construction object in the OpenStudio model
        construction = openstudio.model.Construction(self.model)
        # Set the name of the construction based on the provided construction_name
        construction.setName(construction_name)

        # Loop through the list of materials and add each as a layer in the construction
        for i, material in enumerate(materials):
            # Insert each material layer into the construction at the appropriate index
            construction.insertLayer(i, material)

        # Return the created construction object
        return construction

    def create_material(self, material_name, material_type, properties):
        """
        Create a new material.

        This function creates a new material object in the OpenStudio model based on the provided
        material type and associated properties. The material can be either a standard opaque 
        material (e.g., insulation, concrete, etc.) or simple glazing (e.g., glass). The properties
        of the material, such as thermal conductivity, thickness, density, etc., are set using the 
        values provided in the `properties` dictionary.

        Args:
            material_name (str): The name of the material.
            material_type (str): The type of material to create, e.g., 'StandardOpaqueMaterial' 
                                or 'SimpleGlazing'.
            properties (dict): A dictionary containing the material's properties, such as thermal 
                            conductivity, thickness, density, specific heat, etc.

        Returns:
            openstudio.model.Material: The newly created material object.

        Raises:
            ValueError: If an unknown material type is provided.
        """
        
        # Check if the material type is 'StandardOpaqueMaterial'
        # https://openstudio-sdk-documentation.s3.amazonaws.com/cpp/OpenStudio-1.7.0-doc/model/html/classopenstudio_1_1model_1_1_standard_opaque_material.html#a35c960fd05acf5df6becaa0d7751e9d2
        if material_type == "StandardOpaqueMaterial":
            # Create a StandardOpaqueMaterial object in the OpenStudio model
            material = openstudio.model.StandardOpaqueMaterial(self.model)
            # Set thermal conductivity using the value from the properties dictionary, or a default of 0.1
            material.setThermalConductivity(properties.get("thermal_conductivity", 0.1))
            # Set the material thickness if provided in the properties dictionary
            if "thickness" in properties:
                material.setThickness(properties["thickness"])
            # Set the material density if provided in the properties dictionary
            if "density" in properties:
                material.setDensity(properties["density"])
            # Set the material's specific heat if provided in the properties dictionary
            if "specific_heat" in properties:
                material.setSpecificHeat(properties["specific_heat"])
            # Set thermal absorptance if provided in the properties dictionary
            if "thermal_absorptance" in properties:
                material.setThermalAbsorptance(properties["thermal_absorptance"])
            # Set solar absorptance if provided in the properties dictionary
            if "solar_absorptance" in properties:
                material.setSolarAbsorptance(properties["solar_absorptance"])
            # Set visible absorptance if provided in the properties dictionary
            if "visible_absorptance" in properties:
                material.setVisibleAbsorptance(properties["visible_absorptance"])
        
        # Check if the material type is 'SimpleGlazing'
        # https://openstudio-sdk-documentation.s3.amazonaws.com/cpp/OpenStudio-1.12.3-doc/model/html/classopenstudio_1_1model_1_1_simple_glazing.html
        elif material_type == "SimpleGlazing":
            # Create a SimpleGlazing material in the OpenStudio model
            material = openstudio.model.SimpleGlazing(self.model)
            # Set U-factor using the value from the properties dictionary, or a default of 1.0
            material.setUFactor(properties.get("u_factor", 1.0))
            # Set visible transmittance using the value from the properties dictionary, or a default of 0.8
            material.setVisibleTransmittance(properties.get("visible_transmittance", 0.8))
            # Set the material thickness if provided in the properties dictionary
            if "thickness" in properties:
                material.setThickness(properties["thickness"])

        # If the material type is not recognized, raise an error
        else:
            raise ValueError(f"Unknown material type: {material_type}")

        # Set the name of the material
        material.setName(material_name)
        
        # Return the newly created material object
        return material
    
    ###############################################################
    # Function to calculate the z value of a point on a half_circle
    ###############################################################

    def z_value_half_circle(self, x, radius, h):
        """
        Calculate the z-coordinate of a point on a half-circle.

        This function calculates the z-coordinate of a point on a half-circle (a 2D semi-circle) 
        for a given x-coordinate. The semi-circle is centered horizontally at h, and its vertical 
        size is determined by the radius. The function ensures that the x-coordinate is wrapped 
        within the width of the wall and returns the corresponding z-value for the half-circle. 
        If the x-coordinate is outside the bounds of the semi-circle, it returns 0.

        Args:
            x (float): The x-coordinate of the point on the half-circle.
            radius (float): The radius of the half-circle.
            h (float): The x-coordinate of the center of the half-circle.

        Returns:
            float: The z-coordinate of the point on the half-circle. If the x-coordinate is outside 
                the semi-circle, the function returns 0.
        """

        # Adjust the x-coordinate to be within the range of the current arch (wraps x within the wall width)
        x = x % (self.wall_width)
        
        # If the x value is outside the range of the half-circle (beyond the radius), return z = 0
        if (x - h) ** 2 > radius**2:
            return 0
        else:
            # Calculate the z-coordinate using the equation for a circle: z = sqrt(radius^2 - (x - h)^2)
            return np.sqrt(radius**2 - (x - h) ** 2)
        
    ###############################################################
    # Function to calculate the z value of a point on the types of roof 
    # (triangle, flat arch, gothic arch, sawtooth, sawtooth arch)
    ###############################################################

    def z_value_triangle(self, x, height, width):
        """
        Calculate the z-coordinate of a point on a triangle profile.

        This function calculates the z-coordinate of a point on a triangular profile for a given 
        x-coordinate. The triangle has a given width (the base of the triangle) and height (the 
        peak of the triangle). The function adjusts the x-coordinate to be within the current 
        segment (periodic behavior for repeating triangles) and calculates the corresponding 
        z-value based on the slope of the triangle.

        Args:
            x (float): The x-coordinate of the point on the triangle profile.
            height (float): The height of the triangle (the peak of the triangle).
            width (float): The width of the triangle (the base of the triangle).

        Returns:
            float: The z-coordinate (height) of the point on the triangle profile.
        """
        
        # Adjust the x-coordinate to be within the current triangle segment (modulus by 2 * width)
        # This ensures the x value repeats periodically for repeating triangles
        x = x % (2 * width)

        # If x is in the ascending part of the triangle (0 <= x <= width), calculate z using the slope
        if x <= width:
            # z increases linearly with x: z = (x / width) * height
            return x * height / width
        else:
            # If x is in the descending part of the triangle (width < x <= 2 * width),
            # z decreases linearly as x moves toward 2 * width
            return (2 * width - x) * height / width
        
    def z_value_flat_arch(self, x, height, width):
        """
        Calculate the z-coordinate of a point on a flat arch profile.

        This function calculates the z-coordinate of a point on a flat arch, which follows a 
        parabolic shape. The arch is symmetrical and centered over a base with a given width 
        and peak height. The function adjusts the x-coordinate to fit within the current 
        repeating segment and then calculates the z-value using a parabolic equation.

        Args:
            x (float): The x-coordinate of the point on the flat arch.
            height (float): The height of the arch at its peak (in the middle of the arch).
            width (float): The width of the arch (distance from one side to the other).

        Returns:
            float: The z-coordinate (height) of the point on the flat arch.
        """

        # Adjust the x-coordinate to be within the current arch segment (modulus by 2 * width)
        x = x % (2 * width)

        # Calculate the z-coordinate using a parabolic equation for a flat arch
        # The formula is: z = height * (1 - 4 * ((x - width) / (2 * width)) ** 2)
        # This equation defines a parabola that is symmetrical about x = width
        return height * (1 - 4 * ((x - width) / (2 * width)) ** 2)

    def z_value_gothic_arch(self, x, height, width):
        """
        Calculate the z-coordinate of a point on a gothic arch profile.

        This function calculates the z-coordinate of a point on a gothic arch. The gothic arch 
        is modeled as a pointed arch, where the pointedness can be adjusted. The arch is symmetrical 
        with a given base width and peak height. The function adjusts the x-coordinate to fit 
        within the current repeating segment of the arch and calculates the z-value using an 
        absolute value and power function to create the pointed shape of the gothic arch.

        Args:
            x (float): The x-coordinate of the point on the gothic arch.
            height (float): The height of the arch at its peak (in the middle of the arch).
            width (float): The width of the arch (distance from one side to the other).

        Returns:
            float: The z-coordinate (height) of the point on the gothic arch.
        """
        
        # Set a constant for the 'pointedness' of the arch (controls how sharp the peak is)
        pointedness = 1.4
        
        # Adjust the x-coordinate to be within the current arch segment (modulus by 2 * width)
        # This ensures the x value repeats periodically for repeating arches
        x = x % (2 * width)

        # Calculate the z-coordinate using the gothic arch equation
        # The formula is: z = height * (1 - |(x - width) / width|^pointedness)
        # This creates a pointed arch with a peak at the center (x = width) and tapers off at the edges (x = 0 and x = 2 * width)
        return height * (1 - np.abs((x - width) / width) ** pointedness)
    
    def z_value_sawtooth(self, x, height, width):
        """
        Calculate the z-coordinate of a point on a sawtooth profile.

        This function calculates the z-coordinate of a point on a sawtooth profile. The sawtooth 
        pattern alternates between a linear slope up to a peak and then a sudden drop to zero. 
        The function adjusts the x-coordinate to fit within the current repeating segment and 
        calculates the z-value based on the position along the sawtooth's slope.

        Args:
            x (float): The x-coordinate of the point on the sawtooth profile.
            height (float): The height of the sawtooth at its peak.
            width (float): The width of the sawtooth segment (half of one complete sawtooth cycle).

        Returns:
            float: The z-coordinate (height) of the point on the sawtooth profile.
        """
        
        # Double the width to represent the full cycle of the sawtooth
        width = width * 2

        # Adjust the x-coordinate to be within the current sawtooth cycle (modulus by 2 * width)
        x = x % (2 * width)

        # If x is within the ascending slope of the sawtooth (0 <= x <= width), calculate z
        if x <= width:
            # Linearly decreasing height along the slope: z = height * (1 - x / width)
            return height * (1 - (x / width))
        else:
            # After the peak (width < x <= 2 * width), the z-value drops to 0
            return 0
        
    def z_value_sawtooth_arch(self, x, height, width):
        """
        Calculate the z-coordinate of a point on a sawtooth arch profile.

        This function calculates the z-coordinate of a point on a sawtooth arch, which features 
        a parabolic slope followed by an abrupt drop to zero, creating an arch-like structure 
        for each segment. The function adjusts the x-coordinate to fit within the current repeating 
        segment and calculates the z-value based on the parabolic shape of the arch.

        Args:
            x (float): The x-coordinate of the point on the sawtooth arch.
            height (float): The height of the arch at its peak.
            width (float): The width of the sawtooth arch segment (half of one complete cycle).

        Returns:
            float: The z-coordinate (height) of the point on the sawtooth arch.
        """

        # Double the width to represent the full cycle of the sawtooth arch
        width = width * 2

        # Adjust the x-coordinate to be within the current sawtooth arch cycle (modulus by 2 * width)
        x = x % (2 * width)

        # If x is within the ascending parabolic arch slope (0 <= x <= width), calculate z
        if x <= width:
            # Parabolic curve: z = height * (1 - (x / width)^2)
            # This creates a smooth arch-like curve for the sawtooth profile
            return height * (1 - (x / width) ** 2)
        else:
            # After the arch peak (width < x <= 2 * width), the z-value drops to 0
            return 0
        
    ###############################################################
    # Create a natural ventilation schedule
    ###############################################################

    def calculate_total_window_area(self, space_name):
        """
        Calculate the total area of operable windows in a given space.

        This function retrieves a space by its name from the OpenStudio model and calculates the 
        total area of all operable windows in that space. Operable windows are a type of sub-surface 
        that can be opened or closed for natural ventilation.

        Args:
            space_name (str): The name of the space in which to calculate the total operable window area.

        Returns:
            float: The total area of all operable windows in the specified space (in square meters).
        """
        
        # Retrieve the space from the model by its name
        space = self.model.getSpaceByName(space_name).get()

        # Calculate the total area of all operable windows in the space
        # Iterate over all surfaces in the space, then over all sub-surfaces in each surface
        # Check if the sub-surface type is "OperableWindow" and sum their gross areas
        return sum(
            subsurface.grossArea()  # Get the gross area of the sub-surface
            for surface in space.surfaces()  # Loop through each surface in the space
            for subsurface in surface.subSurfaces()  # Loop through each sub-surface in the surface
            if subsurface.subSurfaceType() == "OperableWindow"  # Check if the sub-surface is an operable window
        )
    
    def create_schedule_type_limits(self):
        """
        Create a new schedule type limits model.

        This function creates a schedule type limits object in the OpenStudio model, which defines 
        the constraints for schedule values, such as their upper and lower limits, numeric type, 
        and unit type. In this case, the schedule type limits are set for fractional values between 
        0 and 1 (typically used for schedules that define fractional availability or load factors).

        Returns:
            openstudio.model.ScheduleTypeLimits: The newly created schedule type limits object.
        """

        # Create a new ScheduleTypeLimits object in the OpenStudio model
        fractional_type_limits = openstudio.model.ScheduleTypeLimits(self.model)

        # Set the name of the schedule type limits to "Fractional"
        fractional_type_limits.setName("Fractional")

        # Set the lower limit for the schedule values to 0 (e.g., representing 0% availability)
        fractional_type_limits.setLowerLimitValue(0)

        # Set the upper limit for the schedule values to 1 (e.g., representing 100% availability)
        fractional_type_limits.setUpperLimitValue(1)

        # Set the numeric type to "Continuous", meaning the schedule can have continuous values
        fractional_type_limits.setNumericType("Continuous")

        # Set the unit type to "Dimensionless", indicating that the values are unitless (fractions)
        fractional_type_limits.setUnitType("Dimensionless")

        # Return the created ScheduleTypeLimits object
        return fractional_type_limits

    def create_default_day_schedule(self, open_area_schedule, hour, minute, value):
        """
        Create a default day schedule for a given schedule object.

        This function sets up a default daily schedule for the provided schedule object. 
        It specifies a schedule value at a specific time of the day. The schedule can 
        represent various properties (such as the fraction of an open area, HVAC settings, etc.).

        Args:
            open_area_schedule (openstudio.model.ScheduleRuleset): The schedule object for which the default day schedule is created.
            hour (int): The hour of the day when the schedule value should be applied (0-23).
            minute (int): The minute of the hour when the schedule value should be applied (0-59).
            value (float): The value to be set at the specified time (e.g., a percentage or fractional value).

        Returns:
            openstudio.model.ScheduleDay: The newly created default day schedule with the applied time and value.
        """

        # Retrieve the default day schedule from the provided schedule object
        default_day_schedule = open_area_schedule.defaultDaySchedule()

        # Set a name for the default day schedule
        default_day_schedule.setName("Default Day Schedule")

        # Add a time and value to the default day schedule
        # openstudio.Time(hour, minute, seconds, fractional_seconds) is used to specify the time
        # The value is the schedule value (e.g., fractional open area or HVAC setting) at the specified time
        default_day_schedule.addValue(
            openstudio.Time(0, hour, minute, 0), value
        )

        # Return the created default day schedule
        return default_day_schedule
    
    def write_window_schedule_to_csv(self, open_area_schedule, file_path):
        """
        Write a window opening schedule to a CSV file.

        This function creates a daily schedule for window openings, alternating between open 
        and closed states based on a time step. It writes this schedule to a CSV file, where 
        each row represents a specific time of the day (in seconds) and the corresponding 
        window opening percentage. The window opening values are randomly shuffled from 
        a predefined list.

        Args:
            open_area_schedule (openstudio.model.ScheduleRuleset): The schedule object for window openings.
            file_path (str): The path to the CSV file where the window schedule will be written.

        Returns:
            None. The function writes the schedule data to a CSV file.
        """
        # Open the specified CSV file for writing
        with open(file_path, 'w', newline='') as file:
            writer = csv.writer(file)
            # Write the CSV header row
            writer.writerow(["Time", "Window_Schedule"])

            # Retrieve the default day schedule from the schedule object
            default_day_schedule = open_area_schedule.defaultDaySchedule()

            # Loop through each hour of the day (0 to 23)
            for hour in range(24):
                # For each hour, loop through the minutes based on the time step (e.g., every 10 minutes)
                for minute in range(0, 60, self.time_step):
                    # Get the value from the schedule at the specific time
                    current_time = openstudio.Time(0, hour, minute, 0)
                    value = default_day_schedule.getValue(current_time)

                    # Write the time (in seconds) and the window schedule value to the CSV file
                    writer.writerow([current_time.totalSeconds(), value])
                    
    def create_window_schedule(self):
        """
        Create window schedules for the left and right roof windows, write schedules to CSV files,
        and set up natural ventilation objects for both sides of the greenhouse roof.

        This function calculates the total roof window area, creates open area fraction schedules 
        for the left and right roof windows, writes these schedules to CSV files, and configures 
        natural ventilation objects for both sides of the roof with specified parameters like 
        temperature and wind speed limits.

        Returns:
            None
        """

        # Calculate the total area of the operable roof windows in the greenhouse space
        total_roof_window_area = self.calculate_total_window_area("Greenhouse Space Roof")
        # print(f"Total roof window area：{total_roof_window_area} m²\n")

        # Create and name the open area schedule for the left side of the roof
        open_area_schedule_left = openstudio.model.ScheduleRuleset(self.model)
        open_area_schedule_left.setName("Open Area Fraction Schedule Left")

        # Create and name the open area schedule for the right side of the roof
        open_area_schedule_right = openstudio.model.ScheduleRuleset(self.model)
        open_area_schedule_right.setName("Open Area Fraction Schedule Right")

        # Set schedule type limits to ensure values are between 0 and 1 (fractional values)
        schedule_type_limits = self.create_schedule_type_limits()
        open_area_schedule_left.setScheduleTypeLimits(schedule_type_limits)
        open_area_schedule_right.setScheduleTypeLimits(schedule_type_limits)

        # Set up the window schedule for both the left and right roof windows
        # Create the default day schedule for both left and right windows
        # Set up the left roof window
        
        # self.create_default_day_schedule(open_area_schedule_left, 1, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 2, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 3, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 4, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 5, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 6, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 7, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 8, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 9, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 10, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 11, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 12, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 13, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 14, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 15, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 16, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 17, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 18, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 19, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 20, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 21, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 22, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 23, 0, 1.0)
        # self.create_default_day_schedule(open_area_schedule_left, 24, 0, 1.0)

        # Set up the right roof window
        # self.create_default_day_schedule(open_area_schedule_right, 1, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 2, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 3, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 4, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 5, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 6, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 7, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 8, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 9, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 10, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 11, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 12, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 13, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 14, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 15, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 16, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 17, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 18, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 19, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 20, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 21, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 22, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 23, 0, 0.0)
        # self.create_default_day_schedule(open_area_schedule_right, 24, 0, 0.0)

        # Create a directory to store the window schedules and set a random seed
        folder_path = 'energyplus_data/window_schedule'
        os.makedirs(folder_path, exist_ok=True)

        # Write the window schedules for the left and right sides of the roof to CSV files
        self.write_window_schedule_to_csv(
            open_area_schedule_left, 
            f'energyplus_data/window_schedule/window_schedule_{self.roof_type}_left_{self.time_step}.csv'
        )
        self.write_window_schedule_to_csv(
            open_area_schedule_right, 
            f'energyplus_data/window_schedule/window_schedule_{self.roof_type}_right_{self.time_step}.csv'
        )

        # Create a natural ventilation object and set the open area of ​​the ventilation zone
        # OpenStudio SDK documentation: https://openstudio-sdk-documentation.s3.amazonaws.com/cpp/OpenStudio-3.0.0-doc/model/html/classopenstudio_1_1model_1_1_zone_ventilation_windand_stack_open_area.html
        # EnergyPlus documentation: https://bigladdersoftware.com/epx/docs/23-2/input-output-reference/group-airflow.html#zoneventilationwindandstackopenarea
        # https://bigladdersoftware.com/epx/docs/23-2/input-output-reference/group-airflow.html#field-delta-temperature-1

        # Create a natural ventilation object for the left side of the roof
        zone_ventilation_left = openstudio.model.ZoneVentilationWindandStackOpenArea(self.model)
        zone_ventilation_left.setName("Roof Window Ventilation Left")
        zone_ventilation_left.setOpeningEffectiveness(0.5)  # Set opening effectiveness
        zone_ventilation_left.setDischargeCoefficientforOpening(0.6)  # Set discharge coefficient
        zone_ventilation_left.setOpeningArea(self.roof_ventilation_area_left)  # Set open area
        zone_ventilation_left.setEffectiveAngle(90)  # Set effective angle for left side
        zone_ventilation_left.setHeightDifference((self.roof_height_relative_to_wall + self.wall_height) / 2)
        zone_ventilation_left.setOpeningAreaFractionSchedule(open_area_schedule_left)  # Set schedule
        zone_ventilation_left.setMinimumIndoorTemperature(self.min_indoor_temp)  # Min indoor temp
        zone_ventilation_left.setMaximumIndoorTemperature(self.max_indoor_temp)  # Max indoor temp
        zone_ventilation_left.setDeltaTemperature(self.max_delta_temp)  # Indoor/outdoor temp diff
        zone_ventilation_left.setMinimumOutdoorTemperature(self.min_outdoor_temp)  # Min outdoor temp
        zone_ventilation_left.setMaximumOutdoorTemperature(self.max_outdoor_temp)  # Max outdoor temp
        zone_ventilation_left.setMaximumWindSpeed(self.max_wind_speed)  # Max wind speed
        zone_ventilation_left.addToThermalZone(self.thermal_zone_roof)  # Add to thermal zone

        # Create a natural ventilation object for the right side of the roof
        zone_ventilation_right = openstudio.model.ZoneVentilationWindandStackOpenArea(self.model)
        zone_ventilation_right.setName("Roof Window Ventilation Right")
        zone_ventilation_right.setOpeningArea(self.roof_ventilation_area_right)  # Set open area
        zone_ventilation_right.setEffectiveAngle(270)  # Set effective angle for right side
        zone_ventilation_right.setHeightDifference((self.roof_height_relative_to_wall + self.wall_height) / 2)
        zone_ventilation_right.setOpeningAreaFractionSchedule(open_area_schedule_right)  # Set schedule
        zone_ventilation_right.setMinimumIndoorTemperature(self.min_indoor_temp)  # Min indoor temp
        zone_ventilation_right.setMaximumIndoorTemperature(self.max_indoor_temp)  # Max indoor temp
        zone_ventilation_right.setDeltaTemperature(self.max_delta_temp)  # Indoor/outdoor temp diff
        zone_ventilation_right.setMinimumOutdoorTemperature(self.min_outdoor_temp)  # Min outdoor temp
        zone_ventilation_right.setMaximumOutdoorTemperature(self.max_outdoor_temp)  # Max outdoor temp
        zone_ventilation_right.setMaximumWindSpeed(self.max_wind_speed)  # Max wind speed
        zone_ventilation_right.addToThermalZone(self.thermal_zone_roof)  # Add to thermal zone

    def create_house_model(self, w, l, number_width, number_length):
        """
        Create a 3D house model with walls, windows, and a segmented roof structure.

        This function defines the geometry of the house including the base (floor), walls, windows,
        and roof segments. It computes each part of the house with appropriate vertex placements and 
        assigns them to the corresponding spaces in the OpenStudio model.
        
        Args:
            w (int): The horizontal index of the house section (used for positioning the house in a grid).
            l (int): The vertical index of the house section.
            number_width (int): The total number of sections along the width.
            number_length (int): The total number of sections along the length.

        Returns:
            float: The total volume of the roof.
        """
        # Create offsets in x and y directions to position each house section
        x_offset = w * self.wall_width  # Offset in x direction based on width
        y_offset = l * self.wall_length  # Offset in y direction based on length

        # Use space and thermal zone specific to this grid position
        space = self.spaces[w * self.number_length + l]
        thermal_zone = self.thermal_zones[w * self.number_length + l]

        # Define the base (floor) vertices using the offsets
        base_vertices = [
            openstudio.Point3d(0 + x_offset, 0 + y_offset, 0),  # Lower-left corner
            openstudio.Point3d(0 + x_offset, self.wall_length + y_offset, 0),  # Upper-left corner
            openstudio.Point3d(self.wall_width + x_offset, self.wall_length + y_offset, 0),  # Upper-right corner
            openstudio.Point3d(self.wall_width + x_offset, 0 + y_offset, 0),  # Lower-right corner
        ]

        # Create the floor surface using the base vertices
        floor = openstudio.model.Surface(base_vertices, self.model)
        floor.setSpace(space)  # Assign the floor to the specific space
        floor.setConstruction(self.floor_construction)  # Set the floor construction (materials)

        # Define the coordinate points of the four windows
        window_points_list = [
            # Front window (located at the front of the house)
            [
                openstudio.Point3d(0 + x_offset, self.frame_width + y_offset, self.wall_height - self.frame_width),
                openstudio.Point3d(0 + x_offset, self.wall_length - self.frame_width + y_offset, self.wall_height - self.frame_width),
                openstudio.Point3d(0 + x_offset, self.wall_length - self.frame_width + y_offset, self.frame_width),
                openstudio.Point3d(0 + x_offset, self.frame_width + y_offset, self.frame_width),
            ],
            # Right window
            [
                openstudio.Point3d(self.frame_width + x_offset, self.wall_length + y_offset, self.wall_height - self.frame_width),
                openstudio.Point3d(self.wall_width - self.frame_width + x_offset, self.wall_length + y_offset, self.wall_height - self.frame_width),
                openstudio.Point3d(self.wall_width - self.frame_width + x_offset, self.wall_length + y_offset, self.frame_width),
                openstudio.Point3d(self.frame_width + x_offset, self.wall_length + y_offset, self.frame_width),
            ],
            # Back window
            [
                openstudio.Point3d(self.wall_width + x_offset, self.wall_length - self.frame_width + y_offset, self.wall_height - self.frame_width),
                openstudio.Point3d(self.wall_width + x_offset, self.frame_width + y_offset, self.wall_height - self.frame_width),
                openstudio.Point3d(self.wall_width + x_offset, self.frame_width + y_offset, self.frame_width),
                openstudio.Point3d(self.wall_width + x_offset, self.wall_length - self.frame_width + y_offset, self.frame_width),
            ],
            # Left window
            [
                openstudio.Point3d(self.wall_width - self.frame_width + x_offset, 0 + y_offset, self.wall_height - self.frame_width),
                openstudio.Point3d(self.frame_width + x_offset, 0 + y_offset, self.wall_height - self.frame_width),
                openstudio.Point3d(self.frame_width + x_offset, 0 + y_offset, self.frame_width),
                openstudio.Point3d(self.wall_width - self.frame_width + x_offset, 0 + y_offset, self.frame_width),
            ],
        ]

        # Create walls and windows for each side of the house
        for i in range(4):
            if number_length != 1:
                # Skip some walls depending on the grid position (for better house alignment)
                if l == 0 and i == 1:
                    continue
                if l != 0 and l != number_length - 1 and (i == 1 or i == 3):
                    continue
                if l == number_length - 1 and i == 3:
                    continue

            if number_width != 1:
                if w == 0 and i == 2:
                    continue
                if w != 0 and w != number_width - 1 and (i == 0 or i == 2):
                    continue
                if w == number_width - 1 and i == 0:
                    continue

            # Define wall vertices
            wall_vertices = [
                openstudio.Point3d(base_vertices[i].x(), base_vertices[i].y(), self.wall_height),  # Top-left of wall
                openstudio.Point3d(base_vertices[(i + 1) % 4].x(), base_vertices[(i + 1) % 4].y(), self.wall_height),  # Top-right of wall
                base_vertices[(i + 1) % 4],  # Bottom-right of wall
                base_vertices[i],  # Bottom-left of wall
            ]
            wall = openstudio.model.Surface(wall_vertices, self.model)  # Create the wall
            wall.setSurfaceType("Wall")  # Set the wall type
            wall.setConstruction(self.frame_construction)  # Set the wall construction (materials)
            wall.setSpace(space)  # Assign the wall to the specific space
            wall.setName("Wall_" + str(w) + "_" + str(l))  # Name the wall

            # Create the window on the wall
            window = openstudio.model.SubSurface(window_points_list[i], self.model)
            window.setConstruction(self.window_construction_main)  # Set window construction
            window.setSubSurfaceType("FixedWindow")  # Set window type as fixed
            window.setName("Window_Wall__" + str(w) + "_" + str(l))  # Name the window
            window.setSurface(wall)  # Attach the window to the wall

            # Create and set the window's frame and divider properties
            frame_and_divider = openstudio.model.WindowPropertyFrameAndDivider(self.model)
            frame_and_divider.setFrameWidth(0.01)  # Set frame width
            frame_and_divider.setFrameConductance(5.7)  # Set frame thermal conductance
            window.setWindowPropertyFrameAndDivider(frame_and_divider)  # Apply frame/divider to window
        
        ###############################################################
        # Creat Roof structure
        ###############################################################
        
        # Calculate the width of each roof segment along the x-axis
        segment_x_axis_width = self.wall_width / self.num_segments
        # This divides the total roof width by the number of segments (self.num_segments), giving the width of each roof section.
        
        # Initialize total volume to accumulate the volume of each roof segment
        total_volume = 0

        # Create a WindowPropertyFrameAndDivider object to define the frame and divider properties for roof windows
        frame_and_divider = openstudio.model.WindowPropertyFrameAndDivider(self.model)

        # Set frame and divider properties for roof windows
        frame_and_divider.setFrameWidth(0.01)  # Set the width of the window frame to 0.01 meters
        frame_and_divider.setFrameConductance(5.7)  # Set the thermal conductance of the frame to 5.7 W/m2K

        # Iterate over the roof width, creating segments from 0 to the total width, in steps of segment_x_axis_width
        for count, i in enumerate(np.arange(0, self.wall_width, segment_x_axis_width)):
            # Calculate the z-value (height) of the roof at the left edge of the segment (i)
            z1 = self.z_value(i, 
                              self.roof_height_relative_to_wall, 
                              self.wall_width / 2)
            
            # Calculate the z-value (height) of the roof at the right edge of the segment (i + segment_x_axis_width)
            z2 = self.z_value(i + segment_x_axis_width, 
                              self.roof_height_relative_to_wall, 
                              self.wall_width / 2)

            # Calculate the top and bottom bases for the trapezoidal prism (used to compute the roof segment volume)
            top_base = z1
            bottom_base = z2
            height_trapezoid = segment_x_axis_width
            height_prism = self.wall_length

            # Calculate the volume of the current trapezoidal prism segment
            segment_volume = self.calculate_trapezoidal_prism_volume(top_base, 
                                                                     bottom_base, 
                                                                     height_trapezoid, 
                                                                     height_prism)
            
            # Accumulate the total volume for all roof segments
            total_volume += segment_volume

            # Calculate the slope of the roof segment
            slope = (z2 - z1) / segment_x_axis_width
            
            # Calculate the segment width, which accounts for the slope of the roof
            segment_width = np.sqrt(segment_x_axis_width**2 + (z2 - z1) ** 2)
            
            # Calculate the window width for this roof segment, subtracting the frame widths on both sides
            window_width = segment_width - 2 * self.frame_width

            # Calculate the x and z coordinates for the left side of the window
            window_x1 = i + self.frame_width * np.cos(np.arctan(slope))  # Adjust x based on frame width and slope
            window_z1 = self.wall_height + z1 + self.frame_width * np.sin(np.arctan(slope))  # Adjust z based on frame width and slope

            # Calculate the x and z coordinates for the right side of the window
            window_x2 = window_x1 + window_width * np.cos(np.arctan(slope))  # Compute x2 based on window width
            window_z2 = window_z1 + window_width * np.sin(np.arctan(slope))  # Compute z2 based on window width and slope

            # Define the vertices of the current roof segment, including x, y, and z values
            segment_vertices = [
                openstudio.Point3d(i + x_offset, 0 + y_offset, self.wall_height + z1),  # Bottom-left corner
                openstudio.Point3d(i + segment_x_axis_width + x_offset, 0 + y_offset, self.wall_height + z2),  # Bottom-right corner
                openstudio.Point3d(i + segment_x_axis_width + x_offset, self.wall_length + y_offset, self.wall_height + z2),  # Top-right corner
                openstudio.Point3d(i + x_offset, self.wall_length + y_offset, self.wall_height + z1)  # Top-left corner
            ]

            # Create the roof segment as a surface using the defined vertices
            roof_segment = openstudio.model.Surface(segment_vertices, self.model)
            # Set the surface type to "RoofCeiling", indicating this is a roof
            roof_segment.setSurfaceType("RoofCeiling")
            # Assign the roof segment to the frame construction, which includes material properties
            roof_segment.setConstruction(self.frame_construction)
            # Set a unique name for the roof segment
            roof_segment.setName("roof_segment_" + str(w) + "_" + str(l))
            # Assign the roof segment to the roof space
            roof_segment.setSpace(self.space_roof)

            # Define the vertices of the window for this roof segment using the previously calculated coordinates
            window_vertices = [
                openstudio.Point3d(window_x1 + x_offset, self.frame_width + y_offset, window_z1),  # Bottom-left corner
                openstudio.Point3d(window_x2 + x_offset, self.frame_width + y_offset, window_z2),  # Bottom-right corner
                openstudio.Point3d(window_x2 + x_offset, self.wall_length - self.frame_width + y_offset, window_z2),  # Top-right corner
                openstudio.Point3d(window_x1 + x_offset, self.wall_length - self.frame_width + y_offset, window_z1)  # Top-left corner
            ]

            # Create the window for the roof segment using the defined vertices
            window = openstudio.model.SubSurface(window_vertices, self.model)
            # Set the window type as "OperableWindow", indicating it can be opened
            window.setSubSurfaceType("OperableWindow")

            # Determine if the window is on the left or right side of the roof
            if i < self.wall_width / 2:
                # For the left side of the roof, assign the left roof window construction
                window.setConstruction(self.window_construction_roof_left)
                window.setName("Window_Roof_Left_" + str(w) + "_" + str(l))
            else:
                # For the right side of the roof, assign the right roof window construction
                window.setConstruction(self.window_construction_roof_right)
                window.setName("Window_Roof_Right_" + str(w) + "_" + str(l))

            # Attach the window to the roof segment (surface)
            window.setSurface(roof_segment)

            # Apply the frame and divider properties to the window
            window.setWindowPropertyFrameAndDivider(frame_and_divider)

            ###############################################################
            # Creat Left Side Surface
            ###############################################################

            # Start creating the left side of the roof
            # Calculate the z-coordinates for the top of the side windows (adjusting by subtracting wall height)
            side_windows_z1 = window_z1 - self.wall_height  # Z-coordinate for the left window's top edge
            side_windows_z2 = window_z2 - self.wall_height  # Z-coordinate for the right window's top edge
            side_wall_offset = self.frame_width / 8  # Small offset for window positioning on the side wall

            if l == 0:  # Condition to determine if we're on the leftmost side (first column of rooms)
                # Define the vertices for the left side wall
                left_side_vertices = [
                    openstudio.Point3d(i + x_offset, 0 + y_offset, self.wall_height),  # Bottom-left corner
                    openstudio.Point3d(i + segment_x_axis_width + x_offset, 0 + y_offset, self.wall_height),  # Bottom-right corner
                    openstudio.Point3d(i + segment_x_axis_width + x_offset, 0 + y_offset, self.wall_height + z2),  # Top-right corner
                    openstudio.Point3d(i + x_offset, 0 + y_offset, self.wall_height + z1),  # Top-left corner
                ]

                # Create the left vertical surface (side wall)
                left_side_surface = openstudio.model.Surface(left_side_vertices, self.model)
                # Associate the vertical surface with the roof space
                left_side_surface.setSpace(self.space_roof)

                # Set the surface type for the side wall as "RoofCeiling"
                left_side_surface.setSurfaceType("RoofCeiling")

                # Set the construction for the vertical side wall, using the frame construction materials
                left_side_surface.setConstruction(self.frame_construction)
                # Assign a name to the left side surface based on the current room position
                left_side_surface.setName("left_side_" + str(w) + "_" + str(l))

                # Begin creating the windows for the left side vertical surface
                # Define the window vertices for the left side surface
                left_side_window_vertices = [
                    openstudio.Point3d(window_x1 + x_offset, 0 + y_offset, self.wall_height + side_wall_offset),  # Bottom-left corner
                    openstudio.Point3d(window_x2 + x_offset, 0 + y_offset, self.wall_height + side_wall_offset),  # Bottom-right corner
                    openstudio.Point3d(window_x2 + x_offset, 0 + y_offset, self.wall_height + side_windows_z2 - side_wall_offset),  # Top-right corner
                    openstudio.Point3d(window_x1 + x_offset, 0 + y_offset, self.wall_height + side_windows_z1 - side_wall_offset),  # Top-left corner
                ]

                # Create a subsurface (window) for the left vertical surface
                left_windows = openstudio.model.SubSurface(left_side_window_vertices, self.model)
                # Set the construction for the window, using the roof window construction materials
                left_windows.setConstruction(self.window_construction_roof)
                # Assign a name to the window based on the room position
                left_windows.setName("Left_side_window_" + str(w) + "_" + str(l))
                # Associate the window with the vertical side surface
                left_windows.setSurface(left_side_surface)
                # Set the window type as a "FixedWindow" (non-operable window)
                left_windows.setSubSurfaceType("FixedWindow")

                # Create a WindowPropertyFrameAndDivider object to define the frame and divider properties for the window
                frame_and_divider = openstudio.model.WindowPropertyFrameAndDivider(self.model)

                # Set the width of the window frame
                frame_and_divider.setFrameWidth(0.01)  # Example: setting frame width to 0.01 meters
                # Set the thermal conductance of the window frame
                frame_and_divider.setFrameConductance(5.7)  # Example: setting frame conductance to 5.7 W/m2K

                # Apply the frame and divider properties to the window (subsurface)
                left_windows.setWindowPropertyFrameAndDivider(frame_and_divider)

            ###############################################################
            # Creat Right Side Surface
            ###############################################################

            # If the current room is in the last column (rightmost side of the model)
            if l == number_length - 1:
                # Define the vertices for the right side wall
                right_side_vertices = [
                    openstudio.Point3d(i + x_offset, self.wall_length + y_offset, self.wall_height + z1),  # Bottom-left
                    openstudio.Point3d(i + segment_x_axis_width + x_offset, self.wall_length + y_offset, self.wall_height + z2),  # Bottom-right
                    openstudio.Point3d(i + segment_x_axis_width + x_offset, self.wall_length + y_offset, self.wall_height),  # Top-right
                    openstudio.Point3d(i + x_offset, self.wall_length + y_offset, self.wall_height)  # Top-left
                ]

                # Create the right vertical surface (side wall)
                right_side_surface = openstudio.model.Surface(right_side_vertices, self.model)
                # Associate the vertical surface with the roof space
                right_side_surface.setSpace(self.space_roof)
                # Set the surface type for the right side wall as "RoofCeiling"
                right_side_surface.setSurfaceType("RoofCeiling")
                # Set the construction for the right vertical side wall, using the frame construction materials
                right_side_surface.setConstruction(self.frame_construction)
                # Assign a name to the right side surface based on the current room position
                right_side_surface.setName("Right_side_" + str(w) + "_" + str(l))

                # Define the window vertices for the right side surface
                right_side_window_vertices = [
                    openstudio.Point3d(window_x1 + x_offset, self.wall_length + y_offset, self.wall_height + side_windows_z1 - side_wall_offset),  # Bottom-left
                    openstudio.Point3d(window_x2 + x_offset, self.wall_length + y_offset, self.wall_height + side_windows_z2 - side_wall_offset),  # Bottom-right
                    openstudio.Point3d(window_x2 + x_offset, self.wall_length + y_offset, self.wall_height + side_wall_offset),  # Top-right
                    openstudio.Point3d(window_x1 + x_offset, self.wall_length + y_offset, self.wall_height + side_wall_offset)  # Top-left
                ]

                # Create a SubSurface (window) for the right vertical surface
                right_windows = openstudio.model.SubSurface(right_side_window_vertices, self.model)
                # Set the construction for the window, using the roof window construction materials
                right_windows.setConstruction(self.window_construction_roof)
                # Assign a name to the window based on the room position
                right_windows.setName("Right_side_window_" + str(w) + "_" + str(l))
                # Associate the window with the right vertical side surface
                right_windows.setSurface(right_side_surface)
                # Set the window type as a "FixedWindow" (non-operable window)
                right_windows.setSubSurfaceType("FixedWindow")

                # Create a WindowPropertyFrameAndDivider object to define the frame and divider properties for the window
                frame_and_divider = openstudio.model.WindowPropertyFrameAndDivider(self.model)

                # Set the width of the window frame
                frame_and_divider.setFrameWidth(0.01)  # Example: setting frame width to 0.01 meters
                # Set the thermal conductance of the window frame
                frame_and_divider.setFrameConductance(5.7)  # Example: setting frame conductance to 5.7 W/m2K

                # Apply the frame and divider properties to the right windows
                right_windows.setWindowPropertyFrameAndDivider(frame_and_divider)

            ###############################################################
            # If the roof type is sawtooth, create the vertical side surfaces
            ###############################################################
            if (i == 0 and self.z_value == self.z_value_sawtooth) or (
                i == 0 and self.z_value == self.z_value_sawtooth_arch
            ):
                # Define the vertices for the vertical side surface of the sawtooth roof
                vertical_surface_vertices = [
                    openstudio.Point3d(i + x_offset, 0 + y_offset, self.wall_height),  # Bottom-left corner
                    openstudio.Point3d(
                        i + x_offset,
                        0 + y_offset,
                        self.wall_height + self.roof_height_relative_to_wall,  # Top-left corner
                    ),
                    openstudio.Point3d(
                        i + x_offset,
                        self.wall_length + y_offset,
                        self.wall_height + self.roof_height_relative_to_wall,  # Top-right corner
                    ),
                    openstudio.Point3d(
                        i + x_offset, self.wall_length + y_offset, self.wall_height  # Bottom-right corner
                    ),
                ]

                # Create the vertical side surface for the sawtooth roof using the vertices
                vertical_surface = openstudio.model.Surface(
                    vertical_surface_vertices, self.model
                )

                # Set the construction of the vertical surface to the frame construction material
                vertical_surface.setConstruction(self.frame_construction)
                # Give the vertical surface a unique name based on the position
                vertical_surface.setName(
                    "Vertical_surface_" + str(w) + "_" + str(l))

                # Assign the vertical surface to the roof space
                vertical_surface.setSpace(self.space_roof)
                # Set the surface type to "RoofCeiling" to indicate it's part of the roof
                vertical_surface.setSurfaceType("RoofCeiling")

                # Define the vertices for the vertical window on the side surface of the sawtooth roof
                roof_vertical_window_vertices = [
                    openstudio.Point3d(
                        i + x_offset,
                        self.frame_width + y_offset,
                        self.wall_height + self.frame_width,  # Bottom-left corner
                    ),
                    openstudio.Point3d(
                        i + x_offset,
                        self.frame_width + y_offset,
                        self.wall_height
                        + self.roof_height_relative_to_wall
                        - self.frame_width,  # Top-left corner
                    ),
                    openstudio.Point3d(
                        i + x_offset,
                        self.wall_length - self.frame_width + y_offset,
                        self.wall_height
                        + self.roof_height_relative_to_wall
                        - self.frame_width,  # Top-right corner
                    ),
                    openstudio.Point3d(
                        i + x_offset,
                        self.wall_length - self.frame_width + y_offset,
                        self.wall_height + self.frame_width,  # Bottom-right corner
                    ),
                ]

                # Create the vertical window (SubSurface) for the sawtooth roof
                roof_vertical_window = openstudio.model.SubSurface(
                    roof_vertical_window_vertices, self.model
                )
                # Set the construction of the window using the roof window construction material
                roof_vertical_window.setConstruction(
                    self.window_construction_roof)
                # Give the vertical window a unique name based on the position
                roof_vertical_window.setName(
                    "Vertical_surface_window_" + str(w) + "_" + str(l)
                )
                # Assign the vertical window to the vertical surface
                roof_vertical_window.setSurface(vertical_surface)
                # Set the window type to "FixedWindow" (non-operable window)
                roof_vertical_window.setSubSurfaceType("FixedWindow")

                # Create a WindowPropertyFrameAndDivider object to define the frame and divider properties for the window
                frame_and_divider = openstudio.model.WindowPropertyFrameAndDivider(
                    self.model)

                # Set the frame width for the window
                frame_and_divider.setFrameWidth(0.01)  # Example: setting frame width to 0.01 meters
                # Set the frame conductance for the window
                frame_and_divider.setFrameConductance(5.7)  # Example: setting frame conductance to 5.7 W/m2K

                # Apply the frame and divider properties to the window (SubSurface)
                roof_vertical_window.setWindowPropertyFrameAndDivider(
                    frame_and_divider)

        # Return the total volume of the roof
        return total_volume
    
    def create_houses(self):
        """
        This function generates a series of house models, computes the total volume of the roof, 
        creates a window schedule, and saves the model in OSM (OpenStudio) and IDF (EnergyPlus) formats.
        """

        # Loop through the length of the grid of houses (number of rows)
        for l in range(self.number_length):
            # Loop through the width of the grid of houses (number of columns)
            for w in range(self.number_width):
                # Create an individual house model and calculate its total roof volume
                total_volume = self.create_house_model(w, l, self.number_width, self.number_length)

                # Add the volume of the current house's roof to the total roof volume
                self.roof_volume += total_volume

        # # Print the total calculated roof volume
        # print("Total Roof Volume:", self.roof_volume)

        # Correct the thermal zone volume by setting the roof thermal zone to the calculated roof volume
        self.thermal_zone_roof.setVolume(self.roof_volume)

        # Define the folder path to save model files
        folder_path = 'energyplus_data/model_files'
        # If the folder does not exist, create it
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # # Generate the window schedule for the model
        # self.create_window_schedule()

        # Set the path for saving the OpenStudio (OSM) file
        osm_path = openstudio.toPath(f"{folder_path}/greenhouse_{self.roof_type}.osm")

        # Save the current model to an OSM file
        self.model.save(osm_path, True)  # True indicates that the file should be overwritten if it already exists

        # Translate the OpenStudio model to an EnergyPlus (IDF) model
        forward_translator = openstudio.energyplus.ForwardTranslator()
        workspace = forward_translator.translateModel(self.model)

        # Set the path for saving the EnergyPlus (IDF) file
        idf_path = f"energyplus_data/model_files/greenhouse_{self.roof_type}.idf"

        # Save the EnergyPlus model to an IDF file
        workspace.save(openstudio.toPath(idf_path), True)  # True indicates that the file should be overwritten if it already exists


    def calculate_surface_area_slope(self, wall_length, wall_width, roof_height_relative_to_wall):
        """
        Calculate the surface area of a building, including the roof and walls, 
        based on the wall dimensions and the height of the roof relative to the wall.

        Args:
            wall_length (float): The length of the wall.
            wall_width (float): The width of the wall.
            roof_height_relative_to_wall (float): The height of the roof relative to the wall.
        
        Returns:
            tuple: A tuple containing the total surface area (float) and the roof slope in degrees (float).
        """

        # Step 1: Calculate the roof slope length using Pythagorean theorem
        # This computes the diagonal length of the roof slope based on half the wall width and the roof height
        roof_slope_length = np.sqrt((wall_width / 2)**2 + roof_height_relative_to_wall**2)

        # Step 2: Calculate the total roof area
        # The roof area is calculated by multiplying the slope length by the wall length (for one side),
        # then multiplying by 2 since the roof has two sloped sides
        roof_area = roof_slope_length * wall_length * 2

        # Step 3: Calculate the total side wall area
        # Side walls are vertical walls with dimensions of wall height by wall length.
        # We multiply by 2 for the two opposite side walls.
        side_wall_area = self.wall_height * wall_length * 2

        # Step 4: Calculate the total end wall area
        # End walls are the vertical walls at the ends of the building (gabled sides),
        # with dimensions of wall height by wall width.
        # Multiply by 2 for the two opposite end walls.
        end_wall_area = self.wall_height * wall_width * 2

        # Step 5: Calculate the total surface area
        # The total surface area is the sum of the roof area, side wall area, and end wall area.
        total_surface_area = roof_area + side_wall_area + end_wall_area

        # Step 6: Calculate the slope of the roof
        # The slope is calculated as the arctangent (inverse tangent) of the roof height over half the wall width.
        # The result is in radians, so we convert it to degrees.
        slope = np.arctan(roof_height_relative_to_wall / (wall_width / 2))

        # # Print the calculated roof slope in degrees
        # print("Roof Slope:", np.degrees(slope))

        # # Print the total surface area of the building
        # print("Total Surface Area:", total_surface_area)

        # Return the total surface area and the roof slope in degrees
        return total_surface_area, np.degrees(slope)

if __name__ == "__main__":
    # Initialize the GreenhouseGeometry class with desired parameters
    greenhouse = GreenhouseGeometry(wall_thickness=0.2,  # Wall thickness (in meters)
        window_thickness=0.3,  # Window thickness (in meters)
        roof_type="triangle",  # Type of roof (e.g., "triangle", "flat", etc.)
        wall_height=4,  # Wall height (in meters)
        wall_width=4,  # Wall width (in meters)
        wall_length=4,  # Wall length (in meters)
        slope=23,  # Roof slope (in degrees)
        num_segments=2,  # Number of segments used for geometry
        frame_width=0.05,  # Frame width (in meters)
        shade_distance_to_roof=3,  # Distance from shade to the roof (in meters)
        time_step=60,  # Time step for the simulation (60/time_step equals to the number of interval in one hour)
        number_width=8,  # Number of grid sections along width
        number_length=8,  # Number of grid sections along length
        max_indoor_temp=60,  # Maximum allowable indoor temperature
        min_indoor_temp=-10,  # Minimum allowable indoor temperature
        max_outdoor_temp=60,  # Maximum allowable outdoor temperature
        min_outdoor_temp=-10,  # Minimum allowable outdoor temperature
        max_delta_temp=-5,  # Maximum temperature difference allowed between indoor and outdoor
        max_wind_speed=30,  # Maximum wind speed allowed (in m/s)
        start_month=1,  # Starting month of the simulation period
        start_day=1,  # Starting day of the simulation period
        end_month=12,  # Ending month of the simulation period
        end_day=31  # Ending day of the simulation period
        )

    # Run the create_houses() method
    greenhouse.create_houses()
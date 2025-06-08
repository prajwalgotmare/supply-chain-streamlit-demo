import pandas as pd
import numpy as np
import random

# --- 1. CONFIGURATION ---
NUM_ROWS = 1000
OUTPUT_FILENAME = 'eta_trip_data.csv'

# Define the possible values for our categorical features
VEHICLE_TYPES = ['van', 'truck_small', 'truck_large']
WEATHER_CONDITIONS = ['clear', 'rain', 'foggy', 'light_snow']
LOAD_TYPES = ['light', 'medium', 'heavy']

# Define the "physics" of our simulation
# These multipliers will affect the base travel time
WEATHER_MODIFIERS = {'clear': 1.0, 'rain': 1.2, 'foggy': 1.35, 'light_snow': 1.5}
VEHICLE_MODIFIERS = {'van': 1.0, 'truck_small': 1.05, 'truck_large': 1.15}
LOAD_MODIFIERS = {'light': 1.0, 'medium': 1.03, 'heavy': 1.1}
BASE_SPEED_KMPH = 70 # Average speed in perfect conditions

print("Starting dataset generation...")

# --- 2. DATA GENERATION LOOP ---
trip_data_list = []
for i in range(NUM_ROWS):
    # --- Choose random features for this trip ---
    route_id = f"R-{i+1:04d}"
    distance_km = round(random.uniform(20, 1500), 1)
    vehicle_type = random.choice(VEHICLE_TYPES)
    weather = random.choice(WEATHER_CONDITIONS)
    load_type = random.choice(LOAD_TYPES)

    # --- Calculate the ETA based on our rules ---
    # a) Calculate the base time from distance and base speed
    base_eta_hours = distance_km / BASE_SPEED_KMPH
    
    # b) Get the multipliers for the chosen features
    weather_mod = WEATHER_MODIFIERS[weather]
    vehicle_mod = VEHICLE_MODIFIERS[vehicle_type]
    load_mod = LOAD_MODIFIERS[load_type]
    
    # c) Calculate the modified ETA by applying all factors
    modified_eta = base_eta_hours * weather_mod * vehicle_mod * load_mod
    
    # d) Add random noise to make it more realistic (+/- 5% variation)
    # This simulates unpredictable factors like traffic, unexpected stops, etc.
    noise_factor = np.random.uniform(0.95, 1.05)
    actual_eta_hours = round(modified_eta * noise_factor, 2)
    
    # Ensure ETA is not impossibly small
    if actual_eta_hours < 0.2:
        actual_eta_hours = 0.2
        
    # --- Append the generated trip data to our list ---
    trip_data_list.append({
        'route_id': route_id,
        'distance_km': distance_km,
        'vehicle_type': vehicle_type,
        'weather': weather,
        'load_type': load_type,
        'actual_eta_hours': actual_eta_hours
    })

# --- 3. CREATE DATAFRAME AND SAVE TO CSV ---
df_trips = pd.DataFrame(trip_data_list)

# Save the DataFrame to a CSV file. `index=False` avoids writing row numbers.
df_trips.to_csv(OUTPUT_FILENAME, index=False)

print(f"\n✅ Successfully generated '{OUTPUT_FILENAME}' with {len(df_trips)} rows.")
print("\nHere's a preview of the first 5 rows of your data:")
print(df_trips.head())
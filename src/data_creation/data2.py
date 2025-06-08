import pandas as pd
import numpy as np
import random

# --- Configuration ---
NUM_ROWS = 500
FILE_NAME = "spoilage_data.csv"

# --- Define Product Characteristics ---
# Each product has a base spoilage chance and sensitivities to different factors.
# - base_chance: Inherent likelihood to spoil.
# - temp_sensitivity: How much temperature affects spoilage.
# - time_sensitivity: How much transit time affects spoilage.
# - shock_sensitivity: How much physical shock affects spoilage.

PRODUCT_PROFILES = {
    "strawberry45": {
        "base_chance": 0.15,
        "temp_sensitivity": 0.020,  # Very sensitive to temp
        "time_sensitivity": 0.004,
        "shock_sensitivity": 0.03
    },
    "milk_carton": {
        "base_chance": 0.05,
        "temp_sensitivity": 0.025,  # MOST sensitive to temp
        "time_sensitivity": 0.002,
        "shock_sensitivity": 0.00
    },
    "cheese_block": {
        "base_chance": 0.02,
        "temp_sensitivity": 0.015,
        "time_sensitivity": 0.001,
        "shock_sensitivity": 0.00
    },
    "banana78": {
        "base_chance": 0.08,
        "temp_sensitivity": 0.012,
        "time_sensitivity": 0.003,
        "shock_sensitivity": 0.05  # Most sensitive to shock (bruising)
    },
    "apple_gala": {
        "base_chance": 0.01,
        "temp_sensitivity": 0.005,  # Very hardy
        "time_sensitivity": 0.0005,
        "shock_sensitivity": 0.01
    }
}

# --- Main Data Generation Function ---
def generate_data(num_rows):
    """Generates the full dataset based on product profiles and realistic scenarios."""
    data = []
    product_list = list(PRODUCT_PROFILES.keys())

    for i in range(num_rows):
        # 1. Select a random product
        sku_id = random.choice(product_list)
        profile = PRODUCT_PROFILES[sku_id]

        # 2. Generate realistic base features
        shipment_id = f"SHP-{i+1:03d}"
        
        # Transit hours can range from short trips to long hauls
        transit_hours = round(np.random.uniform(5, 120), 1) 
        
        # Average temperature can vary widely
        avg_temp = round(np.random.uniform(-5, 35), 1) 
        
        # Shock events are relatively rare
        shock_events = np.random.randint(0, 10)

        # 3. Calculate spoilage probability based on profile and conditions
        # Start with the base chance for the product
        spoilage_prob = profile["base_chance"]

        # Add impact of temperature (this is the dominant factor)
        # Use a non-linear effect (e.g., spoilage increases faster at higher temps)
        spoilage_prob += profile["temp_sensitivity"] * max(0, avg_temp - 5)**1.5

        # Add impact of transit time
        spoilage_prob += profile["time_sensitivity"] * transit_hours

        # Add impact of shock events
        spoilage_prob += profile["shock_sensitivity"] * shock_events
        
        # Add some random noise to make it more realistic
        spoilage_prob += np.random.normal(0, 0.05)

        # 4. Determine the final spoilage_flag
        # Clamp probability between 0 and 1, then decide the outcome
        spoilage_prob = np.clip(spoilage_prob, 0, 1)
        spoilage_flag = 1 if random.random() < spoilage_prob else 0

        # 5. Append the record
        data.append([
            shipment_id,
            sku_id,
            transit_hours,
            avg_temp,
            shock_events,
            spoilage_flag
        ])

    # Create DataFrame
    columns = [
        "shipment_id", "sku_id", "transit_hours",
        "avg_temp", "shock_events", "spoilage_flag"
    ]
    df = pd.DataFrame(data, columns=columns)
    return df

# --- Execution ---
if __name__ == "__main__":
    print(f"Generating {NUM_ROWS} rows of spoilage data...")
    
    dataset = generate_data(NUM_ROWS)
    
    # Save to CSV
    dataset.to_csv(FILE_NAME, index=False)
    
    print(f"Successfully created '{FILE_NAME}'!")
    
    # Optional: Print a summary of the generated data
    print("\n--- Data Summary ---")
    print(f"Total rows: {len(dataset)}")
    print("\nSpoilage Counts:")
    print(dataset['spoilage_flag'].value_counts())
    print("\nSKU Distribution:")
    print(dataset['sku_id'].value_counts())
    print("\nSample of 5 rows:")
    print(dataset.head())
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def create_demand_forecasting_dataset():
    """
    Generates a synthetic dataset for a demand forecasting problem.
    The dataset includes base demand, trend, seasonality, and noise.
    """
    # 1. Define Entities
    skus = ['mango123', 'apple456', 'banana789']
    locations = ['Mumbai', 'Delhi', 'Bangalore']

    base_demand = {
        ('mango123', 'Mumbai'): 50, ('mango123', 'Delhi'): 30, ('mango123', 'Bangalore'): 45,
        ('apple456', 'Mumbai'): 80, ('apple456', 'Delhi'): 95, ('apple456', 'Bangalore'): 70,
        ('banana789', 'Mumbai'): 40, ('banana789', 'Delhi'): 55, ('banana789', 'Bangalore'): 35,
    }

    # 2. Define Timeframe and Patterns
    num_days = 60
    start_date = datetime.now() - timedelta(days=num_days)
    date_range = pd.to_datetime([start_date + timedelta(days=x) for x in range(num_days)])

    trend_factor = 0.5
    weekly_seasonality = {
        0: 1.0, 1: 0.9, 2: 1.05, 3: 1.1, 4: 1.4, 5: 1.6, 6: 1.1
    }

    # 3. Generate Data Rows
    data_rows = []
    for date in date_range:
        for sku in skus:
            for location in locations:
                base = base_demand.get((sku, location), 30)
                days_from_start = (date - start_date).days
                trend = trend_factor * days_from_start
                seasonality_multiplier = weekly_seasonality[date.dayofweek]
                noise = np.random.randint(-15, 16)
                
                quantity = int(base * seasonality_multiplier + trend + noise)
                quantity = max(0, quantity) # Ensure non-negative sales

                data_rows.append({
                    'order_date': date.strftime('%Y-%m-%d'),
                    'sku_id': sku,
                    'location': location,
                    'quantity': quantity
                })
    
    # 4. Create DataFrame and Save
    demand_df = pd.DataFrame(data_rows)
    output_filename = 'synthetic_demand_data.csv'
    demand_df.to_csv(output_filename, index=False)

    print(f"Dataset with {len(demand_df)} rows successfully created.")
    print(f"Saved to '{output_filename}'")
    
    # 5. Display sample data
    print("\n--- Data Head ---")
    print(demand_df.head())
    print("\n--- Data Tail ---")
    print(demand_df.tail())
    
    return demand_df

# Run the function to create the dataset
if __name__ == '__main__':
    df = create_demand_forecasting_dataset()
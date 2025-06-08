import streamlit as st
import pandas as pd
import json
import random
# import joblib
import pickle

# --- Page Configuration ---
st.set_page_config(
    page_title="Supply Chain AI Demo",
    page_icon="🚚",
    layout="wide"
)

# --- Main App Title ---
st.title("🔗 Supply Chain AI Prototype Demo")

# --- Sidebar for Navigation ---
st.sidebar.title("Track Options")
app_mode = st.sidebar.radio(
    "Choose a Supply Chain Feature:",
    ("Demand Forecasting", "Spoilage Prediction", "ETA Prediction")
)

# ==============================================================================
# --- Option A: Demand Forecasting (REVISED AND IMPROVED) ---
# ==============================================================================
if app_mode == "Demand Forecasting":
    st.header("📈 Demand Forecasting")
    st.markdown("Predict next 7-day SKU-level demand based on past order data.")
    st.info("This demo uses a pre-trained model's results. Select an SKU and Location to view the forecast.")

    # --- Load Pre-computed Forecast Data ---
    try:
        with open('forecast_output.json', 'r') as f:
            forecast_data = json.load(f)
        
        # Extract unique SKUs and locations for the dropdowns
        sku_id_options = sorted(list(set([item['sku_id'] for item in forecast_data])))
        location_options = sorted(list(set([item['location'] for item in forecast_data])))

    except FileNotFoundError:
        st.error("Error: `forecast_output.json` not found. Please create this file with your forecast results.")
        st.stop()

    # --- User Input Section ---
    st.subheader("Select Inputs to View Forecast")
    col1, col2 = st.columns(2)
    with col1:
        selected_sku = st.selectbox("Select SKU ID:", sku_id_options, key="df_sku")
    with col2:
        selected_location = st.selectbox("Select Location:", location_options, key="df_loc")
    
    # --- Display Forecast on Button Click ---
    if st.button("Forecast Next 7 Days", key="df_button"):
        
        # Find the matching forecast from our loaded data
        result = next((item for item in forecast_data if item["sku_id"] == selected_sku and item["location"] == selected_location), None)

        st.subheader("Forecast Results")
        if result:
            st.json(result)

            # Display a chart for better visualization
            chart_data = pd.DataFrame({
                'Day': [f"Day {i+1}" for i in range(7)],
                'Forecasted Quantity': result["forecast_next_7_days"]
            })
            st.line_chart(chart_data.set_index('Day'))
            st.success("Forecast retrieved from pre-computed results!")
        else:
            st.warning("No forecast found for the selected SKU and Location combination.")
            
    # --- Optional: Show an example of the input data format ---
    with st.expander("Show an example of the training data format"):
        st.markdown("The model was trained on 60 days of historical data in a CSV format like this:")
        st.code("""
order_date,sku_id,location,quantity
2024-03-01,mango123,Mumbai,100
2024-03-01,apple456,Delhi,152
...
        """, language="csv")

# ==============================================================================
# --- Option B: Spoilage Prediction (FINAL, DEPLOYED VERSION) ---
# ==============================================================================
elif app_mode == "Spoilage Prediction":
    st.header("🌿 Spoilage Prediction")
    st.markdown("Predict the probability that a shipment will spoil based on transit time and temperature logs.")

    # --- LOAD THE TRAINED MODEL AND COLUMNS (uses caching for efficiency) ---
    @st.cache_resource
    def load_model_assets():
        """Loads the trained model and the column list."""
        try:
            # Load the model saved with pickle
            with open('src/training/spoilage_model.pkl', 'rb') as f_model:
                model = pickle.load(f_model)
            
            # Load the columns list
            with open('src/training/spoilage_columns.json', 'r') as f_cols:
                columns = json.load(f_cols)
                
            return model, columns
        except FileNotFoundError:
            st.error("Error: Model asset files ('spoilage_model.pkl' or 'spoilage_columns.json') not found. Please ensure they are in the root of your repository and have been pushed to GitHub.")
            return None, None
        
    ### <<< MOVED THIS SECTION UP >>> ###
    # First, load the assets to define the variables
    model, model_columns = load_model_assets()
    
    # If loading failed, stop the app gracefully
    if model is None or model_columns is None:
        st.stop()
    ### <<< END OF MOVED SECTION >>> ###
        
    st.subheader("Inputs for a New Shipment")
    shipment_id = st.text_input("Shipment ID (e.g., SHP-NEW):", "SHP-NEW-001", key="sp_ship_id")
    
    # Get the list of unique SKUs from the model columns for the dropdown
    # This line will now work because 'model_columns' was defined above
    sku_options = [col.replace('sku_', '') for col in model_columns if 'sku_' in col]
    sku_id_sp = st.selectbox("SKU ID:", sorted(sku_options), key="sp_sku_id")
    
    transit_hours = st.slider("Transit Hours:", min_value=1.0, max_value=200.0, value=24.5, step=0.5, key="sp_hours")
    avg_temp = st.slider("Average Temperature (°C):", min_value=-10.0, max_value=40.0, value=28.2, step=0.1, key="sp_temp")
    shock_events = st.number_input("Number of Shock Events:", min_value=0, max_value=10, value=1, step=1, key="sp_shocks")

    # --- PREDICTION LOGIC ---
    if st.button("Predict Spoilage Risk", key="sp_button"):
        
        # 1. Create a dictionary from the user's input
        input_data = {
            'transit_hours': [transit_hours],
            'avg_temp': [avg_temp],
            'shock_events': [shock_events],
            'sku_id': [sku_id_sp]  # The user-selected SKU
        }
        
        # 2. Convert to a DataFrame
        input_df = pd.DataFrame(input_data)
        
        # 3. Apply the SAME feature engineering as in training
        input_df['temp_x_hours'] = input_df['avg_temp'] * input_df['transit_hours']
        input_df['temp_squared'] = input_df['avg_temp']**2
        
        # 4. One-Hot Encode the SKU
        input_encoded = pd.get_dummies(input_df, columns=['sku_id'], prefix='sku')
        
        # 5. Align columns with the trained model's columns
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
        
        # 6. Make the prediction
        spoilage_prob = model.predict_proba(input_aligned)[0][1] # Probability of class '1' (spoilage)

        # --- DISPLAY RESULTS ---
        st.subheader("Prediction Results")
        
        # Display the performance metrics you got from your evaluation run
        st.write(f"**Trained Model Performance (on test set):**")
        st.metric(label="Accuracy", value="76.0%") # Use your actual score
        st.metric(label="ROC-AUC Score", value="0.868") # Use your actual score

        st.write(f"**Prediction for {shipment_id}:**")
        st.metric(label="Predicted Spoilage Probability", value=f"{spoilage_prob*100:.1f}%")

        # Your excellent summary text logic remains the same here!
        summary_text = f"This shipment ({shipment_id} for {sku_id_sp}) has an estimated "
        if spoilage_prob > 0.7:
            summary_text += f"{spoilage_prob*100:.0f}% spoilage risk, likely due to high temperature ({avg_temp}°C) and/or long transit ({transit_hours} hrs)."
            st.error(f"**Summary:** {summary_text}")
        elif spoilage_prob > 0.4:
            summary_text += f"{spoilage_prob*100:.0f}% spoilage risk. Conditions seem moderate."
            st.warning(f"**Summary:** {summary_text}")
        else:
            summary_text += f"{spoilage_prob*100:.0f}% spoilage risk. Conditions seem favorable."
            st.success(f"**Summary:** {summary_text}")

# ==============================================================================
# --- Option C: ETA Prediction ---
# ==============================================================================
elif app_mode == "ETA Prediction":
    st.header("⏱️ ETA Prediction")
    st.markdown("Estimate expected delivery time for a shipment based on past trip data.")

    st.subheader("Inputs for a New Trip")
    route_id = st.text_input("Route ID (e.g., R1):", "R-NEW-77", key="eta_route")
    distance_km = st.slider("Distance (km):", min_value=10.0, max_value=2000.0, value=320.0, step=10.0, key="eta_dist")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        vehicle_type = st.selectbox("Vehicle Type:", ["van", "truck_small", "truck_large"], key="eta_vehicle")
    with col2:
        weather = st.selectbox("Weather:", ["clear", "rain", "light_snow", "foggy"], key="eta_weather")
    with col3:
        load_type = st.selectbox("Load Type:", ["light", "medium", "heavy"], key="eta_load")

    if st.button("Predict ETA", key="eta_button"):
        st.subheader("Prediction Results (Dummy)")

        # Dummy prediction logic
        base_hours = distance_km / 55 # Avg speed 55 km/h
        if vehicle_type == "truck_large": base_hours *= 1.2
        if weather == "rain": base_hours *= 1.15
        if weather == "light_snow": base_hours *= 1.3
        if load_type == "heavy": base_hours *= 1.1
        
        predicted_hours = round(base_hours * random.uniform(0.95, 1.05), 1)
        confidence_margin = round(predicted_hours * 0.12, 1) # 12% margin

        dummy_eta_data = {
            "route_id": route_id,
            "predicted_eta_hours": predicted_hours,
            "confidence": f"±{confidence_margin} hrs"
        }
        st.json(dummy_eta_data)
        st.success("Dummy ETA prediction generated!")

# --- Footer ---
st.sidebar.markdown("---")
st.sidebar.info("This is a prototype for the AI Developer Assignment.")
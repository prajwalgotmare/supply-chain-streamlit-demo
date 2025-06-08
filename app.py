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

    # --- LOAD THE TRAINED MODEL, COLUMNS, AND METRICS ---
    @st.cache_resource
    def load_model_assets():
        """Loads the model, columns, and performance metrics."""
        try:
            with open('src/training/spoilage_model.pkl', 'rb') as f_model:
                model = pickle.load(f_model)
            
            with open('src/training/spoilage_columns.json', 'r') as f_cols:
                columns = json.load(f_cols)

            # NEW: Load the metrics file
            with open('src/training/spoilage_metrics.json', 'r') as f_metrics:
                metrics = json.load(f_metrics)
                
            return model, columns, metrics
        except FileNotFoundError as e:
            st.error(f"Error: A required model file was not found. Please run the training script. Missing file: {e.filename}")
            return None, None, None
        
    model, model_columns, metrics = load_model_assets()
    
    if model is None:
        st.stop()
        
    # --- Display Model Performance First ---
    st.subheader("Model Performance")
    st.markdown("These scores were calculated on a hold-out test set during model development.")
    
    col1, col2 = st.columns(2)
    with col1:
        # Display Accuracy, formatted as a percentage
        st.metric(label="Accuracy", value=f"{metrics['accuracy']:.1%}")
    with col2:
        # Display ROC-AUC score
        st.metric(label="ROC-AUC Score", value=f"{metrics['roc_auc']:.3f}")

    # --- User Input Section (no changes here) ---
    st.subheader("Inputs for a New Shipment")
    sku_options = [col.replace('sku_', '') for col in model_columns if col.startswith('sku_')]
    sku_id_sp = st.selectbox("SKU ID:", sorted(sku_options), key="sp_sku_id")
    transit_hours = st.slider("Transit Hours:", 1.0, 200.0, 24.5, 0.5, key="sp_hours")
    avg_temp = st.slider("Average Temperature (°C):", -10.0, 40.0, 28.2, 0.1, key="sp_temp")
    shock_events = st.number_input("Number of Shock Events:", 0, 10, 1, 1, key="sp_shocks")

    # --- Prediction Logic and Result Display (no changes here) ---
    if st.button("Predict Spoilage Risk", key="sp_button"):
        input_df = pd.DataFrame({
            'transit_hours': [transit_hours], 'avg_temp': [avg_temp],
            'shock_events': [shock_events], 'sku_id': [sku_id_sp]
        })
        input_df['temp_x_hours'] = input_df['avg_temp'] * input_df['transit_hours']
        input_df['temp_squared'] = input_df['avg_temp']**2
        input_encoded = pd.get_dummies(input_df, columns=['sku_id'], prefix='sku')
        input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
        spoilage_prob = model.predict_proba(input_aligned)[0][1]

        st.subheader("Prediction Results")
        st.metric(label="Predicted Spoilage Probability", value=f"{spoilage_prob*100:.1f}%")
        
        if spoilage_prob > 0.7:
             st.error(f"**Summary:** High risk ({spoilage_prob*100:.0f}%) detected.")
        elif spoilage_prob > 0.4:
             st.warning(f"**Summary:** Moderate risk ({spoilage_prob*100:.0f}%) detected.")
        else:
             st.success(f"**Summary:** Low risk ({spoilage_prob*100:.0f}%) detected.")

# ==============================================================================
# --- Option C: ETA Prediction (FINAL, DEPLOYED VERSION) ---
# ==============================================================================
elif app_mode == "ETA Prediction":
    st.header("⏱️ ETA Prediction")
    st.markdown("Estimate expected delivery time for a shipment based on past trip data.")

    # --- LOAD THE TRAINED MODEL AND ASSETS ---
    # @st.cache_resource is a Streamlit decorator that loads this data only once,
    # making the app much faster.
    @st.cache_resource
    def load_eta_model_assets():
        """Loads the trained model, column list, and performance metrics."""
        try:
            with open('src/training/eta_model.pkl', 'rb') as f_model:
                model = pickle.load(f_model)
            
            with open('src/training/eta_model_assets.json', 'r') as f_assets:
                assets = json.load(f_assets)
                
            model_columns = assets["model_columns"]
            confidence_mae = assets["performance_metrics"]["confidence_mae_hours"]
            return model, model_columns, confidence_mae
        except FileNotFoundError:
            st.error(
                "Error: Model files ('eta_model.pkl', 'eta_model_assets.json') not found. "
                "Please run the training script first to generate them."
            )
            return None, None, None
        
    model, model_columns, confidence_mae = load_eta_model_assets()
    
    # If files were not found, stop the app from running further.
    if model is None:
        st.stop()
        
    # --- Display Model Performance ---
    st.info(f"This model predicts ETAs with an average confidence of **±{confidence_mae} hours** (based on MAE).")

    # --- User Input Section (UI remains the same) ---
    st.subheader("Inputs for a New Trip")
    route_id = st.text_input("Route ID (e.g., R-NEW-77):", "R-NEW-77", key="eta_route")
    distance_km = st.slider("Distance (km):", min_value=10.0, max_value=2000.0, value=320.0, step=10.0, key="eta_dist")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        # These lists should match the categories used in your training data
        vehicle_type = st.selectbox("Vehicle Type:", ["van", "truck_small", "truck_large"], key="eta_vehicle")
    with col2:
        weather = st.selectbox("Weather:", ["clear", "rain", "light_snow", "foggy"], key="eta_weather")
    with col3:
        load_type = st.selectbox("Load Type:", ["light", "medium", "heavy"], key="eta_load")

    # --- Prediction Logic (This is the core change) ---
    if st.button("Predict ETA", key="eta_button"):
        
        # 1. Create a DataFrame from user inputs
        input_data = {
            'distance_km': [distance_km],
            'vehicle_type': [vehicle_type],
            'weather': [weather],
            'load_type': [load_type]
        }
        input_df = pd.DataFrame(input_data)

        # 2. Preprocess the input to match the model's training format
        # a) One-hot encode the categorical features
        input_processed = pd.get_dummies(input_df)
        
        # b) Reindex to ensure it has the exact same columns as the training data
        #    This is a CRITICAL step. It adds any missing columns and fills them with 0.
        input_aligned = input_processed.reindex(columns=model_columns, fill_value=0)

        # 3. Make the prediction
        prediction_array = model.predict(input_aligned)
        predicted_hours = round(prediction_array[0], 1)

        # 4. Format the final output as required
        final_output = {
            "route_id": route_id,
            "predicted_eta_hours": predicted_hours,
            "confidence": f"±{confidence_mae} hrs"
        }
        
        st.subheader("Prediction Results")
        st.json(final_output)
        st.success("ETA prediction generated successfully from the trained model!")
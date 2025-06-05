import streamlit as st
import pandas as pd # We'll use it for potential future data display
import json
import random # For generating dummy forecast values

# --- Page Configuration (Optional but good practice) ---
st.set_page_config(
    page_title="Supply Chain AI Demo",
    page_icon="🚚",
    layout="wide"
)

# --- Main App Title ---
st.title("🔗 Supply Chain AI Prototype Demo")
st.markdown("Choose a feature from the sidebar to explore.")

# --- Sidebar for Navigation ---
st.sidebar.title("Track Options")
app_mode = st.sidebar.radio(
    "Choose a Supply Chain Feature:",
    ("Demand Forecasting", "Spoilage Prediction", "ETA Prediction")
)

# --- Helper function for dummy forecast ---
def generate_dummy_forecast():
    return [random.randint(50, 150) for _ in range(7)]

# ==============================================================================
# --- Option A: Demand Forecasting ---
# ==============================================================================
if app_mode == "Demand Forecasting":
    st.header("📈 Demand Forecasting")
    st.markdown("Predict next 7-day SKU-level demand based on past order data.")

    st.subheader("Inputs")
    # Dummy input - in reality, you'd process this file
    uploaded_file = st.file_uploader("Upload 60-day order data (CSV)", type="csv")
    if uploaded_file:
        st.success(f"File '{uploaded_file.name}' uploaded (dummy processing).")
        # For dummy, we won't parse. Let's assume some SKUs and locations.
        sku_id_options = ["mango123", "apple456", "banana789"]
        location_options = ["Mumbai", "Delhi", "Bangalore"]
        selected_sku = st.selectbox("Select SKU ID:", sku_id_options, key="df_sku")
        selected_location = st.selectbox("Select Location:", location_options, key="df_loc")
    else:
        st.info("Please upload a CSV file to simulate SKU/Location selection.")
        selected_sku = "mango123" # Default for display if no file
        selected_location = "Mumbai" # Default for display if no file

    if st.button("Forecast Next 7 Days", key="df_button"):
        if uploaded_file or (selected_sku and selected_location): # Allow forecast if defaults are used or file uploaded
            st.subheader("Forecast Results (Dummy)")
            dummy_forecast_data = {
                "sku_id": selected_sku,
                "location": selected_location,
                "forecast_next_7_days": generate_dummy_forecast()
            }
            st.json(dummy_forecast_data)
            st.success("Dummy forecast generated!")

            # Optional: Dummy chart
            if "forecast_next_7_days" in dummy_forecast_data:
                chart_data = pd.DataFrame({
                    'Day': [f"Day {i+1}" for i in range(7)],
                    'Forecasted Quantity': dummy_forecast_data["forecast_next_7_days"]
                })
                st.line_chart(chart_data.set_index('Day'))
        else:
            st.warning("Please upload a file or ensure SKU/Location are selected.")


# ==============================================================================
# --- Option B: Spoilage Prediction ---
# ==============================================================================
elif app_mode == "Spoilage Prediction":
    st.header("🌿 Spoilage Prediction")
    st.markdown("Predict the probability that a shipment will spoil based on transit time and temperature logs.")

    st.subheader("Inputs for a New Shipment (Dummy)")
    shipment_id = st.text_input("Shipment ID (e.g., SHP-NEW):", "SHP-DUMMY-001", key="sp_ship_id")
    sku_id_sp = st.text_input("SKU ID (e.g., banana78):", "banana_fresh", key="sp_sku_id")
    transit_hours = st.number_input("Transit Hours:", min_value=1.0, max_value=200.0, value=24.5, step=0.5, key="sp_hours")
    avg_temp = st.number_input("Average Temperature (°C):", min_value=-10.0, max_value=40.0, value=28.2, step=0.1, key="sp_temp")
    shock_events = st.number_input("Number of Shock Events:", min_value=0, max_value=10, value=1, step=1, key="sp_shocks")

    if st.button("Predict Spoilage Risk", key="sp_button"):
        st.subheader("Prediction Results (Dummy)")
        dummy_accuracy = 0.85 # Example model accuracy
        dummy_roc_auc = 0.92  # Example model ROC-AUC

        st.write(f"**Trained Model Performance (Dummy):**")
        st.write(f"- Accuracy: {dummy_accuracy*100:.1f}%")
        st.write(f"- ROC-AUC Score: {dummy_roc_auc:.2f}")

        dummy_spoilage_prob = random.uniform(0.05, 0.95) # Random probability
        st.write(f"**Prediction for {shipment_id}:**")
        st.metric(label="Predicted Spoilage Probability", value=f"{dummy_spoilage_prob*100:.1f}%")

        # Optional summary
        summary_text = f"This shipment ({shipment_id} for {sku_id_sp}) has an estimated "
        if dummy_spoilage_prob > 0.7:
            summary_text += f"{dummy_spoilage_prob*100:.0f}% spoilage risk, likely due to high temperature ({avg_temp}°C) and/or long transit ({transit_hours} hrs)."
        elif dummy_spoilage_prob > 0.4:
            summary_text += f"{dummy_spoilage_prob*100:.0f}% spoilage risk. Conditions seem moderate."
        else:
            summary_text += f"{dummy_spoilage_prob*100:.0f}% spoilage risk. Conditions seem favorable."
        st.info(f"**Optional Summary (Dummy):**\n{summary_text}")
        st.success("Dummy spoilage prediction generated!")

# ==============================================================================
# --- Option C: ETA Prediction ---
# ==============================================================================
elif app_mode == "ETA Prediction":
    st.header("⏱️ ETA Prediction")
    st.markdown("Estimate expected delivery time for a shipment based on past trip data.")

    st.subheader("Inputs for a New Trip (Dummy)")
    route_id = st.text_input("Route ID (e.g., R1):", "R-DUMMY-77", key="eta_route")
    distance_km = st.number_input("Distance (km):", min_value=10.0, max_value=5000.0, value=320.0, step=10.0, key="eta_dist")
    vehicle_type_options = ["van", "truck_small", "truck_large", "bike"]
    vehicle_type = st.selectbox("Vehicle Type:", vehicle_type_options, key="eta_vehicle")
    weather_options = ["clear", "rain", "light_snow", "foggy"]
    weather = st.selectbox("Weather:", weather_options, key="eta_weather")
    load_type_options = ["light", "medium", "heavy", "empty"]
    load_type = st.selectbox("Load Type:", load_type_options, key="eta_load")

    if st.button("Predict ETA", key="eta_button"):
        st.subheader("Prediction Results (Dummy)")

        # Dummy prediction logic
        base_hours = distance_km / 50 # Simple base calculation
        if vehicle_type == "truck_large": base_hours *= 1.2
        if weather == "rain": base_hours *= 1.15
        if load_type == "heavy": base_hours *= 1.1
        predicted_hours = round(base_hours * random.uniform(0.9, 1.1), 1) # Add some randomness
        confidence_margin = round(predicted_hours * 0.1, 1) # 10% margin for dummy

        dummy_eta_data = {
            "route_id": route_id,
            "predicted_eta_hours": predicted_hours,
            "confidence": f"±{confidence_margin} hrs"
        }
        st.json(dummy_eta_data)
        st.success("Dummy ETA prediction generated!")

# --- Footer (Optional) ---
st.sidebar.markdown("---")
st.sidebar.info("This is a prototype for the AI Developer Assignment. ML models are not yet integrated.")
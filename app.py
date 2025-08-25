import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load the trained model, scaler, and label encoder
model = joblib.load("xgboost_failure_mode.pkl")
scaler = joblib.load("scaler.pkl")
label_encoder = joblib.load("label_encoder.pkl")

# Define feature names based on your dataset
feature_names = [
    "Temperature (°C)", "Vibration (mm/s)", "Pressure (bar)", "Rotational Speed (RPM)",
    "Torque (Nm)", "Tool Wear (%)", "Humidity (%)", "Energy Consumption (kWh)",
    "Operating Hours", "Load (%)", "Ambient Light (lux)"
]

# Streamlit UI
st.title("🚀 Machine Failure Mode Prediction")
st.write("Enter machine parameters below to predict the failure mode.")

# Collect user inputs (no min/max limits)
inputs = {}
for feature in feature_names:
    inputs[feature] = st.number_input(f"{feature}", value=0.0)

# Predict button
if st.button("🔍 Predict Failure Mode"):
    # Convert inputs to NumPy array
    new_data = np.array([[inputs[feature] for feature in feature_names]])
    
    # Scale the input
    new_data_scaled = scaler.transform(new_data)

    # Predict failure mode
    prediction = model.predict(new_data_scaled)
    predicted_label = label_encoder.inverse_transform(prediction)

    # Display result
    st.success(f"💡 Predicted Failure Mode: **{predicted_label[0]}**")

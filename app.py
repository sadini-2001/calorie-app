import streamlit as st
import numpy as np
import pandas as pd
import joblib

# Load trained model
model = joblib.load("model.pkl")

st.title("BurnMeter: Precision Calorie Tracking")
st.header("Calorie Burn Predictor")

# Wrap inputs inside a form
with st.form("calorie_form"):
    Gender = st.selectbox("Select Gender:", ["male", "female"])
    Age = st.number_input("Enter Age (years):", min_value=1, step=1)
    Height = st.number_input("Enter Height (cm):", min_value=50, step=1)
    Weight = st.number_input("Enter Weight (kg):", min_value=10, step=1)
    Duration = st.number_input("Enter Workout Duration (minutes):", min_value=1, step=1)
    Heart_Rate = st.number_input("Enter Heart Rate (bpm):", min_value=30, step=1)
    Body_Temp = st.number_input("Enter Body Temperature (°C):", min_value=30.0, max_value=45.0, step=0.1)

    # Submit button inside the form
    submit = st.form_submit_button("Predict Calories Burned")

# Perform prediction when the button is clicked
if submit:
    # ✅ Convert input to a DataFrame with column names
    input_data = pd.DataFrame([[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]],
                              columns=["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"])

    # Get the prediction
    prediction = model.predict(input_data)

    # Display the result
    st.success(f"Estimated Calories Burned: **{prediction[0]:.2f}** kcal")


import streamlit as st
import numpy as np
import pandas as pd
import joblib
import mysql.connector

# Load trained model
model = joblib.load("model.pkl")

# Function to connect to MySQL database
def get_db_connection():
    return mysql.connector.connect(
        host="your_host",
        user="your_user",
        password="your_password",
        database="your_database"
    )

# Function to authenticate user
def get_user_id(username, password):
    conn = get_db_connection()
    cursor = conn.cursor()
    query = "SELECT user_id FROM users WHERE username = %s AND password = %s"
    cursor.execute(query, (username, password))
    result = cursor.fetchone()
    cursor.close()
    conn.close()
    return result[0] if result else None  # Return user_id if found

# Function to insert data into user_data table
def insert_data(user_id, gender, age, height, weight, duration, heart_rate, body_temp, calories):
    try:
        conn = get_db_connection()
        cursor = conn.cursor()

        query = """
        INSERT INTO user_data (user_id, gender, age, height, weight, duration, heart_rate, body_temp, calories_burned)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = (user_id, gender, age, height, weight, duration, heart_rate, body_temp, calories)

        cursor.execute(query, values)
        conn.commit()
        cursor.close()
        conn.close()
        return True
    except Exception as e:
        st.error(f"Error inserting data: {e}")
        return False

# User login
st.title("BurnMeter: Precision Calorie Tracking")

if "user_id" not in st.session_state:
    st.subheader("Login")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")

    if login_button:
        user_id = get_user_id(username, password)
        if user_id:
            st.session_state["user_id"] = user_id  # Store user_id in session
            st.success("✅ Login successful!")
            st.experimental_rerun()  # Refresh page after login
        else:
            st.error("❌ Invalid username or password.")

# If logged in, show prediction form
if "user_id" in st.session_state:
    user_id = st.session_state["user_id"]
    st.header("Calorie Burn Predictor")

    # Form for user input
    with st.form("calorie_form"):
        Gender = st.selectbox("Select Gender:", ["male", "female"])
        Age = st.number_input("Enter Age (years):", min_value=1, step=1)
        Height = st.number_input("Enter Height (cm):", min_value=50, step=1)
        Weight = st.number_input("Enter Weight (kg):", min_value=10, step=1)
        Duration = st.number_input("Enter Workout Duration (minutes):", min_value=1, step=1)
        Heart_Rate = st.number_input("Enter Heart Rate (bpm):", min_value=30, step=1)
        Body_Temp = st.number_input("Enter Body Temperature (°C):", min_value=30.0, max_value=45.0, step=0.1)

        # Submit button inside form
        submit = st.form_submit_button("Predict Calories Burned")

    # Perform prediction and store data when button is clicked
    if submit:
        # Convert input to DataFrame
        input_data = pd.DataFrame([[Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp]],
                                  columns=["Gender", "Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp"])

        # Get prediction
        prediction = model.predict(input_data)
        calories_burned = round(prediction[0], 2)

        # Display the result
        st.success(f"Estimated Calories Burned: **{calories_burned}** kcal")

        # Store data in MySQL database
        if insert_data(user_id, Gender, Age, Height, Weight, Duration, Heart_Rate, Body_Temp, calories_burned):
            st.success("✅ Data successfully saved to the database!")

        # Logout option
        if st.button("Logout"):
            del st.session_state["user_id"]
            st.experimental_rerun()

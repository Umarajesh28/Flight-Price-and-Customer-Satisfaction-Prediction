import numpy as np
import pandas as pd
import pickle
import streamlit as st
from datetime import datetime

# Define the paths for the necessary files
SATISFACTION_BASE_PATH = r'C:\Users\user\Desktop\Flight_Customer_Prediction\Customer_Satisfaction_Prediction'
PRICE_BASE_PATH = r'C:\Users\user\Desktop\Flight_Customer_Prediction\Flight_Price_Prediction'

XGB_MODEL_PATH = f"{SATISFACTION_BASE_PATH}\\xgb_classifier.pkl"
SCALER_PATH = f"{SATISFACTION_BASE_PATH}\\scaler.pkl"
FEATURE_NAMES_PATH = f"{SATISFACTION_BASE_PATH}\\feature_names.pkl"

FLIGHT_MODEL_PATH = f"{PRICE_BASE_PATH}\\Flight.pkl"

# Load the trained models, scaler, and feature names
try:
    with open(XGB_MODEL_PATH, 'rb') as file:
        xgb_classifier = pickle.load(file)

    with open(SCALER_PATH, 'rb') as file:
        scaler = pickle.load(file)

    with open(FEATURE_NAMES_PATH, 'rb') as file:
        feature_names = pickle.load(file)

    with open(FLIGHT_MODEL_PATH, 'rb') as file:
        Flight = pickle.load(file)

except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Please ensure all necessary files are in the correct directories.")
    st.stop()

# Function to preprocess user input
def preprocess_input(data):
    # Perform one-hot encoding for categorical features
    data = pd.get_dummies(data, columns=['Gender', 'Customer Type', 'Type of Travel'], drop_first=True)

    # Map class categories to integers
    class_mapping = {"Eco": 0, "Eco Plus": 1, "Business": 2}
    data['Class'] = data['Class'].map(class_mapping)

    # Add missing features with default values
    for feature in feature_names:
        if feature not in data.columns:
            data[feature] = 0

    # Reorder the columns to match the training order
    data = data[feature_names]

    return data

# Function to make satisfaction predictions
def make_satisfaction_prediction(input_data):
    # Preprocess the input data
    processed_data = preprocess_input(input_data)

    # Scale the input data
    scaled_data = scaler.transform(processed_data)

    # Predict using the loaded model
    prediction = xgb_classifier.predict(scaled_data)
    return prediction

def flight_prediction(input_data):
    input_data_as_numpy_array = np.asarray(input_data)
    input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
    prediction = Flight.predict(input_data_reshaped)
    rounded_value = round(prediction[0], 2)
    return rounded_value

# Main function
def main():
    st.sidebar.title("Airline Prediction App")
    app_mode = st.sidebar.radio("Choose the app", ["Passenger Satisfaction Prediction", "Flight Price Prediction"])

    st.markdown("""
        <style>
        .sidebar .sidebar-content {
            background-color: #000000;
        }
        .stButton > button {
            background-color: #0a84ff;
            color: white;
            border-radius: 5px;
        }
        .stButton > button:hover {
            background-color: #00449e;
        }
        .main-title {
            color: #0a84ff;
            text-align: center;
        }
        .subheader {
            color: #555;
            text-align: center;
        }
        .success-message {
            background-color: #d4edda;
            color: #155724;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
            text-align: center;
        }
        .error-message {
            background-color: #f8d7da;
            color: #721c24;
            border-radius: 5px;
            padding: 10px;
            margin-top: 10px;
            text-align: center;
        }
        </style>
    """, unsafe_allow_html=True)

    if app_mode == "Passenger Satisfaction Prediction":
        st.markdown("<h1 class='main-title'>✈️ Passenger Satisfaction Prediction</h1>", unsafe_allow_html=True)
        st.markdown("<h3 class='subheader'>Input Passenger Details Below</h3>", unsafe_allow_html=True)

        # Collect user input
        customer_type = st.selectbox("Are you a New Customer for this airline?", ['Disloyal Customer', 'Loyal Customer'], index=1)
        type_travel = st.selectbox("What was the purpose of your flight?", ["Personal Travel", "Business Travel"])
        class1 = st.selectbox("What was the Class of your flight?", ['Eco', 'Business', 'Eco Plus'])
        gender = st.selectbox("Select your Gender", ['Male', 'Female'])
        online_boarding = st.radio("Satisfaction level for Online Boarding?", [0, 1, 2, 3, 4, 5], horizontal=True)
        inflight_wifi = st.radio("Satisfaction level of the Inflight Wifi Service?", [0, 1, 2, 3, 4, 5], horizontal=True)
        entertainment = st.radio("Satisfaction level of the Inflight Entertainment?", [0, 1, 2, 3, 4, 5], horizontal=True)
        seat_comfort = st.radio("Satisfaction level of the Seat Comfort?", [0, 1, 2, 3, 4, 5], horizontal=True)
        online_booking = st.radio("Satisfaction level of Ease of making an Online Booking?", [0, 1, 2, 3, 4, 5], horizontal=True)
        leg_room = st.radio("How would you rate the Leg Room Service?", [0, 1, 2, 3, 4, 5], horizontal=True)
        cleanliness = st.radio("How would you rate the Cleanliness?", [0, 1, 2, 3, 4, 5], horizontal=True)

        # Create a DataFrame from the user input
        input_data = pd.DataFrame({
            'Online boarding': online_boarding,
            'Inflight wifi service': inflight_wifi,
            'Type of Travel': type_travel,
            'Class': class1,
            'Gender': gender,
            'Inflight entertainment': entertainment,
            'Seat comfort': seat_comfort,
            'Ease of Online booking': online_booking,
            'Leg room service': leg_room,
            'Cleanliness': cleanliness,
            'Customer Type': customer_type
        }, index=[0])  # Added index to DataFrame creation

        # Button to make prediction
        if st.button("Predict Satisfaction"):
            prediction = make_satisfaction_prediction(input_data)
            if prediction[0] == 1:
                st.success("The passenger is satisfied.")
            elif prediction[0] == 0:
                st.error("The passenger is not satisfied.")
                

    elif app_mode == "Flight Price Prediction":
        st.markdown("<h1 class='main-title'>💸 Flight Price Prediction</h1>", unsafe_allow_html=True)

        sources = [f"Source_{source}" for source in ['Chennai', 'Delhi', 'Kolkata', 'Mumbai']]
        destinations = [f"Destination_{destination}" for destination in ['Cochin', 'Delhi', 'Hyderabad', 'Kolkata']]
        airlines = [f"Airline_{airline}" for airline in ['Air India', 'GoAir', 'IndiGo', 'Jet Airways', 'Jet Airways Business', 'Multiple carriers', 'Multiple carriers Premium economy', 'SpiceJet', 'Trujet', 'Vistara', 'Vistara Premium economy']]

        selected_source = st.selectbox('Select Source', [s.split("_")[1] for s in sources])
        source_mapping = {source: 1 if source.split("_")[1] == selected_source else 0 for source in sources}
        Source = list(source_mapping.values())

        selected_destination = st.selectbox('Select Destination', [d.split("_")[1] for d in destinations])
        destination_mapping = {destination: 1 if destination.split("_")[1] == selected_destination else 0 for destination in destinations}
        Destination = list(destination_mapping.values())

        dep_date = st.date_input("Departure Date")
        dep_time = st.time_input("Departure Time")

        arrival_date = st.date_input("Arrival Date")
        arrival_time = st.time_input("Arrival Time")

        Journey_Day = dep_date.day
        Journey_Month = dep_date.month

        Departure_Hour = dep_time.hour
        Departure_Min = dep_time.minute

        Arrival_Hour = arrival_time.hour
        Arrival_Min = arrival_time.minute

        Departure_Datetime = datetime.combine(dep_date, dep_time)
        Arrival_Datetime = datetime.combine(arrival_date, arrival_time)

        duration = Arrival_Datetime - Departure_Datetime
        Duration_Hours = duration.days * 24 + duration.seconds // 3600
        Duration_Minutes = (duration.seconds % 3600) // 60

        Total_Stops = st.number_input("Number of Stops", min_value=0, step=1, value=0)

        selected_airline = st.selectbox('Select Airline', [a.split("_")[1] for a in airlines])
        airline_mapping = {airline: 1 if airline.split("_")[1] == selected_airline else 0 for airline in airlines}
        Airlines = list(airline_mapping.values())

        journey_input = [Total_Stops, Journey_Day, Journey_Month, Departure_Hour, Departure_Min, Arrival_Hour, Arrival_Min, Duration_Hours, Duration_Minutes]
        airline_input = Airlines
        source_input = Source
        destination_input = Destination
        Input = journey_input + airline_input + source_input + destination_input

        st.write("### Flight Details Summary", pd.DataFrame(
            [Input],
            columns=[
                "Total Stops", "Journey Day", "Journey Month", "Departure Hour", "Departure Minute", 
                "Arrival Hour", "Arrival Minute", "Duration Hours", "Duration Minutes"
            ] + airlines + sources + destinations
        ))

        if st.button('Predict Price'):
            price_prediction = flight_prediction(Input)
            st.markdown(
                f"""
                <h3 style='text-align: center;'>Estimated Flight Price: ₹ {price_prediction}</h3>
                """, unsafe_allow_html=True
            )

        
if __name__ == "__main__":
    main()

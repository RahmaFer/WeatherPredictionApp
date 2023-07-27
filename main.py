import streamlit as st
from PIL import Image, ImageFilter
import datetime
import joblib
import requests
import speech_recognition as sr
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import pickle
import openai

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )


def get_weather_data(city):
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    api_key = "e44b093d9186c95a9e142492c85c308c"  # Replace with your OpenWeatherMap API key
    params = {
        "q": city,
        "appid": api_key,
        "units": "metric"  # You can change the units if desired
    }
    try:
        response = requests.get(base_url, params=params)
        data = response.json()
        if data["cod"] == 200:
            return data
        else:
            st.write("Error retrieving weather data.")
            return None
    except requests.exceptions.RequestException as e:
        st.write(f"Error: {e}")
        return None


# Process the weather data and display the forecast
def process_weather_data(data):
    temperature = data["main"]["temp"]
    description = data["weather"][0]["description"]
    humidity = data["main"]["humidity"]
    wind_speed = data["wind"]["speed"]

    st.write(f"Temperature: {temperature}°C")
    st.write(f"Description: {description}")
    st.write(f"Humidity: {humidity}%")
    st.write(f"Wind Speed: {wind_speed} m/s")


# Train your SVM model
def train_svm_model(X_test_scaled,selected_date):

    loaded_model = joblib.load('savedModel.pkl')
    # Use the loaded model to make predictions
    predictions = loaded_model.predict(X_test_scaled)
    if predictions >38:
        img = Image.open("high_temp.png")
        st.markdown(f'<p style="color: white;" >The temperature for <span style="color: yellow;font-weight:bold;">{ selected_date}</span> seems to be very high</p>', unsafe_allow_html=True)

        # Resize the image
        new_size = (100,100)
        resized_image = img.resize(new_size)

        col1, col2 = st.columns([1, 2])

        col1.image(resized_image, use_column_width=True)

        col2.markdown(f'<p style="color: yellow;font-size:50px;font-weight:bold;" >{predictions[0].round(0)}<span style="color: yellow;" font-weight:"bold;" style=“font-size:50px;”>°C</span> </p>', unsafe_allow_html=True)

        st.markdown(f'<p style="color: red;font-weight:bold;" >Tip: DO NOT  FORGET TO DRINK A LOT OF WATER. TAKE CARE</p>', unsafe_allow_html=True)

    elif predictions<=38 and predictions>=20:
        img = Image.open("moderate_temp.jpg")
        st.markdown(
            f'<p style="color: white;" font-weight:"bold;" >The temperature for <span style="color: yellow;font-weight:bold;">{selected_date}</span> seems to be moderate</p>',
            unsafe_allow_html=True)

        # Resize the image
        new_size = (100, 100)
        resized_image = img.resize(new_size)

        col1, col2 = st.columns([1, 2])

        col1.image(resized_image, use_column_width=True)

        col2.markdown(
            f'<p style="color: yellow;font-size:50px;" font-weight:"bold;" font-size="300px;">{predictions[0].round(0)}<span style="color: yellow;" font-weight:"bold;" style=“font-size:50px;”>°C</span> </p>',
            unsafe_allow_html=True)
    elif predictions < 20:
        img = Image.open("rainy.jpg")
        st.markdown(
            f'<p style="color: white;" font-weight:"bold;" >The temperature for <span style="color: yellow;font-weight:bold;">{selected_date}</span> seems to be rainy</p>',
            unsafe_allow_html=True)

        # Resize the image
        new_size = (100, 100)
        resized_image = img.resize(new_size)

        col1, col2 = st.columns([1, 2])

        col1.image(resized_image, use_column_width=True)

        col2.markdown(
            f'<p style="color: yellow;font-size:50px;" font-weight:"bold;" font-size="300px;">{predictions[0].round(0)}<span style="color: yellow;" font-weight:"bold;" style=“font-size:50px;”>°C</span> </p>',
            unsafe_allow_html=True)

#Speech recognition and weather update
def voice_input():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        st.write("Please speak...")
        audio = r.listen(source)
        st.write("Recognizing...")
    try:
        query = r.recognize_google(audio)
        st.write(f"You said: {query}")
        data = get_weather_data(query)
        process_weather_data(data)
    except sr.UnknownValueError:
        st.write("Sorry, I couldn't understand your voice.")
    except sr.RequestError as e:
        st.write(f"Speech recognition service error: {e}")


# Chatbot interaction
def chatbot_interaction():
    city = st.text_input("Enter a city name")
    if city:
        if st.button("Get Weather"):
            data = get_weather_data(city)
            process_weather_data(data)
    else:
        st.write("Please enter a city name.")


# SVM weather prediction
def svm_prediction():

    user_input = st.selectbox("Select a location, please: ",
					['CARTHAGE','GABES', 'GAFSA', 'JENDOUBA', 'KAIROUAN'])
    if user_input=="CARTHAGE":
        location_selected=3
    elif user_input=="GABES":
        location_selected=1
    elif user_input == "GAFSA":
        location_selected = 4
    elif user_input == "JENDOUBA":
        location_selected = 0
    elif user_input == "KAIROUAN":
        location_selected = 2

    TAVG = st.number_input("Enter the temperature average, please: ")
    TMIN = st.number_input("Enter the minimum temperature, please: ")

    today = datetime.date.today()
    tomorrow = today + datetime.timedelta(days=1)
    days5 = tomorrow + datetime.timedelta(days=4)



    selected_date = st.date_input("Select a date, please:", value=tomorrow,  min_value=tomorrow, max_value=days5)

    #Preprocess new data
    date = pd.to_datetime(selected_date)
    YEAR = date.year
    MONTH = date.month
    DAY = date.day
    parameter_default_values = [location_selected, TAVG, TMIN, YEAR, MONTH, DAY]
    test = pd.DataFrame([parameter_default_values], columns=['STATION', 'TAVG', 'TMIN', 'YEAR', 'MONTH', 'DAY'],
                        dtype=float)

    parameter_default_values1=[4,20,11,YEAR,MONTH,DAY]
    test1 = pd.DataFrame([parameter_default_values1], columns=['STATION', 'TAVG', 'TMIN', 'YEAR', 'MONTH', 'DAY'],
                        dtype=float)

    scaler = StandardScaler()
    scaler.fit(test1)
    X_test_scaled = scaler.transform(test)

    if st.button("Get Weather"):
        # Train the SVM model
        train_svm_model(X_test_scaled,selected_date)



def main():
    add_bg_from_local('weather.jpg')
    st.markdown('<h1 style="color: #772717 ;">Weather Prediction System</h1>', unsafe_allow_html=True)
    st.markdown('<p style="color: #946535 ;font-weight:bold;">Welcome to the Weather Prediction System!</p>', unsafe_allow_html=True)
    st.write("\n\n")

    st.markdown("""
    <style>
        [data-testid=stSidebar] {
            background-color: #F8F8FF;
        }
    </style>
    """, unsafe_allow_html=True)


    option = st.sidebar.selectbox("Select an option, please: ", ["SVM Weather Prediction", "Speech Recognition", "Chatbot Interaction" ])

    if option == "SVM Weather Prediction":
        svm_prediction()
    elif option == "Speech Recognition":
        voice_input()
    elif option == "Chatbot Interaction":
        chatbot_interaction()


if __name__ == "__main__":
    main()
import streamlit as st
import pickle as pkl
import pandas as pd
from wrapper_model import XGBWrapper, CTBWrapper

# Load the model, pca and scaler
with open('./models/stacking_model.pkl', 'rb') as file:
    stacking_model = pkl.load(file)

with open('./models/pca.pkl', 'rb') as file:
    pca = pkl.load(file)

with open('./models/scaler.pkl', 'rb') as file:
    scaler = pkl.load(file)

# load base model
with open('./models/ctb_model.pkl', 'rb') as file:
    ctb_model = pkl.load(file)

with open('./models/lgbm_model.pkl', 'rb') as file:
    lgbm = pkl.load(file)

with open('./models/xgb_model.pkl', 'rb') as file:
    xgb_model = pkl.load(file)

with open('./models/dc_model.pkl', 'rb') as file:
    dc = pkl.load(file)

with open('./models/svm_model.pkl', 'rb') as file:
    svm = pkl.load(file)

# Create the UI
st.title('Predicting the AQI Category')

st.write('This app predicts the AQI category based on the input features.')

# Input features
st.sidebar.title('Input Features')

st.sidebar.write('Please enter the following input features:')
pollutant_pm25 = st.sidebar.number_input('Pollutant_PM2.5_µg/m³', min_value=0.0, max_value=500.0, value=0.0)
pollutant_pm10 = st.sidebar.number_input('Pollutant_PM10_µg/m³', min_value=0.0, max_value=500.0, value=0.0)
pollutant_o3 = st.sidebar.number_input('Pollutant_O3_ppb', min_value=0.0, max_value=500.0, value=0.0)
pollutant_no2 = st.sidebar.number_input('Pollutant_NO2_ppb', min_value=0.0, max_value=500.0, value=0.0)
pollutant_co = st.sidebar.number_input('Pollutant_CO_ppm', min_value=0.0, max_value=500.0, value=0.0)
pollutant_so2 = st.sidebar.number_input('Pollutant_SO2_ppb', min_value=0.0, max_value=500.0, value=0.0)
urban_vegetation_area = st.sidebar.number_input('UrbanVegetationArea_m2', min_value=0.0, max_value=500.0, value=0.0)
humidity = st.sidebar.number_input('Humidity_%', min_value=0.0, max_value=100.0, value=0.0)
air_temperature = st.sidebar.number_input('AirTemperature_C', min_value=-50.0, max_value=50.0, value=0.0)
annual_energy_savings = st.sidebar.number_input('AnnualEnergySavings_%', min_value=0.0, max_value=100.0, value=0.0)
population_density = st.sidebar.number_input('PopulationDensity_people/km²', min_value=0.0, max_value=100000.0, value=0.0)
renewable_energy_percentage = st.sidebar.number_input('RenewableEnergyPercentage_%', min_value=0.0, max_value=100.0, value=0.0)
annual_energy_consumption = st.sidebar.number_input('AnnualEnergyConsumption_kWh', min_value=0.0, max_value=1000000.0, value=0.0)
green_space_index = st.sidebar.number_input('GreenSpaceIndex_%', min_value=0.0, max_value=100.0, value=0.0)
historic_pollutant_levels = st.sidebar.number_input('HistoricPollutantLevels', min_value=0.0, max_value=100.0, value=0.0)

# Scale the input features

input_features = [[pollutant_pm25, pollutant_pm10, pollutant_o3, pollutant_no2, pollutant_co, pollutant_so2,
                    urban_vegetation_area, humidity, air_temperature, annual_energy_savings, population_density,
                    renewable_energy_percentage, annual_energy_consumption, green_space_index, historic_pollutant_levels]]

scaled_input_features = scaler.transform(input_features)

# Apply PCA
pca_input_features = pca.transform(scaled_input_features)

# Predict the AQI category
prediction = stacking_model.predict(pca_input_features)
probability = stacking_model.predict_proba(pca_input_features)
probability = pd.DataFrame(probability, columns=stacking_model.classes_)
probability.index = ['probability']

# AQI Categories' color codes
color_codes = {
    'Good': '#00e400', 
    'Moderate': '#ff0', 
    'Unhealthy for Sensitive Groups': '#ff7e00',
    'Unhealthy': '#f00', 
    'Very Unhealthy': '#99004c'
}

# Display the prediction
st.title('Prediction')
predicted_category = prediction[0]
st.markdown(f'The predicted AQI category is: <span style="color:{color_codes[predicted_category]};">**{predicted_category}**</span>', unsafe_allow_html=True)

# Display the probability of each class
st.title('Prediction Probability')
st.write('The probability of each class is:')
st.write(probability)

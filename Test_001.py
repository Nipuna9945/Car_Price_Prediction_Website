

import streamlit as st
import pandas as pd
import joblib

# Load datasets
raw_data = pd.read_csv('test.csv')

# Load the trained machine learning model
model = joblib.load('RF_model.pkl')

# Define Streamlit UI
st.title('Car Price Prediction')

model_selection = st.selectbox('Select Car Model:', raw_data['model'].unique())
year = st.number_input('Enter Car Year:', min_value=int(raw_data['year'].min()), max_value=int(raw_data['year'].max()))
motor_type = st.selectbox('Select Motor Type:', raw_data['motor_type'].unique())
running = st.selectbox('Select Running Status:', raw_data['running'].unique())
color = st.selectbox('Select Car Color:', raw_data['color'].unique())
car_type = st.selectbox('Select Car Type:', raw_data['type'].unique())
status = st.selectbox('Select Car Status:', raw_data['status'].unique())
motor_volume = st.number_input('Enter Motor Volume:', min_value=float(raw_data['motor_volume'].min()), max_value=float(raw_data['motor_volume'].max()))

# Define Prediction Button
if st.button('Car Price Prediction'):
    # Perform one-hot encoding for categorical features
    prediction_features = pd.get_dummies(pd.DataFrame([[model_selection, year, motor_type, running, color, car_type, status, motor_volume]], 
                                                      columns=['model', 'year', 'motor_type', 'running', 'color', 'type', 'status', 'motor_volume']))
    
    # Make prediction using the loaded model
    predicted_price = model.predict(prediction_features)[0]
    st.success(f'Predicted Car Price: ${predicted_price}')

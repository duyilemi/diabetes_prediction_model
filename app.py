import streamlit as st
import numpy as np
import pandas as pd
import pickle
import joblib
import gspread
from oauth2client.service_account import ServiceAccountCredentials
# from sklearn.preprocessing import StandardScaler

# ------------- Google Sheets Setup ------------- #


# Define the scope for Google Sheets API
scope = ["https://spreadsheets.google.com/feeds", 'https://www.googleapis.com/auth/drive']

# Authenticate using the credentials JSON file
creds = ServiceAccountCredentials.from_json_keyfile_name('diabetes-prediction-437513-0041256b4309.json', scope)

# Authorize the client
client = gspread.authorize(creds)

# Open the Google Sheet
sheet = client.open('User Predictions').sheet1  # My Google Sheet name


# ------------- Function to log user input to Google Sheets Setup ------------- #

# List of field names (columns) from your app
field_names = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level', 'gender', 'hypertension', 'heart_disease', 'smoking_history', 'prediction']

# Function to log user input to Google Sheets
def log_user_input_to_google_sheet(data):
    try:
        # Check if the Google Sheet is empty
        values = sheet.get_all_values()
        if not values or (len(values) == 1 and not values[0]):
            # Write the column headers in the first row
            sheet.insert_row(field_names, 1)
        
        sheet.append_row(data)
    except Exception as e:
        st.error(f"An error occurred while logging to Google Sheets: {e}")
    
    
# Load the trained model
model = joblib.load('grid_search_xgb.pkl')



# List of columns after pandas.get_dummies() from training
model_columns = ['age', 'hypertension', 'heart_disease', 'bmi', 'HbA1c_level',
       'blood_glucose_level', 'gender_Male', 'smoking_history_current',
       'smoking_history_ever', 'smoking_history_former',
       'smoking_history_never', 'smoking_history_not current']



# Set up the Streamlit interface
st.title('Diabetes Prediction App')
st.write("""
This app predicts whether a person has diabetes based on the following medical factors. 
Please input the values for the features below.
""")

# Feature inputs based on the dataset
gender = st.selectbox('Gender', ('Male', 'Female'))
age = st.number_input('Age', min_value=1, max_value=80, value=35)
hypertension = st.selectbox('Hypertension (High Blood Pressure)', (0, 1))  # 0 for No, 1 for Yes
heart_disease = st.selectbox('Heart Disease', (0, 1))  # 0 for No, 1 for Yes
smoking_history = st.selectbox('Smoking History', ('No Info', 'former', 'never', 'not current', 'current', 'ever'))
bmi = st.number_input('BMI', min_value=10.19, max_value=95.69, value=25.0)
HbA1c_level = st.number_input('HbA1c Level (Blood Sugar)', min_value=3.5, max_value=9.0, value=5.5)
blood_glucose_level = st.number_input('Blood Glucose Level', min_value=80, max_value=300, value=300)

# Create a button for prediction
if st.button('Predict'):
    try:
        # Create a DataFrame to hold the input values
        input_data = pd.DataFrame({
            'gender': [gender],
            'age': [age],
            'hypertension': [hypertension],
            'heart_disease': [heart_disease],
            'smoking_history': [smoking_history],
            'bmi': [bmi],
            'HbA1c_level': [HbA1c_level],
            'blood_glucose_level': [blood_glucose_level]
        })

        # One-Hot Encode categorical variables to match the training process
        input_data_encoded = pd.get_dummies(input_data, drop_first=False)

        # Align the encoded input with the model's expected columns
        # Add any missing columns with default value of 0
        for col in model_columns:
            if col not in input_data_encoded.columns:
                input_data_encoded[col] = 0

        # Ensure the input data columns are in the same order as the model's columns
        input_data_encoded = input_data_encoded[model_columns]

        # Scale the input data (if scaling was applied during training)
        # input_data_scaled = scaler.transform(input_data_encoded)

        # Make the prediction
        prediction = model.predict(input_data_encoded)

        # Display the prediction result
        if prediction[0] == 1:
            st.write("The model predicts that the person has diabetes.")
        else:
            st.write("The model predicts that the person does not have diabetes.")
            
        # Log the input data and prediction result to Google Sheets
        user_data_to_log = [age, bmi, HbA1c_level, blood_glucose_level, gender, hypertension, heart_disease, smoking_history, int(prediction[0])]
        log_user_input_to_google_sheet(user_data_to_log)
        st.success("User input data logged successfully!")
        
        
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
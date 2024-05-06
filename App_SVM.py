import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load the trained SVM model
svc_model = joblib.load('mymodel.joblib')

# Set up Streamlit app
st.title('Loan Eligibility Prediction')

# Create input form for user
st.write('Enter Applicant Details:')
gender = st.selectbox('Gender', ['Male', 'Female'])
married = st.selectbox('Marital Status', ['Yes', 'No'])
dependents = st.text_input('Number of Dependents', '0')
education = st.selectbox('Education', ['Graduate', 'Not Graduate'])
self_employed = st.selectbox('Self Employed', ['Yes', 'No'])
applicant_income = st.text_input('Applicant Income', '')
coapplicant_income = st.text_input('Coapplicant Income', '')
loan_amount = st.text_input('Loan Amount', '')
loan_amount_term = st.text_input('Loan Amount Term', '')
credit_history = st.selectbox('Credit History', ['1', '0'])
property_area = st.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])

# Map categorical variables to numerical values
gender_map = {'Male': 1, 'Female': 0}
married_map = {'Yes': 1, 'No': 0}
education_map = {'Graduate': 1, 'Not Graduate': 0}
self_employed_map = {'Yes': 1, 'No': 0}
property_area_map = {'Urban': 1, 'Rural': 2, 'Semiurban': 0}

# Convert categorical variables to numerical values
gender_val = gender_map[gender]
married_val = married_map[married]
education_val = education_map[education]
self_employed_val = self_employed_map[self_employed]
property_area_val = property_area_map[property_area]

# When user clicks the 'Predict' button
if st.button('Predict'):
    # Create input data array with numerical values
    input_data = np.array([[gender_val, married_val,
                        int(dependents), 
                        education_val, 
                        self_employed_val, 
                        float(applicant_income), 
                        float(coapplicant_income), 
                        float(loan_amount), 
                        float(loan_amount_term), 
                        float(credit_history),
                        property_area_val]])

    
    st.write("Input Data Shape:", input_data.shape)
    st.write("Input Data:", input_data)

    # Predict loan eligibility
    eligibility = svc_model.predict(input_data)
    
    # Display prediction result
    st.subheader(f'Eligibility: {eligibility[0]}')

    
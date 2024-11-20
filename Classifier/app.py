import pickle
import streamlit as st
import pandas as pd

# Load the trained model
with open('LoanClassifier.pkl', 'rb') as file:
    model = pickle.load(file)

# Sidebar for user input
st.sidebar.header('Input Features')

# Collect user input for features
Gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])
Married = st.sidebar.selectbox('Married', ['No', 'Yes'])
Dependents = st.sidebar.selectbox('Dependents', ['0', '1', '2', '3+'])
Education = st.sidebar.selectbox('Education', ['Graduate', 'Not Graduate'])
Self_Employed = st.sidebar.selectbox('Self Employed', ['No', 'Yes'])
ApplicantIncome = st.sidebar.number_input('Applicant Income', min_value=0)
CoapplicantIncome = st.sidebar.number_input('Coapplicant Income', min_value=0)
LoanAmount = st.sidebar.number_input('Loan Amount', min_value=0)
Loan_Amount_Term = st.sidebar.selectbox('Loan Amount Term', [12, 36, 60, 84, 120, 240, 360])
Credit_History = st.sidebar.selectbox('Credit History', [0.0, 1.0])
Property_Area = st.sidebar.selectbox('Property Area', ['Urban', 'Rural', 'Semiurban'])

# Prepare input data
input_data = pd.DataFrame({
    'Gender': [Gender],
    'Married': [Married],
    'Dependents': [Dependents],
    'Education': [Education],
    'Self_Employed': [Self_Employed],
    'ApplicantIncome': [ApplicantIncome],
    'CoapplicantIncome': [CoapplicantIncome],
    'LoanAmount': [LoanAmount],
    'Loan_Amount_Term': [Loan_Amount_Term],
    'Credit_History': [Credit_History],
    'Property_Area': [Property_Area]
})

# One-Hot Encoding for categorical columns
categorical_columns = [
    'Gender', 'Married', 'Dependents', 'Education', 
    'Loan_Amount_Term', 'Self_Employed', 'Property_Area', 'Credit_History'
]
input_data_encoded = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True)

# Align features with the model's training set
required_columns = [
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount', 
    'Gender_Male', 'Married_Yes', 'Dependents_1', 'Dependents_2', 'Dependents_3+', 
    'Education_Not Graduate', 'Self_Employed_Yes', 'Loan_Amount_Term_36', 
    'Loan_Amount_Term_60', 'Loan_Amount_Term_84', 'Loan_Amount_Term_120', 
    'Loan_Amount_Term_240', 'Loan_Amount_Term_360', 'Credit_History_1.0', 
    'Property_Area_Semiurban', 'Property_Area_Urban'
]

# Ensure all columns exist in input_data_encoded
for col in required_columns:
    if col not in input_data_encoded.columns:
        input_data_encoded[col] = 0

# Reorder columns to match the model's training set
input_data_encoded = input_data_encoded[required_columns]

# Predict and display results
if st.sidebar.button('Predict'):
    prediction = model.predict(input_data_encoded)
    result = 'Approved' if prediction[0] == 1 else 'Rejected'
    st.write(f'Loan Status Prediction: *{result}*')
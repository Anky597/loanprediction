import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Load the trained model
model = joblib.load("best_model.pkl")

# Set up the app title and description
st.title('Loan Approval System with 3D Data Visualization')
st.write('This application predicts loan approval based on user input and visualizes data relationships in 3D.')

# Sidebar for user input
st.sidebar.header("Input Loan Applicant Data")

def get_user_input():
    Gender = st.sidebar.selectbox('Gender', [None, 0, 1], format_func=lambda x: 'Select' if x is None else ('Male' if x == 0 else 'Female'))
    Married = st.sidebar.selectbox('Married', [None, 0, 1], format_func=lambda x: 'Select' if x is None else ('No' if x == 0 else 'Yes'))
    Dependents = st.sidebar.number_input('Dependents', min_value=0, step=1, format='%d')
    Education = st.sidebar.selectbox('Education', [None, 0, 1], format_func=lambda x: 'Select' if x is None else ('Graduate' if x == 0 else 'Not Graduate'))
    Self_Employed = st.sidebar.selectbox('Self Employed', [None, 0, 1], format_func=lambda x: 'Select' if x is None else ('No' if x == 0 else 'Yes'))
    ApplicantIncome = st.sidebar.number_input('Applicant Income', min_value=0, step=1000)
    CoapplicantIncome = st.sidebar.number_input('Coapplicant Income', min_value=0, step=1000)
    LoanAmount = st.sidebar.number_input('Loan Amount', min_value=0, step=10)
    Loan_Amount_Term = st.sidebar.number_input('Loan Amount Term', min_value=12, max_value=360, step=12)
    Credit_History = st.sidebar.selectbox('Credit History', [None, 0, 1], format_func=lambda x: 'Select' if x is None else ('No History' if x == 0 else 'Has History'))
    Property_Area = st.sidebar.selectbox('Property Area', [None, 0, 1, 2], format_func=lambda x: 'Select' if x is None else ('Urban' if x == 0 else ('Semiurban' if x == 1 else 'Rural')))
    
    if None in [Gender, Married, Dependents, Education, Self_Employed, ApplicantIncome, CoapplicantIncome, LoanAmount, Loan_Amount_Term, Credit_History, Property_Area]:
        st.sidebar.warning("Please fill all input fields.")
        return None

    input_data = {
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
    }

    return pd.DataFrame(input_data)

input_df = get_user_input()

if input_df is not None:
    st.subheader('User Input Data')
    st.write(input_df)

    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0]

    if prediction == 1:
        st.success('Loan Approved')
    else:
        st.error('Loan Not Approved')

    st.subheader('Prediction Probabilities')
    st.write(f"Probability of Loan Approval: {prediction_prob[1]:.2f}")
    st.write(f"Probability of Loan Rejection: {prediction_prob[0]:.2f}")

    # Example 3D Scatter Plot using Plotly
    st.subheader('3D Data Visualization')
    st.write('This 3D plot visualizes Applicant Income, Loan Amount, and Coapplicant Income.')

    # Example of random sample data for visualization
    df = pd.DataFrame({
        'ApplicantIncome': [3000, 4000, 5000, 6000, 7000, 8000],
        'LoanAmount': [120, 150, 180, 200, 220, 250],
        'CoapplicantIncome': [0, 1000, 2000, 1500, 0, 3000],
        'Loan_Status': [1, 0, 1, 1, 0, 1]  # Loan approval status (1 for approved, 0 for not approved)
    })

    fig = px.scatter_3d(df, x='ApplicantIncome', y='LoanAmount', z='CoapplicantIncome',
                        color='Loan_Status', title="3D Scatter Plot of Loan Application Data")
    st.plotly_chart(fig)

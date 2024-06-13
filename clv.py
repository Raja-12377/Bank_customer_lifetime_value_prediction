import streamlit as st
import pandas as pd
import numpy as np
import pickle

# Function to calculate CLV based on features
def calculate_clv(age, income, account_balance, frequency_of_mortgage, frequency_of_loan, frequency_of_credit_card):
    base_clv = account_balance * 0.1
    age_factor = (80 - age) / 80  # Assume max age of 80 for normalization
    income_factor = income / 100000  # Normalize income assuming max income of 100000
    frequency_factor = (frequency_of_mortgage + frequency_of_loan + frequency_of_credit_card) / 15  # Normalize frequency per year (max 5 per category)
    return base_clv * age_factor * income_factor * (1 + frequency_factor)

# Load the saved Decision Tree model
def load_model(filename):
    loaded_model = pickle.load(open(filename, 'rb'))
    return loaded_model

# Function to predict CLV
def predict_clv(model, age, gender, income, account_type, account_balance, Transaction_Amount,frequency_of_mortgage, frequency_of_loan, frequency_of_credit_card):
    # Encode categorical variables
    gender_encoded = 1 if gender == 'Male' else 0
    account_type_encoded = 0 if account_type == 'Savings' else (1 if account_type == 'Checking' else 2)

    # Create input array for prediction
    input_data = np.array([[age, gender_encoded, income, account_type_encoded, account_balance, Transaction_Amount, frequency_of_mortgage, frequency_of_loan, frequency_of_credit_card]])

    # Predict CLV
    clv_prediction = model.predict(input_data)[0]

    return clv_prediction

# Main function to run the Streamlit web app
def main():
    # Title and description
    st.title('Bank Customer Lifetime Value Prediction')
    st.write('Enter customer details to predict CLV')

    # Sidebar with user input fields
    st.sidebar.header('User Input Features')

    # Age input
    age = st.sidebar.slider('Age', 18, 80, 30)

    # Gender input
    gender = st.sidebar.selectbox('Gender', ['Male', 'Female'])

    # Income input
    income = st.sidebar.number_input('Income', min_value=20000, max_value=100000, value=50000, step=1000)

    # Account type input
    account_type = st.sidebar.selectbox('Account Type', ['Savings', 'Checking', 'Credit'])

    # Account balance input
    account_balance = st.sidebar.number_input('Account Balance', min_value=1000, max_value=100000, value=5000, step=100)
    #Transaction Amount
    Transaction_Amount = st.sidebar.number_input('Transaction Amount',min_value=0,max_value=100000, value=5000, step=100)

    # Frequency of Mortgage input
    frequency_of_mortgage = st.sidebar.slider('Frequency of Mortgage', 0, 5, 0)

    # Frequency of Loan input
    frequency_of_loan = st.sidebar.slider('Frequency of Loan', 0, 5, 0)

    # Frequency of Credit Card input
    frequency_of_credit_card = st.sidebar.slider('Frequency of Credit Card', 0, 5, 0)

    # Predict CLV on user input
    if st.sidebar.button('Predict CLV'):
        # Load the model
        model = load_model('clv_dt_model.pkl')

        # Predict CLV
        clv_prediction = predict_clv(model, age, gender, income, account_type, account_balance, Transaction_Amount, frequency_of_mortgage, frequency_of_loan, frequency_of_credit_card)

        # Show prediction result
        st.success(f'Predicted CLV: ${clv_prediction:.2f}')

# Run the main function
if __name__ == '__main__':
    main()

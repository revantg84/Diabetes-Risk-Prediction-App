import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = joblib.load('diabetes_model.pkl')

# Set the title of the app
st.title("Diabetes Risk Prediction App")

# Create input fields for user to enter data
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=0)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure (mmHg)", min_value=0, max_value=200, value=80)
bmi = st.number_input("BMI", min_value=0, max_value=50, value=25)
insulin = st.number_input("Insulin Level", min_value=0, max_value=1000, value=50)
age = st.number_input("Age", min_value=1, max_value=120, value=30)
diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=2.5, value=0.5)

# Button to trigger prediction
if st.button('Predict'):
    # Prepare input data for prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, bmi, insulin, age, diabetes_pedigree_function]])

    # Make prediction
    prediction = model.predict(input_data)
    
    # Show prediction result
    if prediction == 1:
        st.subheader("You have a high risk of diabetes.")
    else:
        st.subheader("You have a low risk of diabetes.")

    # Additional information after prediction
    st.write("### More Information:")
    st.write("Based on the input data, the model has predicted whether you are at risk of diabetes.")
    st.write("Here are some suggestions to reduce your risk of diabetes:")
    st.write("- Maintain a healthy diet.")
    st.write("- Engage in regular physical activity.")
    st.write("- Monitor your blood sugar levels.")
    st.write("- Consult with a healthcare professional for personalized advice.")

    # Motivational Quotes
    st.write("### Motivational Quotes:")
    st.write("""
    "The greatest wealth is health." – Virgil  
    "It is health that is real wealth and not pieces of gold and silver." – Mahatma Gandhi  
    "A healthy outside starts from the inside." – Robert Urich  
    """)

    # External Resources
    st.write("### Helpful Resources:")
    st.write("""
    - [American Diabetes Association](https://www.diabetes.org/)
    - [World Health Organization: Diabetes](https://www.who.int/news-room/fact-sheets/detail/diabetes)
    - [Nutrition and Diet Tips for Diabetics](https://www.healthline.com/nutrition/diabetes-diet-plan)
    """)

   

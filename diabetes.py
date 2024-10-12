import streamlit as st
import pickle
import numpy as np

# Load the saved model
pickle_file = 'model_log.pkl'
with open(pickle_file, 'rb') as file: 
    model = pickle.load(file)
    

# Streamlit app
st.title('Diabetes Prediction')

st.write("""
### Discover Your Diabetes Risk: A Step Towards Better Health
Take a proactive step in your health journey! By entering a few key details, you can gain valuable insights into your risk for diabetes. Knowledge is power—let’s empower you to make informed decisions for a healthier tomorrow.
""")


# Input fields for user data
pregnancies = st.number_input('Pregnancies', min_value=0, max_value=20, value=0)
glucose = st.number_input('Glucose', min_value=0.0, max_value=200.0, value=100.0)
blood_pressure = st.number_input('Blood Pressure', min_value=0.0, max_value=150.0, value=70.0)
skin_thickness = st.number_input('Skin Thickness', min_value=0.0, max_value=100.0, value=20.0)
insulin = st.number_input('Insulin', min_value=0.0, max_value=900.0, value=80.0)
bmi = st.number_input('BMI', min_value=0.0, max_value=70.0, value=25.0)
dpf = st.number_input('Diabetes Pedigree Function', min_value=0.0, max_value=2.5, value=0.5)
age = st.number_input('Age', min_value=0, max_value=120, value=30)

# Button to make prediction
if st.button('Predict'):
    # Create input array for prediction
    input_data = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]])
    
    # Make prediction
    prediction = model.predict(input_data)
    
    # Display result
    if prediction[0] == 1:
        st.write("The patient is Diabetic")
    else:
        st.write("The patient is Not Diabetic")

         # Additional insights
    st.write("This prediction is based on a machine learning model that analyzes your input data. However, it is important to consult with a healthcare professional for a detailed diagnosis.")

# Footer
st.write("""
---
**Disclaimer:** This tool is for educational purposes only and should not be used as a substitute for professional medical advice.
""")

from controller.LoadModel import LoadModel
from controller.GetPrediction import GetPredection
from sklearn.preprocessing import QuantileTransformer
import streamlit as st
import numpy as np


model = LoadModel("E:\MastringPython_ElZero\Hesham asem matrial\Projects\Diabetes Prediction\Diabetes-Prediction\Diabetes app\model\diabets_model.pkl")

# Function to clear the output
def clear_output():
    st.session_state.output = ""

# Initialize session state
if 'output' not in st.session_state:
    st.session_state.output = ""

st.title('Diabetes Prediction App')
st.write("We need your information to assess your risk. Please provide the necessary data in the form below.")
Pregnancies = st.number_input('Pregnancies: Number of pregnancies (count).', min_value=0.0, max_value=17.0, step=1.0)
Glucose = st.number_input('Glucose: Blood glucose level (mg/dL).', min_value=0.0, max_value=199.0, step=1.0)
Insulin = st.number_input('Insulin: Blood insulin level (mU/L).', min_value=0.0, max_value=846.0, step=1.0)
BMI = st.number_input('BMI: Body Mass Index (kg/m²).', min_value=0.0, max_value=67.0, step=1.0)
Age = st.number_input('Age: Age (years).', min_value=0.0, max_value=100.0, step=1.0)
SkinThickness = st.number_input('SkinThickness: Skin thickness (mm).', min_value=0.0, max_value=99.0, step=1.0)


user_data = [Pregnancies, Glucose,SkinThickness, Insulin, BMI, Age]

quantile = QuantileTransformer()
data_transformed = quantile.fit_transform([user_data]) 

if st.button('Predict'):
    
    prediction = GetPredection(model, data_transformed[0]) 
    st.write('Prediction:', prediction)


# Button to clear output
if st.button("Clear Output"):
    clear_output()
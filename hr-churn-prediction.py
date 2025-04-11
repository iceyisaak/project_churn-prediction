# Streamlit Documentation: https://docs.streamlit.io/


import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os 

# Title/Text
st.set_page_config(
    page_title="HR Churn Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load data function
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("HR_Dataset.csv")
        return df
    except:
        # Generate sample data if file doesn't exist
        data = {
            'TV': np.random.uniform(10, 300, 200),
            'radio': np.random.uniform(1, 50, 200),
            'newspaper': np.random.uniform(0, 120, 200),
            'sales': np.random.uniform(1, 30, 200)
        }
        df = pd.DataFrame(data)
        return df


# Load model function with proper caching and error handling
# @st.cache_resource

# def load_model():
#     model_path = "best_rf_model.pkl"
#     try:
#         if not os.path.exists(model_path):
#             st.error(f"Model file not found: {model_path}")
#             return None
        
#         with open(model_path, "rb") as f:
#             model = pickle.load(f)
#         return model
#     except Exception as e:
#         st.error(f"Error loading model: {str(e)}")
#         return None

model = None

# Load Model
with open('best_rf_model.pkl', 'rb') as f:
    model = pickle.load(f)

df = load_data()
# model = load_model()

print("Type of loaded model:", type(model))

##########################





##########################

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #4267B2;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #1E3A8A;
    }
    .description {
        font-size: 1rem;
        color: #4B5563;
    }
    .highlight {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-header">💼 ML HR Churn Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="description">Employee Churn Prediction</p>', unsafe_allow_html=True)

st.subheader("Enter the Input for Churn Prediction")

col1, col2 = st.columns(2)

with col1: 
    satisfaction_level = st.slider(
        "Satisfaction Level", 
        min_value=float(df['satisfaction_level'].min()), 
        max_value=float(df['satisfaction_level'].max()),
        value=float(df['satisfaction_level'].median()), 
        step=0.01
    )
    last_evaluation = st.slider(
            "Lastest Evaluation by Employer", 
            min_value=float(df['last_evaluation'].min()), 
            max_value=float(df['last_evaluation'].max()), 
            value=float(df['last_evaluation'].median()), 
            step=0.01
        )
    number_project = st.slider(
            "Number of Projects", 
            min_value=int(df['number_project'].min()), 
            max_value=int(df['number_project'].max()), 
            value=int(df['number_project'].median()), 
            step=1
        )
   

with col2:
    time_spend_company = st.slider(
        "Years Spend at Company", 
        min_value=int(df['time_spend_company'].min()), 
        max_value=int(df['time_spend_company'].max()), 
        value=3, 
        step=1
    )
    average_montly_hours = st.slider(
        "Average Monthly Hours", 
        min_value=int(df['average_montly_hours'].min()), 
        max_value=int(df['average_montly_hours'].max()), 
        value=int(df['average_montly_hours'].median()), 
        step=1
    )



input_data = pd.DataFrame({
            'satisfaction_level': [satisfaction_level],
            'last_evaluation': [last_evaluation],
            'number_project': [number_project],
            'average_montly_hours':[average_montly_hours],
            'time_spend_company':[time_spend_company],
        })



# Prediction with user inputs
predict = st.button("Predict")
result = model.predict(input_data)
if predict :
    st.success(f"Churn Prediction: {result[0]}")
    
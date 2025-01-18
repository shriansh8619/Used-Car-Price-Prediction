import streamlit as st
import pandas as pd
import pickle
# Set page configuration
st.set_page_config(
    page_title="Used Car Price Prediction",  # This changes the browser tab name
    page_icon="🚗"  # You can use emojis or a link to an icon
)
# Load the trained model
model = pickle.load(open("build.pkl", 'rb'))

# Load the target encoding map for the 'Model' column
target_encoding_map = pickle.load(open("model_target_mean.pkl", 'rb'))

# Load the label encoders for categorical columns
label_encoders = pickle.load(open("label_encoders.pkl", 'rb'))

# App title
st.title("Used Car Price Prediction")
st.write("Predict the resale value of used cars based on their features.")

# Sidebar for user inputs
st.sidebar.header("Enter Car Details")

# Input for numerical columns
age = st.sidebar.number_input("Car Age (in years)", min_value=0, max_value=30, step=1, value=1)
kilometer = st.sidebar.number_input("Kilometers Driven", min_value=0, step=1000, value=50000)
quality_score = st.sidebar.number_input("Quality Score", min_value=0.0, max_value=10.0, step=0.1, value=7.0)

# Input for car model with target encoding
model_name = st.sidebar.selectbox("Car Model", list(target_encoding_map.keys()))
model_encoded = target_encoding_map.get(model_name, 0)  # Map the model name using target encoding

# Collecting inputs for categorical columns
categorical_columns = ['Company', 'FuelType', 'Colour', 'BodyStyle', 'Owner', 'DealerState']
categorical_inputs = {}
for col in categorical_columns:
    options = label_encoders[col].classes_  # Get unique classes for the column
    categorical_inputs[col] = st.sidebar.selectbox(f"Select {col}", options)

# Transform the categorical inputs using label encoders
encoded_inputs = {}
for col, value in categorical_inputs.items():
    encoded_inputs[col] = label_encoders[col].transform([value])[0]

# Create input DataFrame for prediction in the specified column order
input_data = {
    'Company': [encoded_inputs['Company']],
    'FuelType': [encoded_inputs['FuelType']],
    'Colour': [encoded_inputs['Colour']],
    'Kilometer': [kilometer],
    'BodyStyle': [encoded_inputs['BodyStyle']],
    'Age': [age],
    'Owner': [encoded_inputs['Owner']],
    'DealerState': [encoded_inputs['DealerState']],
    'QualityScore': [quality_score],
    'Model_encoded': [model_encoded]
}

input_df = pd.DataFrame(input_data)

# Predict and display the result
if st.sidebar.button("Predict Price"):
    prediction = model.predict(input_df)[0]
    st.success(f"The predicted resale price is ₹{prediction:,.2f}")

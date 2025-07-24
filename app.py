import streamlit as st
import pickle
import numpy as np

# Load the model
with open('ecom_model.pkl', 'rb') as f:
    model = pickle.load(f)

st.set_page_config(page_title="CreCat - Product Rating Predictor", layout="centered")

# App title
st.title("🛒 CreCat: E-Commerce Product Rating Predictor")
st.markdown("Predict product ratings based on e-commerce features.")

# Input features (adjust according to your model's expected features)
# Example input fields - change based on your model
feature_1 = st.number_input("Feature 1 (e.g., Price)", min_value=0.0)
feature_2 = st.slider("Feature 2 (e.g., Number of Reviews)", 0, 1000, 100)
feature_3 = st.selectbox("Feature 3 (e.g., Category)", ['Electronics', 'Clothing', 'Home'])

# Encode categorical values if needed
category_map = {'Electronics': 0, 'Clothing': 1, 'Home': 2}
encoded_feature_3 = category_map[feature_3]

# Predict button
if st.button("Predict Rating"):
    input_data = np.array([[feature_1, feature_2, encoded_feature_3]])
    prediction = model.predict(input_data)
    st.success(f"🌟 Predicted Product Rating: {prediction[0]:.2f}")
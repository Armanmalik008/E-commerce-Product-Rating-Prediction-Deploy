import streamlit as st
import pickle
import numpy as np
import os

# Set page config
st.set_page_config(page_title="CreCat - Product Rating Predictor", layout="centered")

# App title
st.title("🛒 CreCat: E-Commerce Product Rating Predictor")
st.markdown("Predict product ratings based on e-commerce features.")

# === Load the Model ===
model_path = 'E-commerce Product Rating.pkl'

if not os.path.exists(model_path):
    st.error(f"❌ Model file not found: `{model_path}`. Please upload the correct `.pkl` file.")
    st.stop()

try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    # Check if model has `predict` method
    if not hasattr(model, 'predict'):
        st.error("❌ Loaded object is not a model. Make sure the `.pkl` file contains a trained model.")
        st.stop()

except Exception as e:
    st.error(f"❌ Failed to load model: {e}")
    st.stop()

# === Input Features (Adjust these to match your trained model) ===
feature_1 = st.number_input("Feature 1 (e.g., Price)", min_value=0.0)
feature_2 = st.slider("Feature 2 (e.g., Number of Reviews)", 0, 1000, 100)
feature_3 = st.selectbox("Feature 3 (e.g., Category)", ['Electronics', 'Clothing', 'Home'])

# Encode categorical value
category_map = {'Electronics': 0, 'Clothing': 1, 'Home': 2}
encoded_feature_3 = category_map[feature_3]

# === Predict Rating ===
if st.button("Predict Rating"):
    input_data = np.array([[feature_1, feature_2, encoded_feature_3]])

    try:
        prediction = model.predict(input_data)
        st.success(f"🌟 Predicted Product Rating: **{prediction[0]:.2f}**")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

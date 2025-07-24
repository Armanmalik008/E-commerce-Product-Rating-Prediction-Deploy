import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="CreCat - Product Rating Predictor", layout="centered")
st.title("🛒 CreCat: E-Commerce Product Rating Predictor")
st.markdown("Predict product ratings based on e-commerce features.")

# Load the model
try:
    with open('E-commerce Product Rating.pkl', 'rb') as f:
        model = pickle.load(f)

    if not hasattr(model, 'predict'):
        st.error("❌ The loaded object is not a valid ML model.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    st.stop()

# Inputs
feature_1 = st.number_input("Feature 1 (e.g., Price)", min_value=0.0)
feature_2 = st.slider("Feature 2 (e.g., Number of Reviews)", 0, 1000, 100)
feature_3 = st.selectbox("Feature 3 (e.g., Category)", ['Electronics', 'Clothing', 'Home'])

category_map = {'Electronics': 0, 'Clothing': 1, 'Home': 2}
encoded_feature_3 = category_map[feature_3]

# Predict
if st.button("Predict Rating"):
    try:
        input_data = np.array([[feature_1, feature_2, encoded_feature_3]])
        st.write("🔍 Input Data:", input_data)
        prediction = model.predict(input_data)
        st.success(f"🌟 Predicted Product Rating: {prediction[0]:.2f}")
    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# ========================
# Config
# ========================
st.set_page_config(page_title="Crop Yield Prediction", layout="wide")

MODEL_PATH = 'crop_yield_model.pkl'
ENCODERS_PATH = 'crop_yield_encoders.pkl'

# ========================
# Data Loading & Encoding
# ========================
@st.cache_data
def load_and_prepare_data():
    """Loads, encodes, and returns the dataset."""
    try:
        data = pd.read_csv("crop_yeild_dataset.csv")
    except FileNotFoundError:
        st.error("❌ Dataset 'crop_yeild_dataset.csv' not found. Please upload the file.")
        return None, None, None, None

    # Encode categorical columns
    le_crop = LabelEncoder()
    le_region = LabelEncoder()

    data['crop_encoded'] = le_crop.fit_transform(data['crop'])
    data['region_encoded'] = le_region.fit_transform(data['region'])

    # Features and target
    X = data[['crop_encoded', 'region_encoded', 'N', 'P', 'K',
              'temperature', 'humidity', 'rainfall', 'ph', 'area_ha']]
    y = data['production_t']

    return X, y, le_crop, le_region


# ========================
# Model Training / Loading
# ========================
@st.cache_resource
def train_and_save_model(X, y):
    """Trains or loads a model and returns it."""
    if os.path.exists(MODEL_PATH):
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
    else:
        st.info("🔄 Training new model...")
        model = RandomForestRegressor(n_estimators=50, max_depth=20, random_state=42)
        model.fit(X, y)
        with open(MODEL_PATH, 'wb') as f:
            pickle.dump(model, f)
    return model


# ========================
# Main App
# ========================
st.title("🌾 AI-powered Crop Yield Prediction")
st.markdown("Enter the crop details and environmental conditions to predict the yield.")

X, y, le_crop, le_region = load_and_prepare_data()

if X is not None:
    model = train_and_save_model(X, y)

    # Sidebar inputs
    with st.sidebar:
        st.header("📊 Input Parameters")
        crop = st.selectbox("Select Crop", le_crop.classes_)
        region = st.selectbox("Select Region", le_region.classes_)

        st.markdown("---")
        st.subheader("🌱 Soil & Climate Data")
        N = st.number_input("Nitrogen (N) - kg/ha", min_value=0.0, max_value=300.0, value=90.0)
        P = st.number_input("Phosphorus (P) - kg/ha", min_value=0.0, max_value=200.0, value=40.0)
        K = st.number_input("Potassium (K) - kg/ha", min_value=0.0, max_value=300.0, value=50.0)
        temperature = st.number_input("Temperature (°C)", min_value=0.0, max_value=50.0, value=28.0)
        humidity = st.number_input("Humidity (%)", min_value=0.0, max_value=100.0, value=65.0)
        rainfall = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=200.0)
        ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.8)
        area_ha = st.number_input("Area (acres)", min_value=1, max_value=10000, value=500)

    # Prediction
    st.header("🔮 Prediction Result")
    if st.button("Predict Yield"):
        try:
            crop_enc = le_crop.transform([crop])[0]
            region_enc = le_region.transform([region])[0]

            features = [[crop_enc, region_enc, N, P, K, temperature, humidity, rainfall, ph, area_ha]]
            prediction = model.predict(features)[0]

            st.success(f"🌱 Predicted Yield: **{prediction:.2f} Quintal**")
            st.info(f"This prediction is for a **{area_ha:.0f}** acre plot of **{crop}** in the **{region}** region.")
        except Exception as e:
            st.error(f"⚠️ An error occurred during prediction: {e}")

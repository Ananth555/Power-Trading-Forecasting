import streamlit as st
import numpy as np
import pandas as pd
import pickle

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="⚡ Power Price Forecasting", layout="centered")

# =========================
# LOAD MODEL
# =========================
@st.cache_resource
def load_model():
    with open("model.pkl", "rb") as f:
        data = pickle.load(f)
    return data["model"], data["features"]

model, features = load_model()

# =========================
# TITLE
# =========================
st.title("⚡ Power Trading Price Forecasting")
st.markdown("Predict **MCP (₹/MWh)** using market, weather, and historical trends")

st.info(f"Model Loaded | Features Used: {len(features)}")

st.markdown("---")

# =========================
# INPUT SECTIONS
# =========================

# MARKET INPUTS
st.subheader("📊 Market Inputs")
col1, col2 = st.columns(2)

with col1:
    hour = st.number_input("Hour (0–23)", 0, 23, 10)
    purchase = st.number_input("Purchase Bid (MW)", value=8000.0)
    sell = st.number_input("Sell Bid (MW)", value=12000.0)

with col2:
    mcv = st.number_input("MCV (MW)", value=6000.0)
    scheduled = st.number_input("Final Scheduled Volume (MW)", value=6000.0)

st.markdown("---")

# WEATHER INPUTS
st.subheader("🌦 Weather (Mundra)")
col3, col4 = st.columns(2)

with col3:
    temp = st.number_input("Temperature (°C)", value=30.0)
    humidity = st.number_input("Humidity (%)", value=60.0)

with col4:
    cloud = st.number_input("Cloud Cover (%)", value=20.0)
    wind = st.number_input("Wind Speed (km/h)", value=10.0)

st.markdown("---")

# LAG FEATURES
st.subheader("⏱ Historical MCP (Lag Features)")
col5, col6 = st.columns(2)

with col5:
    lag_1 = st.number_input("Lag 1 (15 min ago)", value=2800.0)
    lag_2 = st.number_input("Lag 2", value=2790.0)
    lag_4 = st.number_input("Lag 4 (1 hour ago)", value=2750.0)

with col6:
    lag_8 = st.number_input("Lag 8", value=2700.0)
    lag_96 = st.number_input("Lag 96 (Previous day)", value=2900.0)

rolling_mean = st.number_input("Rolling Mean (4 blocks)", value=2800.0)
rolling_std = st.number_input("Rolling Std (4 blocks)", value=50.0)

st.markdown("---")

# =========================
# PREDICTION
# =========================
if st.button("🔮 Predict MCP"):

    try:
        input_dict = {
            'Hour': hour,
            'Purchase Bid (MW)': purchase,
            'Sell Bid (MW)': sell,
            'MCV (MW)': mcv,
            'Final Scheduled Volume (MW)': scheduled,
            'Mundra_temperature_2m (°C)': temp,
            'Mundra_relative_humidity_2m (%)': humidity,
            'Mundra_cloud_cover (%)': cloud,
            'Mundra_wind_speed_10m (km/h)': wind,
            'lag_1': lag_1,
            'lag_2': lag_2,
            'lag_4': lag_4,
            'lag_8': lag_8,
            'lag_96': lag_96,
            'rolling_mean_4': rolling_mean,
            'rolling_std_4': rolling_std
        }

        # Ensure correct order
        input_df = pd.DataFrame([input_dict])[features]

        prediction = model.predict(input_df)

        st.success(f"💰 Predicted MCP: ₹ {prediction[0]:.2f} per MWh")
        st.caption("⚡ Estimated electricity market clearing price based on inputs")

    except Exception as e:
        st.error(f"❌ Error: {e}")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.caption("Built with ❤️ using Machine Learning & Streamlit")
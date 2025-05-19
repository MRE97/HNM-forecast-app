import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

st.write("📂 Current working directory:", os.getcwd())
st.write("📄 Files in this folder:", os.listdir("."))

st.set_page_config(page_title="Outlet Forecast App", layout="centered")

# === Load Data ===
@st.cache_data
def load_data(brand_name):
    file_path = f"brand_data/{brand_name}.csv"
    df = pd.read_csv(file_path)

    def to_number(x):
        if isinstance(x, str):
            return float(x.replace(',', '').replace('%', '').replace('Rp', '').strip())
        return x

    numeric_cols = [
        'Density (Pop/Area)', 'Population', 'Area (Km²)',
        'Revenue (Monthly)', 'Full Capacity (Max)', 'Rent/Year',
        'Width (m)', 'Building Area', 'Ceiling Avg',
        'Car Park', 'Bike Park', 'Total Park', 'Capex (IDR)', 'ROI'
    ]
    for col in numeric_cols:
        df[col] = df[col].apply(to_number)

    df['ROI'] = df['ROI'] / 100
    df['Operating_Income'] = (df['Revenue (Monthly)'] * 12) - df['Rent/Year']

    encoders = {}
    for col in ['City', 'Kecamatan', 'Near Residential?']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

    features = [
        'Density (Pop/Area)', 'Population', 'Area (Km²)', 'Full Capacity (Max)',
        'Rent/Year', 'Near Residential?', 'Width (m)', 'Building Area',
        'Ceiling Avg', 'Car Park', 'Bike Park', 'Total Park',
        'City', 'Kecamatan'
    ]

    X = df[features]
    y_revenue = df['Revenue (Monthly)']
    y_capex = df['Capex (IDR)']

    model_revenue = RandomForestRegressor().fit(X, y_revenue)
    model_capex = RandomForestRegressor().fit(X, y_capex)

    return model_revenue, model_capex, encoders

# === Brand Selector ===
st.title("📈 Outlet Forecast App")
brand_data_path = os.path.join(os.path.dirname(__file__), "brand_data")
brand_list = [f.replace(".csv", "") for f in os.listdir(brand_data_path) if f.endswith(".csv")]
brand = st.selectbox("Select Brand", brand_list)

model_revenue, model_capex, le = load_data(brand)

# === Input Form ===
st.subheader("🔢 Enter Outlet Details")

density = st.number_input("Population Density", value=10000)
population = st.number_input("Population", value=150000)
area = st.number_input("Area (Km²)", value=15.0)
full_capacity = st.number_input("Full Capacity", value=500)
rent = st.number_input("Annual Rent (Rp)", value=600_000_000)
near_res = st.selectbox("Near Residential?", le['Near Residential?'].classes_)
width = st.number_input("Width (m)", value=20.0)
building_area = st.number_input("Building Area (sqm)", value=800)
ceiling_avg = st.number_input("Avg Ceiling Height", value=3.0)
car_park = st.number_input("Car Parking", value=6)
bike_park = st.number_input("Motorbike Parking", value=12)
total_park = car_park + bike_park
city = st.selectbox("City", le['City'].classes_)
kecamatan = st.selectbox("Kecamatan", le['Kecamatan'].classes_)

if st.button("📊 Forecast"):
    X_new = pd.DataFrame([{
        'Density (Pop/Area)': density,
        'Population': population,
        'Area (Km²)': area,
        'Full Capacity (Max)': full_capacity,
        'Rent/Year': rent,
        'Near Residential?': le['Near Residential?'].transform([near_res])[0],
        'Width (m)': width,
        'Building Area': building_area,
        'Ceiling Avg': ceiling_avg,
        'Car Park': car_park,
        'Bike Park': bike_park,
        'Total Park': total_park,
        'City': le['City'].transform([city])[0],
        'Kecamatan': le['Kecamatan'].transform([kecamatan])[0]
    }])

    revenue = model_revenue.predict(X_new)[0]
    capex = model_capex.predict(X_new)[0]
    operating_income = (revenue * 12) - rent
    roi = operating_income / capex

    st.success(f"💵 Predicted Monthly Revenue: Rp {revenue:,.0f}")
    st.success(f"🏗️ Predicted Capex: Rp {capex:,.0f}")
    st.success(f"📈 Predicted Operating Income: Rp {operating_income:,.0f}")
    st.success(f"📊 Predicted ROI: {roi * 100:.2f}%")

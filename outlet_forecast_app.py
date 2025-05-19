import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import os

# Helper: Convert text to numbers
def to_number(x):
    if isinstance(x, str):
        clean_x = x.replace('.', '').replace(',', '').replace('%', '').replace('Rp', '').strip()
        return float(clean_x)
    return x

# Define brand name
brand_name = "brand_a"  # Change this to brand_b, brand_c later

# Load dataset
file_path = f"brand_data/{brand_name}.csv"
df = pd.read_csv(r"C:\Users\User10\Documents\HWG\Python Projects\HNM_forecasting_app\brand_data\Brand_HNM.csv", sep=";")
df = df.drop(columns=['Car Park', 'Bike Park', 'Total Park'], errors='ignore')





# Clean numeric columns
numeric_cols = [
    'Density (Pop/Area)', 'Population', 'Area (Km²)', 'Revenue (Monthly)',
    'Full Capacity (Max)', 'Rent/Year', 'Width (m)', 'Building Area',
    'Ceiling Avg', 'Capex (IDR)', 'ROI'
]


for col in numeric_cols:
    df[col] = df[col].apply(to_number)

df['ROI'] = df['ROI'] / 100  # convert % to decimal
df['Operating_Income'] = (df['Revenue (Monthly)'] * 12) - df['Rent/Year']

# Label encode categorical columns
label_encoders = {}
for col in ['City', 'Kecamatan', 'Near Residential?']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le

# Define input features (no Revenue as input!)
features = [
    'Density (Pop/Area)', 'Population', 'Area (Km²)', 'Full Capacity (Max)',
    'Rent/Year', 'Near Residential?', 'Width (m)', 'Building Area',
    'Ceiling Avg', 'City', 'Kecamatan'
]


X = df[features]
y_revenue = df['Revenue (Monthly)']
y_capex = df['Capex (IDR)']

# Train models
model_revenue = RandomForestRegressor(random_state=42).fit(X, y_revenue)
model_capex = RandomForestRegressor(random_state=42).fit(X, y_capex)

# === Example Input (change values as needed) ===
new_outlet = pd.DataFrame([{
    'Density (Pop/Area)': 12000,
    'Population': 79000,
    'Area (Km²)': 6.54,
    'Full Capacity (Max)': 600,
    'Rent/Year': 600000000,
    'Near Residential?': label_encoders['Near Residential?'].transform(['Ya'])[0],
    'Width (m)': 26,
    'Building Area': 717,
    'Ceiling Avg': 3.2,
    'City': label_encoders['City'].transform(['Semarang'])[0],
    'Kecamatan': label_encoders['Kecamatan'].transform(['Candisari'])[0]
}])

# === Predict ===
predicted_revenue = model_revenue.predict(new_outlet)[0]
predicted_capex = model_capex.predict(new_outlet)[0]
predicted_income = (predicted_revenue * 12) - new_outlet['Rent/Year'].values[0]
predicted_roi = predicted_income / predicted_capex

# === Output ===
print("📊 Forecast Results:")
print(f"Monthly Revenue : Rp {predicted_revenue:,.0f}")
print(f"Capex           : Rp {predicted_capex:,.0f}")
print(f"Operating Income: Rp {predicted_income:,.0f}")
print(f"ROI             : {predicted_roi * 100:.2f}%")

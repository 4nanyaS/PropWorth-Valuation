from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# TRAIN MODEL 

df = pd.read_csv("DubaiPropertyForSale.csv")

df['Beds'] = (
    df['Beds']
    .replace('Studio', 0)
    .astype(str)
    .str.extract(r'(\d+)')
    .fillna(0)
    .astype(int)
)

df['Baths'] = (
    df['Baths']
    .astype(str)
    .str.extract(r'(\d+)')
    .fillna(0)
    .astype(int)
)

df['Area_sqft'] = (
    df['Area_sqft']
    .astype(str)
    .str.extract(r'(\d+\.?\d*)')
    .fillna(0)
    .astype(float)
)

df['Furnishing'] = df['Furnishing'].map({
    'Unfurnished': 0,
    'Furnished': 1,
}).fillna(0)

df['Price'] = (
    df['Price']
    .astype(str)
    .str.replace(',', '', regex=False)
    .str.extract(r'(\d+\.?\d*)')
    .astype(float)
)

X = df[['Beds', 'Baths', 'Area_sqft', 'Furnishing']]
y = df['Price']

model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

#  FLASK APP

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "PropWorth API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    beds = int(data.get("beds", 0))
    baths = int(data.get("baths", 0))
    area = float(data.get("area_sqft", 0))
    furnishing = int(data.get("furnishing", 0))

    features = np.array([[beds, baths, area, furnishing]])
    prediction = model.predict(features)[0]

    return jsonify({
        "predicted_price": round(float(prediction), 2)
    })

# ❌ DO NOT add app.run() for Render

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np

app = Flask(__name__)
CORS(app)  

# Load trained model
with open("price_model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/")
def home():
    return "PropWorth API is running 🚀"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    beds = int(data.get("beds", 0))
    baths = int(data.get("baths", 0))
    area = float(data.get("area_sqft", 0))
    furnishing = int(data.get("furnishing", 0))  # 0 = unfurnished, 1 = furnished

    features = np.array([[beds, baths, area, furnishing]])
    prediction = model.predict(features)[0]

    return jsonify({
        "predicted_price": round(float(prediction), 2)
    })

if __name__ == "__main__":
    app.run(debug=True)

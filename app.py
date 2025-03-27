from flask import Flask, request, render_template, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load saved model and scaler
model = joblib.load("blood_sugar_model.pkl")
scaler = joblib.load("scaler.pkl")

@app.route("/")
def home():
    return render_template("index.html")  # Loads the HTML form

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Get input values from the form
        data = [float(x) for x in request.form.values()]
        data_scaled = scaler.transform([data])  # Scale input

        # Make prediction
        prediction = model.predict(data_scaled)[0]
        probability = model.predict_proba(data_scaled)[0][1]  # Probability of High Blood Sugar

        result = "High Blood Sugar" if prediction == 1 else "Normal Blood Sugar"

        return render_template("index.html", prediction_text=f"Prediction: {result} (Probability: {probability:.2f})")

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)

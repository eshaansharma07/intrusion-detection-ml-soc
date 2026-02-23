from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import json
from logger import log_event
from alert_system import generate_alert

app = Flask(__name__)
CORS(app)   # 🔥 Allows React (3000) to talk to Flask (5000)

print("Loading Model...")

# Load Model, Scaler & Training Columns
model = joblib.load("../models/ids_model.pkl")
scaler = joblib.load("../models/scaler.pkl")
columns = joblib.load("../models/columns.pkl")

print("Model Loaded Successfully!")

# ==============================
# 🔥 PREDICTION API
# ==============================

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json['features']

        # Convert input to DataFrame
        df = pd.DataFrame([data])

        # Apply same encoding as training
        df = pd.get_dummies(df)

        # Align with training columns
        df = df.reindex(columns=columns, fill_value=0)

        # Scale Features
        features_scaled = scaler.transform(df)

        # Predict
        prediction = model.predict(features_scaled)[0]
        probability = model.predict_proba(features_scaled)[0][1]

        # Risk Score
        risk_score = int(probability * 100)

        # Log Event
        log_event(prediction, risk_score)

        # Generate Alert if High Risk
        if risk_score > 70:
            generate_alert(prediction, risk_score)

        return jsonify({
            "prediction": int(prediction),
            "risk_score": risk_score
        })

    except Exception as e:
        return jsonify({"error": str(e)})

# ==============================
# 🚨 ALERTS API
# ==============================

@app.route('/alerts', methods=['GET'])
def get_alerts():
    try:
        alerts = []
        with open("../alerts/alerts.json", "r") as f:
            for line in f:
                alerts.append(json.loads(line))
        return jsonify(alerts)
    except:
        return jsonify([])

# ==============================
# 📊 METRICS API
# ==============================

@app.route('/metrics', methods=['GET'])
def get_metrics():
    try:
        total = 0
        threats = 0

        with open("../logs/events.json", "r") as f:
            for line in f:
                event = json.loads(line)
                total += 1
                if event["risk_score"] > 70:
                    threats += 1

        safe = total - threats

        detection_rate = (threats / total) * 100 if total > 0 else 0
        false_positive_rate = (safe / total) * 100 if total > 0 else 0

        return jsonify({
            "total_predictions": total,
            "alerts": threats,
            "safe": safe,
            "detection_rate": round(detection_rate, 2),
            "false_positive_rate": round(false_positive_rate, 2)
        })
    except:
        return jsonify({})

# ==============================

if __name__ == '__main__':
    app.run(debug=True, use_reloader=False)
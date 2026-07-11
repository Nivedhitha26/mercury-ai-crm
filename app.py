from flask import Flask, request, jsonify
import joblib

app = Flask(__name__)

case_model = joblib.load("case_classifier_logistic.pkl")
priority_model = joblib.load("priority_text_classifier.pkl")

HIGH_KEYWORDS = [
    "damaged",
    "broken",
    "unsafe",
    "exploded",
    "fire",
    "fraud",
    "missing",
    "never arrived",
    "not received",
    "urgent",
    "stopped working",
    "defective",
    "critical"
]

LOW_KEYWORDS = [
    "track",
    "tracking",
    "where is my order",
    "order status",
    "delivery date",
    "shipment",
    "courier"
]

@app.route("/")
def home():
    return "SmartKart AI API Running Successfully"

@app.route("/predict", methods=["POST"])
def predict():

    data = request.get_json()

    description = data["description"]

    case_type = case_model.predict([description])[0]

    text = description.lower()

    if any(word in text for word in HIGH_KEYWORDS):
        priority = "High"

    elif any(word in text for word in LOW_KEYWORDS):
        priority = "Low"

    else:
        priority = priority_model.predict([description])[0]

    return jsonify({
        "caseType": case_type,
        "priority": priority
    })

if __name__ == "__main__":
    app.run()

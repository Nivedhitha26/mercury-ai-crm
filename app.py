from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

app = Flask(__name__)
CORS(app)  

case_model = joblib.load("case_classifier.pkl")
priority_model = joblib.load("priority_classifier.pkl")


CASE_CONFIDENCE_THRESHOLD = 0.35


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
    data = request.get_json(silent=True)
    if not data:
        return jsonify({
            "success": False,
            "message": "Invalid request. Please send a JSON body with a 'description' field."
        }), 400

    description = data.get("description", "")
    if not isinstance(description, str):
        description = ""
    description = description.strip()

    if len(description) < 3:
        return jsonify({
            "success": False,
            "message": "Please enter the correct support issue details.",
            "block_submission": True
        }), 400

    proba = case_model.predict_proba([description])[0]
    classes = case_model.classes_
    idx = proba.argmax()
    case_type = classes[idx]
    case_confidence = proba[idx]

    if case_type == "Invalid" or case_confidence < CASE_CONFIDENCE_THRESHOLD:
        return jsonify({
            "success": False,
            "message": "Please enter the correct support issue details.",
            "block_submission": True
        }), 400

    text = description.lower()

    if any(word in text for word in HIGH_KEYWORDS):
        priority = "High"
        priority_source = "keyword_override"
    elif any(word in text for word in LOW_KEYWORDS):
        priority = "Low"
        priority_source = "keyword_override"
    else:
        priority = priority_model.predict([description])[0]
        priority_source = "ml_model"

    return jsonify({
        "success": True,
        "caseType": case_type,
        "caseTypeConfidence": round(float(case_confidence) * 100, 2),
        "priority": priority,
        "prioritySource": priority_source
    })


if __name__ == "__main__":
    # debug=False for anything beyond local testing (avoid leaking stack traces)
    app.run(host="0.0.0.0", port=5000, debug=False)

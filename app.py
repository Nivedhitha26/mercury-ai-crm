from flask import Flask
import joblib

app = Flask(__name__)

case_model = joblib.load("case_classifier_logistic.pkl")
priority_model = joblib.load("priority_text_classifier.pkl")

@app.route("/")
def home():
    return "Both Models Loaded Successfully"

if __name__ == "__main__":
    app.run()

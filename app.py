from flask import Flask
import joblib

app = Flask(__name__)

case_model = joblib.load("case_classifier_logistic.pkl")

@app.route("/")
def home():
    return "Case Model Loaded Successfully"

if __name__ == "__main__":
    app.run()

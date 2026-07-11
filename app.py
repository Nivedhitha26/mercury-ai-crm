from flask import Flask
import joblib

app = Flask(__name__)

@app.route("/")
def home():
    return "Joblib Imported Successfully"

if __name__ == "__main__":
    app.run()

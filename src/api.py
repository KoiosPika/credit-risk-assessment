from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import pandas as pd
import os
import boto3

app = Flask(__name__)
CORS(app)

# AWS S3 configure
S3_BUCKET = "credit-risk-assessment"
MODEL_FILE = "model_pipeline.pkl"
LOCAL_MODEL_PATH = "/tmp/model_pipeline.pkl"

# Download model from S3
s3 = boto3.client("s3")

def download_model():
    if not os.path.exists(LOCAL_MODEL_PATH):
        print("Downloading model from s3")
        s3.download_file(S3_BUCKET,MODEL_FILE,LOCAL_MODEL_PATH)
        print("Download complete.")
    return joblib.load(LOCAL_MODEL_PATH)

model = download_model()

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict credit risk from JSON input."""
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        return jsonify({'credit_risk': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

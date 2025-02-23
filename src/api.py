from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS
import pandas as pd
import os
import boto3
import json
from datetime import datetime

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

def save_request_to_s3(request_data, prediction):
    """Saves the request data and prediction to S3 as a JSON file."""
    try:
        timestamp = datetime.utcnow().isoformat()
        file_name = f"predictions/{timestamp}.json"

        log_data = {
            "timestamp": timestamp,
            "input": request_data,
            "prediction": int(prediction)
        }

        # Convert to JSON string
        json_data = json.dumps(log_data)

        # Upload to S3
        s3.put_object(Bucket=S3_BUCKET, Key=file_name, Body=json_data)

        print(f"Saved request data to S3: {file_name}")
    except Exception as e:
        print(f"Error saving request to S3: {e}")

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict credit risk from JSON input."""
    try:
        data = request.get_json()
        df = pd.DataFrame([data])
        prediction = model.predict(df)
        save_request_to_s3(data, prediction[0])
        return jsonify({'credit_risk': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)

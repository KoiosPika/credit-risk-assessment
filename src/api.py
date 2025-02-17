from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the trained model
model = joblib.load("model_pipeline.pkl")

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint to predict credit risk from JSON input."""
    try:
        data = request.get_json()
        df = pd.DataFrame([data])  # Convert JSON to DataFrame
        prediction = model.predict(df)
        return jsonify({'credit_risk': int(prediction[0])})
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True)

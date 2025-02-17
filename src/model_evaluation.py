from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load preprocessed data
X_train, X_test, y_train, y_test = joblib.load("processed_data.pkl")

def evaluate_model():
    """Load trained model and evaluate performance."""
    model = joblib.load("model_pipeline.pkl")
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"📊 Model Accuracy: {accuracy:.2f}")
    print("📄 Classification Report:\n", report)

evaluate_model()

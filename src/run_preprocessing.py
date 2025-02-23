from data_processing import preprocess_data

# Load the dataset
file_path = "../data/balanced_credit_risk_dataset.csv"  # Adjust path if needed

# Preprocess the dataset
X_train, X_test, y_train, y_test = preprocess_data(file_path)

# Save processed data for training
import joblib
joblib.dump((X_train, X_test, y_train, y_test), "processed_data.pkl")
print("Preprocessing completed and data saved!")
exit()
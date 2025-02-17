import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define categorical and numerical features
categorical_cols = ['person_home_ownership', 'loan_intent']
ordinal_cols = ['loan_grade', 'cb_person_default_on_file']
numerical_cols = ['person_emp_length', 'loan_int_rate']

# Custom function to apply Label Encoding for ordinal columns
def label_encode_ordinal(X):
    X = X.copy()
    label_encoders = {
        "loan_grade": LabelEncoder(),
        "cb_person_default_on_file": LabelEncoder()
    }
    
    # Fit encoders
    for col, encoder in label_encoders.items():
        X[col] = encoder.fit_transform(X[col])
    
    return X

def preprocess_data(file_path):
    # Define ColumnTransformer
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('scaler', StandardScaler())
            ]), numerical_cols),
            
            ('cat', Pipeline([
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
            ]), categorical_cols),

            ('ord', FunctionTransformer(label_encode_ordinal, validate=False), ordinal_cols)
        ]
    )

    # Create the full pipeline including model training
    model_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    # Load data
    df = pd.read_csv(file_path)

    # Define X and y
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Fit the pipeline on training data
    model_pipeline.fit(X_train, y_train)

    # Save the pipeline (includes preprocessing and model)
    with open("model_pipeline.pkl", "wb") as f:
        pickle.dump(model_pipeline, f)

    return X_train, X_test, y_train, y_test
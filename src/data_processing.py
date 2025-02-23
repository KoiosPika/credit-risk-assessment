import pandas as pd
import pickle
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define categorical and numerical features
categorical_cols = ['person_home_ownership', 'loan_intent','loan_grade', 'cb_person_default_on_file']
numerical_cols = ['person_emp_length', 'loan_int_rate','loan_amnt','loan_percent_income','cb_person_cred_hist_length']

def preprocess_data(file_path):
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline([
                ('scaler', StandardScaler())
            ]), numerical_cols),
            
            ('cat', Pipeline([
                ('onehot', OneHotEncoder(handle_unknown='ignore', drop='first'))
            ]), categorical_cols),
        ]
    )

    model_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
    ])

    df = pd.read_csv(file_path)

    # Define X and y
    X = df.drop(columns=['loan_status'])
    y = df['loan_status']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_pipeline.fit(X_train, y_train)

    with open("model_pipeline.pkl", "wb") as f:
        pickle.dump(model_pipeline, f)

    return X_train, X_test, y_train, y_test
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def train_and_evaluate():
    print("Loading and processing data...")
    # 1. Load Data
    try:
        df = pd.read_csv("Sleep_health_and_lifestyle_dataset (1).csv")
    except FileNotFoundError:
        print("Error: Dataset file not found.")
        return

    # 2. Preprocessing
    # CRITICAL FIX: Handle Missing Target. NaN in 'Sleep Disorder' implies 'None' (Healthy)
    # The original analysis likely dropped these or treated them as missing, losing ~60% of data.
    df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

    # Drop Person ID as it's an identifier, not a feature
    if "Person ID" in df.columns:
        df = df.drop(columns=["Person ID"])

    # Fix BMI Category inconsistencies ("Normal Weight" -> "Normal")
    df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})

    # Split Blood Pressure into Systolic and Diastolic
    if "Blood Pressure" in df.columns:
        df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
        df = df.drop(columns=['Blood Pressure'])

    # Define Features and Target
    X = df.drop(columns=["Sleep Disorder"])
    y = df["Sleep Disorder"]

    # Encode Target
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    print(f"Target Classes: {le.classes_}")

    # 3. Feature Engineering Pipeline
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    # 4. Model Definition
    # Random Forest: Robust to outliers and non-linear data
    rf_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
    ])

    # XGBoost: High performance gradient boosting
    xgb_model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", XGBClassifier(
            n_estimators=200, 
            learning_rate=0.05, 
            max_depth=5, 
            use_label_encoder=False, 
            eval_metric='mlogloss',
            random_state=42
        ))
    ])

    # Voting Classifier: Combines both models for better stability and accuracy
    voting_clf = VotingClassifier(
        estimators=[('rf', rf_model), ('xgb', xgb_model)],
        voting='soft'
    )

    # 5. Train and Test
    # Stratified Split to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )

    print("\nTraining Ensemble Model (Random Forest + XGBoost)...")
    voting_clf.fit(X_train, y_train)

    print("\nEvaluating on Test Set...")
    y_pred = voting_clf.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prinht(f"\nFINAL ACCURACY: {acc*100:.2f}%")
    print("-" * 30)
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

if __name__ == "__main__":
    train_and_evaluate()

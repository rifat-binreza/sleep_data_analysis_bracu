import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load data
df = pd.read_csv("Sleep_health_and_lifestyle_dataset (1).csv")

# Preprocessing
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")
if "Person ID" in df.columns:
    df.drop(columns=["Person ID"], inplace=True)

df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})

if "Blood Pressure" in df.columns:
    df[["Systolic_BP", "Diastolic_BP"]] = (
        df["Blood Pressure"].astype(str).str.split("/", expand=True).astype(int)
    )
    df.drop(columns=["Blood Pressure"], inplace=True)

X = df.drop(columns=["Sleep Disorder"])
y = df["Sleep Disorder"]

le = LabelEncoder()
y_encoded = le.fit_transform(y)

categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Define base estimators with some initial tuning
rf = RandomForestClassifier(random_state=42)
xgb = XGBClassifier(eval_metric="mlogloss", random_state=42)

# Create a pipeline for GridSearch on RF
rf_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", rf)
])

rf_params = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__max_depth": [None, 10, 20],
    "classifier__min_samples_split": [2, 5],
    "classifier__class_weight": [None, "balanced"]
}

print("Tuning Random Forest...")
rf_grid = GridSearchCV(rf_pipeline, rf_params, cv=5, n_jobs=-1)
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_.named_steps["classifier"]
print(f"Best RF Params: {rf_grid.best_params_}")

# Create a pipeline for GridSearch on XGB
xgb_pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", xgb)
])

xgb_params = {
    "classifier__n_estimators": [100, 200, 300],
    "classifier__learning_rate": [0.01, 0.05, 0.1],
    "classifier__max_depth": [3, 5, 7]
}

print("Tuning XGBoost...")
xgb_grid = GridSearchCV(xgb_pipeline, xgb_params, cv=5, n_jobs=-1)
xgb_grid.fit(X_train, y_train)
best_xgb = xgb_grid.best_estimator_.named_steps["classifier"]
print(f"Best XGB Params: {xgb_grid.best_params_}")

# Stacking with best estimators
stack_clf = StackingClassifier(
    estimators=[("rf", best_rf), ("xgb", best_xgb)],
    final_estimator=LogisticRegression(max_iter=4000),
    stack_method="predict_proba",  
    n_jobs=-1
)

stacking_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", stack_clf)
])

stacking_model.fit(X_train, y_train)

y_pred = stacking_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print(f"STACKING ACCURACY: {acc*100:.2f}%")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:\n", cm)

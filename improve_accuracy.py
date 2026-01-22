import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer

# 1. Load Data
df = pd.read_csv("Sleep_health_and_lifestyle_dataset (1).csv")

# 2. Preprocessing
# Handle Missing Target: NaN in 'Sleep Disorder' means 'None'
df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

# Drop Person ID
if "Person ID" in df.columns:
    df = df.drop(columns=["Person ID"])

# Fix BMI Category inconsistencies
df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})

# Split Blood Pressure
if "Blood Pressure" in df.columns:
    df[['Systolic_BP', 'Diastolic_BP']] = df['Blood Pressure'].str.split('/', expand=True).astype(int)
    df = df.drop(columns=['Blood Pressure'])

# Define Features and Target
X = df.drop(columns=["Sleep Disorder"])
y = df["Sleep Disorder"]

# Encode Target
le = LabelEncoder()
y_encoded = le.fit_transform(y)
print("Classes:", le.classes_)

# 3. Feature Engineering Pipeline
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
    ]
)

# 4. Model Training & Evaluation
# Using Random Forest as a baseline robust model
rf_model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(n_estimators=200, random_state=42))
])

# Using XGBoost
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

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import GridSearchCV

# Train Random Forest
print("\n--- Random Forest ---")
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)
acc_rf = accuracy_score(y_test, y_pred_rf)
print(f"Accuracy: {acc_rf:.4f}")
print(classification_report(y_test, y_pred_rf, target_names=le.classes_))

# Train XGBoost
print("\n--- XGBoost ---")
xgb_model.fit(X_train, y_train)
y_pred_xgb = xgb_model.predict(X_test)
acc_xgb = accuracy_score(y_test, y_pred_xgb)
print(f"Accuracy: {acc_xgb:.4f}")
print(classification_report(y_test, y_pred_xgb, target_names=le.classes_))

# Ensemble (Voting Classifier)
print("\n--- Voting Classifier (Soft) ---")
voting_clf = VotingClassifier(
    estimators=[('rf', rf_model), ('xgb', xgb_model)],
    voting='soft'
)
voting_clf.fit(X_train, y_train)
y_pred_voting = voting_clf.predict(X_test)
acc_voting = accuracy_score(y_test, y_pred_voting)
print(f"Accuracy: {acc_voting:.4f}")
print(classification_report(y_test, y_pred_voting, target_names=le.classes_))

# Cross Validation on Voting Classifier
print("\n--- Cross Validation (Voting Classifier) ---")
cv_scores = cross_val_score(voting_clf, X, y_encoded, cv=5, scoring='accuracy')
print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
print("Individual CV Scores:", cv_scores)

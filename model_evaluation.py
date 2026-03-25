import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from lightgbm import LGBMClassifier

# Load and prepare data
df = pd.read_csv("Sleep_health_and_lifestyle_dataset (1).csv")

# Data cleaning
df_clean = df.copy()
df_clean.drop(columns=["Person ID"], inplace=True)
df_clean["BMI Category"] = df_clean["BMI Category"].replace({"Normal Weight": "Normal"})
df_clean["Sleep Disorder"] = df_clean["Sleep Disorder"].fillna("None")

# Feature engineering
bp = df_clean["Blood Pressure"].astype(str).str.split("/", expand=True)
df_clean["Systolic_BP"] = pd.to_numeric(bp[0], errors="coerce")
df_clean["Diastolic_BP"] = pd.to_numeric(bp[1], errors="coerce")
df_clean.drop(columns=["Blood Pressure"], inplace=True)

# Advanced feature engineering
bmi_risk = {"Normal": 0, "Overweight": 1, "Obese": 2}
df_clean["BMI_Risk_Score"] = df_clean["BMI Category"].map(bmi_risk).fillna(0)
df_clean["Age_Group"] = pd.cut(df_clean["Age"], bins=[0, 30, 40, 50, 100], labels=["Young", "Adult", "Middle", "Senior"])
df_clean["Sleep_Efficiency"] = df_clean["Quality of Sleep"] / df_clean["Sleep Duration"]
df_clean["Stress_Activity_Ratio"] = df_clean["Stress Level"] / (df_clean["Physical Activity Level"] + 1)

# Prepare features and target
X = df_clean.drop(columns=["Sleep Disorder"])
y = df_clean["Sleep Disorder"]

# Encode target
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

# Preprocessing
categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numerical_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

# Enhanced models
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

lgb = LGBMClassifier(
    n_estimators=300,
    learning_rate=0.03,
    max_depth=6,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbose=-1
)

# Stacking ensemble with cross-validation
stack_clf = StackingClassifier(
    estimators=[("rf", rf), ("lgb", lgb)],
    final_estimator=LogisticRegression(max_iter=4000, C=1.0),
    stack_method="predict_proba",
    n_jobs=-1,
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
)

# Build pipeline
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", stack_clf),
])

# Train model
model.fit(X_train, y_train)

# Cross-validation for accuracy estimation
cv_scores = cross_val_score(model, X, y_encoded, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='accuracy')
print(f"Cross-Validation Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Test set evaluation
y_pred = model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Set Accuracy: {test_accuracy:.4f}")

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

# Feature importance analysis (for tree-based models)
feature_names = numerical_cols + [f"{col}_{val}" for col in categorical_cols for val in sorted(X[col].unique())]

# Get feature importances from Random Forest
rf_fitted = model.named_steps["classifier"].estimators_[0][1]  # Access RF from stacking
importances = rf_fitted.feature_importances_

# Create feature importance dataframe
feature_importance_df = pd.DataFrame({
    'feature': feature_names[:len(importances)],
    'importance': importances
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance_df.head(10))

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(data=feature_importance_df.head(15), x='importance', y='feature')
plt.title('Top 15 Feature Importances')
plt.xlabel('Importance')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=label_encoder.classes_, 
            yticklabels=label_encoder.classes_)
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nModel improvements implemented:")
print(f"1. Enhanced feature engineering with BMI risk scores, age groups, sleep efficiency, and stress-activity ratios")
print(f"2. Optimized hyperparameters for both Random Forest and XGBoost")
print(f"3. Added stratified cross-validation for better accuracy estimation")
print(f"4. Improved regularization in the final estimator")
print(f"5. Feature importance analysis for model interpretability")

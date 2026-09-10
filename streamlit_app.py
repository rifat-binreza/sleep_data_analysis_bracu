from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


DATA_FILE = "Sleep_health_and_lifestyle_dataset (1).csv"
TARGET_COLUMN = "Sleep Disorder"


@st.cache_data(show_spinner=False)
def load_and_clean_data() -> pd.DataFrame:
    """Apply the same data preparation steps as voting_cls.ipynb."""
    data_path = Path(__file__).resolve().parent / DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Include {DATA_FILE} in the repository."
        )

    df = pd.read_csv(data_path).copy()
    df[TARGET_COLUMN] = df[TARGET_COLUMN].fillna("None")
    df = df.drop(columns=["Person ID"], errors="ignore")

    if "BMI Category" in df.columns:
        df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})

    if "Blood Pressure" in df.columns:
        blood_pressure = df["Blood Pressure"].astype(str).str.split("/", expand=True)
        df["Systolic_BP"] = pd.to_numeric(blood_pressure[0], errors="coerce")
        df["Diastolic_BP"] = pd.to_numeric(blood_pressure[1], errors="coerce")
        df = df.drop(columns=["Blood Pressure"])

    return df.dropna().reset_index(drop=True)


@st.cache_resource(show_spinner="Training the voting model...")
def get_artifacts() -> dict[str, Any]:
    df = load_and_clean_data()
    X = df.drop(columns=[TARGET_COLUMN])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN])

    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )
    voting_model = VotingClassifier(
        estimators=[
            ("rf", RandomForestClassifier(n_estimators=200, random_state=42)),
            (
                "xgb",
                XGBClassifier(
                    n_estimators=200,
                    learning_rate=0.05,
                    max_depth=5,
                    eval_metric="mlogloss",
                    random_state=42,
                ),
            ),
        ],
        voting="soft",
    )
    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", voting_model)]
    )
    model.fit(X, y)

    categorical_choices = {
        column: sorted(X[column].astype(str).unique().tolist())
        for column in categorical_columns
    }
    numerical_settings = {
        column: {
            "min_value": float(X[column].min()),
            "max_value": float(X[column].max()),
            "value": float(X[column].median()),
            "step": 1.0 if pd.api.types.is_integer_dtype(X[column]) else 0.1,
        }
        for column in numerical_columns
    }

    return {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": X.columns.tolist(),
        "categorical_choices": categorical_choices,
        "numerical_settings": numerical_settings,
    }


def main() -> None:
    st.set_page_config(page_title="Sleep Disorder Prediction", page_icon="😴")
    st.title("😴 Sleep Disorder Prediction")
    st.write(
        "Enter lifestyle and biometric information to predict a sleep-disorder "
        "category with the Random Forest + XGBoost soft-voting model."
    )

    artifacts = get_artifacts()
    feature_columns = artifacts["feature_columns"]
    categorical_choices = artifacts["categorical_choices"]
    numerical_settings = artifacts["numerical_settings"]

    values: dict[str, Any] = {}
    with st.form("sleep_prediction_form"):
        left_column, right_column = st.columns(2)
        for index, feature in enumerate(feature_columns):
            container = left_column if index % 2 == 0 else right_column
            with container:
                if feature in categorical_choices:
                    choices = categorical_choices[feature]
                    values[feature] = st.selectbox(feature, choices)
                else:
                    values[feature] = st.number_input(
                        feature, **numerical_settings[feature]
                    )
        submitted = st.form_submit_button("Predict Sleep Disorder", type="primary")

    if submitted:
        input_df = pd.DataFrame([values], columns=feature_columns)
        model: Pipeline = artifacts["model"]
        label_encoder: LabelEncoder = artifacts["label_encoder"]

        predicted_code = int(model.predict(input_df)[0])
        predicted_label = label_encoder.inverse_transform([predicted_code])[0]
        probabilities = model.predict_proba(input_df)[0]
        scores = dict(
            zip(label_encoder.classes_, (float(score) for score in probabilities))
        )

        st.success(f"Predicted sleep disorder: **{predicted_label}**")
        st.subheader("Class probabilities")
        st.bar_chart(pd.Series(scores, name="Probability"))
        st.dataframe(
            pd.DataFrame(
                {"Sleep Disorder": list(scores), "Probability": list(scores.values())}
            ).sort_values("Probability", ascending=False),
            hide_index=True,
            use_container_width=True,
        )

    st.caption("For educational use only; this application is not a medical diagnosis.")


if __name__ == "__main__":
    main()

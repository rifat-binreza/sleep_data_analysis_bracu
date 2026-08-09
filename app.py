from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import gradio as gr
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier


DATA_FILE = "Sleep_health_and_lifestyle_dataset (1).csv"
TARGET_COLUMN = "Sleep Disorder"


def load_and_clean_data() -> pd.DataFrame:
    """Load and apply the same feature preparation used in voting_cls.ipynb."""
    data_path = Path(__file__).resolve().parent / DATA_FILE
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path}. Upload {DATA_FILE} to the Space repository."
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


def build_voting_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    """Build the Random Forest + XGBoost soft-voting pipeline from the notebook."""
    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_columns),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_columns),
        ]
    )

    random_forest = RandomForestClassifier(n_estimators=200, random_state=42)
    xgboost = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        eval_metric="mlogloss",
        random_state=42,
    )
    voting_classifier = VotingClassifier(
        estimators=[("rf", random_forest), ("xgb", xgboost)],
        voting="soft",
    )

    model = Pipeline(
        steps=[("preprocessor", preprocessor), ("classifier", voting_classifier)]
    )
    model.fit(X, y)
    return model


@lru_cache(maxsize=1)
def get_artifacts() -> dict[str, Any]:
    df = load_and_clean_data()
    X = df.drop(columns=[TARGET_COLUMN])

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df[TARGET_COLUMN])
    model = build_voting_model(X, y)

    categorical_columns = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_columns = X.select_dtypes(exclude=["object"]).columns.tolist()
    categorical_choices = {
        column: sorted(X[column].astype(str).unique().tolist())
        for column in categorical_columns
    }
    numerical_defaults = {
        column: {
            "minimum": float(X[column].min()),
            "maximum": float(X[column].max()),
            "value": float(X[column].median()),
            "step": 1 if pd.api.types.is_integer_dtype(X[column]) else 0.1,
        }
        for column in numerical_columns
    }

    return {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": X.columns.tolist(),
        "categorical_choices": categorical_choices,
        "numerical_defaults": numerical_defaults,
        "examples": X.sample(n=min(3, len(X)), random_state=42).values.tolist(),
    }


def predict_sleep_disorder(*values: Any) -> tuple[str, dict[str, float]]:
    artifacts = get_artifacts()
    input_df = pd.DataFrame([values], columns=artifacts["feature_columns"])

    model: Pipeline = artifacts["model"]
    label_encoder: LabelEncoder = artifacts["label_encoder"]
    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    label = label_encoder.inverse_transform([prediction])[0]
    class_names = label_encoder.inverse_transform(range(len(probabilities)))
    scores = {
        str(class_name): float(probability)
        for class_name, probability in zip(class_names, probabilities)
    }
    return str(label), scores


def build_inputs() -> list[gr.Component]:
    artifacts = get_artifacts()
    categorical_choices = artifacts["categorical_choices"]
    numerical_defaults = artifacts["numerical_defaults"]

    inputs: list[gr.Component] = []
    for column in artifacts["feature_columns"]:
        if column in categorical_choices:
            choices = categorical_choices[column]
            inputs.append(gr.Dropdown(choices=choices, value=choices[0], label=column))
        else:
            settings = numerical_defaults[column]
            inputs.append(gr.Slider(label=column, **settings))
    return inputs


with gr.Blocks(title="Sleep Disorder Prediction") as demo:
    gr.Markdown(
        """
        # Sleep Disorder Prediction
        Enter lifestyle and biometric information to obtain a prediction from the
        Random Forest + XGBoost soft-voting model.

        *For educational use only; this is not medical advice or a clinical diagnosis.*
        """
    )

    with gr.Row():
        with gr.Column():
            inputs = build_inputs()
            predict_button = gr.Button("Predict Sleep Disorder", variant="primary")
        with gr.Column():
            predicted_label = gr.Textbox(label="Predicted Sleep Disorder")
            probabilities = gr.Label(label="Class Probabilities", num_top_classes=3)

    gr.Examples(
        examples=get_artifacts()["examples"],
        inputs=inputs,
        label="Try an example",
    )
    predict_button.click(
        fn=predict_sleep_disorder,
        inputs=inputs,
        outputs=[predicted_label, probabilities],
    )


if __name__ == "__main__":
    demo.launch()

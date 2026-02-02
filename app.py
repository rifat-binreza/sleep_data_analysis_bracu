from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

DATA_FILE = "Sleep_health_and_lifestyle_dataset (1).csv"


def _dataset_path() -> Path:
    return Path(__file__).resolve().parent / DATA_FILE


def _load_raw_data() -> pd.DataFrame:
    path = _dataset_path()
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    return pd.read_csv(path)


def _clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "Person ID" in df.columns:
        df.drop(columns=["Person ID"], inplace=True)

    if "BMI Category" in df.columns:
        df["BMI Category"] = df["BMI Category"].replace({"Normal Weight": "Normal"})

    if "Blood Pressure" in df.columns:
        bp = df["Blood Pressure"].astype(str).str.split("/", expand=True)
        df["Systolic_BP"] = pd.to_numeric(bp[0], errors="coerce")
        df["Diastolic_BP"] = pd.to_numeric(bp[1], errors="coerce")
        df.drop(columns=["Blood Pressure"], inplace=True)

    if "Sleep Disorder" in df.columns:
        df["Sleep Disorder"] = df["Sleep Disorder"].fillna("None")

    return df


def _build_preprocessor(X: pd.DataFrame) -> Tuple[ColumnTransformer, List[str], List[str]]:
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
        ]
    )

    return preprocessor, categorical_cols, numerical_cols


def _build_model(X: pd.DataFrame, y: pd.Series) -> Pipeline:
    preprocessor, _, _ = _build_preprocessor(X)

    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        eval_metric="mlogloss",
        random_state=42,
    )

    stack_clf = StackingClassifier(
        estimators=[("rf", rf), ("xgb", xgb)],
        final_estimator=LogisticRegression(max_iter=4000),
        stack_method="predict_proba",
        n_jobs=-1,
    )

    model = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", stack_clf),
        ]
    )

    model.fit(X, y)
    return model


def _numeric_step(series: pd.Series) -> float:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 1.0
    is_int = values.apply(lambda v: float(v).is_integer()).all()
    return 1.0 if is_int else 0.1


def _numeric_stats(series: pd.Series) -> Tuple[float, float, float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    if values.empty:
        return 0.0, 1.0, 0.5, 0.1
    min_val = float(values.min())
    max_val = float(values.max())
    median_val = float(values.median())
    if min_val == max_val:
        min_val -= 1.0
        max_val += 1.0
    return min_val, max_val, median_val, _numeric_step(values)


@lru_cache(maxsize=1)
def _artifacts() -> Dict[str, Any]:
    raw_df = _load_raw_data()
    df = _clean_dataset(raw_df)

    X = df.drop(columns=["Sleep Disorder"])
    y = df["Sleep Disorder"]

    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    model = _build_model(X, y_encoded)

    feature_columns = X.columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    numerical_cols = X.select_dtypes(exclude=["object"]).columns.tolist()

    categorical_options: Dict[str, List[str]] = {}
    for col in categorical_cols:
        values = X[col].dropna().astype(str).unique().tolist()
        categorical_options[col] = sorted(values)

    numeric_info: Dict[str, Tuple[float, float, float, float]] = {}
    for col in numerical_cols:
        numeric_info[col] = _numeric_stats(X[col])

    return {
        "model": model,
        "label_encoder": label_encoder,
        "feature_columns": feature_columns,
        "categorical_options": categorical_options,
        "numeric_info": numeric_info,
    }


def predict_sleep_disorder(*inputs: Any) -> Tuple[str, Dict[str, float]]:
    artifacts = _artifacts()
    model: Pipeline = artifacts["model"]
    label_encoder: LabelEncoder = artifacts["label_encoder"]
    feature_columns: List[str] = artifacts["feature_columns"]

    input_dict = {col: val for col, val in zip(feature_columns, inputs)}
    input_df = pd.DataFrame([input_dict])

    pred_encoded = model.predict(input_df)[0]
    pred_label = label_encoder.inverse_transform([pred_encoded])[0]

    proba = model.predict_proba(input_df)[0]
    classes = label_encoder.inverse_transform(list(range(len(proba))))
    scores = {str(cls): float(score) for cls, score in zip(classes, proba)}
    scores = dict(sorted(scores.items(), key=lambda item: item[1], reverse=True))

    return str(pred_label), scores


def _build_inputs() -> List[gr.components.Component]:
    artifacts = _artifacts()
    feature_columns: List[str] = artifacts["feature_columns"]
    categorical_options: Dict[str, List[str]] = artifacts["categorical_options"]
    numeric_info: Dict[str, Tuple[float, float, float, float]] = artifacts["numeric_info"]

    components: List[gr.components.Component] = []
    for col in feature_columns:
        if col in categorical_options:
            choices = categorical_options[col]
            default_value = choices[0] if choices else ""
            components.append(
                gr.Dropdown(choices=choices, value=default_value, label=col)
            )
        else:
            min_val, max_val, median_val, step = numeric_info[col]
            components.append(
                gr.Slider(
                    minimum=min_val,
                    maximum=max_val,
                    value=median_val,
                    step=step,
                    label=col,
                )
            )
    return components


with gr.Blocks(title="Sleep Disorder Prediction (Stacking Model)") as demo:
    gr.Markdown(
        """
        # Sleep Disorder Prediction
        This app uses a stacking ensemble (Random Forest + XGBoost with Logistic Regression)
        to predict sleep disorder categories based on lifestyle and biometric inputs.
        """
    )

    with gr.Row():
        with gr.Column():
            inputs = _build_inputs()
        with gr.Column():
            predicted_label = gr.Textbox(label="Predicted Sleep Disorder")
            prediction_scores = gr.Label(label="Class Probabilities")

    predict_btn = gr.Button("Predict")
    predict_btn.click(
        fn=predict_sleep_disorder,
        inputs=inputs,
        outputs=[predicted_label, prediction_scores],
    )


if __name__ == "__main__":
    demo.launch()

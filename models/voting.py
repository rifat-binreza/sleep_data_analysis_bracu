"""Voting-model helper: build pipeline with OneHotEncoder(drop='first').

Usage:
    from models.voting import build_voting_pipeline, count_transformed_features
    pipeline = build_voting_pipeline(numerical_cols, categorical_cols)
    pipeline.fit(X_train, y_train)

Helpers here intentionally keep I/O out of the module so the notebook
or script can control data loading and train/test splitting.
"""
from typing import List, Optional

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from xgboost import XGBClassifier
import numpy as np


def build_voting_pipeline(
    numerical_cols: List[str],
    categorical_cols: List[str],
    rf_params: Optional[dict] = None,
    xgb_params: Optional[dict] = None,
) -> Pipeline:
    """Return a sklearn Pipeline that preprocesses then fits a VotingClassifier.

    The OneHotEncoder is constructed with `drop='first'` so each categorical
    column contributes (n_unique - 1) dummy columns. This matches the
    request to keep the total one-hot features reduced (e.g. 16 -> 13).
    """
    rf_params = rf_params or {}
    xgb_params = xgb_params or {}

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False),
                categorical_cols,
            ),
        ],
    )

    rf = RandomForestClassifier(n_estimators=200, random_state=42, **rf_params)
    xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        eval_metric="mlogloss",
        random_state=42,
        **xgb_params,
    )

    voting = VotingClassifier(estimators=[("rf", rf), ("xgb", xgb)], voting="soft", n_jobs=-1)

    pipeline = Pipeline([("preprocessor", preprocessor), ("classifier", voting)])
    return pipeline


def count_transformed_features(X, numerical_cols: List[str], categorical_cols: List[str]) -> int:
    """Estimate the number of columns produced by the preprocessor when fitted on X.

    Returns the number of output features after `ColumnTransformer` with
    OneHotEncoder(drop='first'). Useful to verify the final feature count.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numerical_cols),
            (
                "cat",
                OneHotEncoder(handle_unknown="ignore", drop="first", sparse=False),
                categorical_cols,
            ),
        ],
    )
    # fit to infer categories
    preprocessor.fit(X)
    # transform a single row to get final width without large memory use
    sample = X.iloc[:1]
    transformed = preprocessor.transform(sample)
    return int(np.shape(transformed)[1])

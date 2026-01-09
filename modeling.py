"""Model training for early mortality prediction in sepsis patients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier


@dataclass
class ModelResult:
    model_name: str
    roc_auc: float
    pr_auc: float


def _split_data(
    df: pd.DataFrame,
    target_col: str,
    group_col: str | None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if group_col and group_col in df.columns:
        groups = df[group_col].astype(str)
        unique_groups = groups.unique()
        train_groups, test_groups = train_test_split(
            unique_groups, test_size=test_size, random_state=random_state
        )
        train_df = df[groups.isin(train_groups)]
        test_df = df[groups.isin(test_groups)]
        return train_df, test_df

    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=random_state, stratify=df[target_col]
    )
    return train_df, test_df


def _build_models(random_state: int = 42) -> Dict[str, Pipeline]:
    logistic = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    solver="liblinear",
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ]
    )
    gbdt = HistGradientBoostingClassifier(
        max_depth=5, learning_rate=0.05, random_state=random_state
    )

    return {"logistic_regression": logistic, "hist_gbdt": gbdt}


def train_mortality_models(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "mortality_28d",
    group_col: str | None = "subject_id",
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[ModelResult], Dict[str, Pipeline]]:
    """Train mortality prediction models.

    Args:
        df: Input dataframe with features and target.
        feature_cols: Feature column names.
        target_col: Column with mortality label.
        group_col: Optional patient-level group column.
    """

    for col in feature_cols + [target_col]:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    train_df, test_df = _split_data(
        df, target_col=target_col, group_col=group_col, test_size=test_size
    )

    x_train = train_df[feature_cols]
    y_train = train_df[target_col]
    x_test = test_df[feature_cols]
    y_test = test_df[target_col]

    models = _build_models(random_state=random_state)
    results: List[ModelResult] = []

    for name, model in models.items():
        model.fit(x_train, y_train)
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(x_test)[:, 1]
        else:
            y_score = model.decision_function(x_test)

        roc_auc = roc_auc_score(y_test, y_score)
        pr_auc = average_precision_score(y_test, y_score)
        results.append(ModelResult(model_name=name, roc_auc=roc_auc, pr_auc=pr_auc))

    return results, models

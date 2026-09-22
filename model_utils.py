"""Data cleaning + model training helpers.

This file does NOT import Streamlit, so all of it can be tested on its own.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

TARGET = "Price"
ID_COLUMN = "Id"
NUM_FEATURES = ["Area", "Bedrooms", "Bathrooms", "Floors", "YearBuilt"]
CAT_FEATURES = ["Location", "Condition", "Garage"]
FEATURES = NUM_FEATURES + CAT_FEATURES
REQUIRED_COLUMNS = FEATURES + [TARGET]
BASELINE_NAME = "Baseline (always predict the mean)"


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def missing_columns(df: pd.DataFrame) -> list[str]:
    """Return the required columns that are not present in df."""
    return [c for c in REQUIRED_COLUMNS if c not in df.columns]


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Strip text, fix types, drop duplicates / rows with missing values.

    Returns the cleaned frame and a small quality report.
    """
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    raw_rows = len(df)
    missing_cells = int(df[REQUIRED_COLUMNS].isna().sum().sum())

    for col in NUM_FEATURES + [TARGET]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in CAT_FEATURES:
        df[col] = df[col].astype(str).str.strip()
        df.loc[df[col].isin(["nan", "None", ""]), col] = np.nan

    before = len(df)
    df = df.drop_duplicates()
    duplicates = before - len(df)

    df = df.dropna(subset=REQUIRED_COLUMNS).reset_index(drop=True)
    quality = {
        "raw_rows": raw_rows,
        "missing_cells": missing_cells,
        "duplicates": duplicates,
        "final_rows": len(df),
    }
    return df, quality


def ordered_values(df: pd.DataFrame, col: str) -> list:
    """Sorted unique values of a column (Condition gets a natural order)."""
    values = list(df[col].dropna().unique())
    if col == "Condition":
        order = ["Poor", "Fair", "Good", "Excellent"]
        known = [v for v in order if v in values]
        other = sorted(v for v in values if v not in order)
        return known + other
    try:
        return sorted(values)
    except TypeError:
        return sorted(values, key=str)


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
def make_preprocessor() -> ColumnTransformer:
    return ColumnTransformer(
        [
            ("num", StandardScaler(), NUM_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ]
    )


def get_models(seed: int) -> dict:
    return {
        "Linear Regression": LinearRegression(),
        "Ridge Regression": Ridge(alpha=1.0),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, min_samples_leaf=10, random_state=seed),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, max_depth=8, min_samples_leaf=10, random_state=seed, n_jobs=1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=100, max_depth=2, learning_rate=0.05, random_state=seed
        ),
    }


def _metrics(y_true, y_pred) -> dict:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
    }


def train_and_evaluate(df: pd.DataFrame, test_size: float = 0.2, seed: int = 42) -> dict:
    """Split the data, train every model, and collect scores + plots data."""
    X = df[FEATURES]
    y = df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed
    )

    estimators = {BASELINE_NAME: DummyRegressor(strategy="mean")}
    estimators.update(get_models(seed))

    rows, pipelines, test_preds, importances = [], {}, {}, {}
    for name, estimator in estimators.items():
        pipe = Pipeline([("prep", make_preprocessor()), ("model", estimator)])
        cv_r2 = float(np.mean(cross_val_score(pipe, X_train, y_train, cv=5, scoring="r2")))
        pipe.fit(X_train, y_train)

        train_m = _metrics(y_train, pipe.predict(X_train))
        pred = pipe.predict(X_test)
        test_m = _metrics(y_test, pred)

        rows.append(
            {
                "Model": name,
                "CV R2 (train)": cv_r2,
                "Train R2": train_m["r2"],
                "Test R2": test_m["r2"],
                "Test MAE": test_m["mae"],
                "Test RMSE": test_m["rmse"],
            }
        )
        pipelines[name] = pipe
        test_preds[name] = pred
        if name != BASELINE_NAME:
            imp = permutation_importance(
                pipe, X_test, y_test, n_repeats=5, random_state=seed, scoring="r2", n_jobs=1
            )
            importances[name] = pd.Series(imp.importances_mean, index=list(X_test.columns))

    results = pd.DataFrame(rows)
    real = results[results["Model"] != BASELINE_NAME]
    best_name = str(real.loc[real["CV R2 (train)"].idxmax(), "Model"])

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "results": results,
        "pipelines": pipelines,
        "test_preds": test_preds,
        "importances": importances,
        "best_name": best_name,
    }


def predict_price(pipeline: Pipeline, inputs: dict) -> float:
    """Predict the price for one house described by `inputs`."""
    row = pd.DataFrame([{col: inputs[col] for col in FEATURES}])
    return float(max(pipeline.predict(row)[0], 0.0))

"""Focused test suite for Prediction Playground logic, dynamic schema generation,
inference on fitted pipelines, classification/regression support, and input validation."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automl.prediction import (
    get_feature_input_schema,
    make_single_prediction,
    explain_prediction,
)
from automl.core import run_automl


def test_schema_generation():
    print("Testing dynamic feature schema generation...")
    df = pd.DataFrame({
        "num_col": [1.0, 2.5, 5.0, 10.0, np.nan],
        "int_col": [10, 20, 30, 40, 50],
        "cat_col": ["low", "medium", "high", "medium", "low"],
        "bool_col": [True, False, True, True, False],
        "target": ["class_a", "class_b", "class_a", "class_b", "class_a"],
    })
    feature_cols = ["num_col", "int_col", "cat_col", "bool_col"]

    # Schema must strictly exclude target column
    schema = get_feature_input_schema(df, feature_cols, target_col="target")
    schema_names = [item["name"] for item in schema]

    assert "target" not in schema_names, "Target column must not be present in feature schema"
    assert len(schema) == 4, f"Expected 4 feature items, got {len(schema)}"

    schema_dict = {item["name"]: item for item in schema}

    # Verify numeric column
    assert schema_dict["num_col"]["type"] == "numeric"
    assert schema_dict["num_col"]["min"] == 1.0
    assert schema_dict["num_col"]["max"] == 10.0
    assert schema_dict["num_col"]["default"] == 3.75  # median of [1.0, 2.5, 5.0, 10.0] is 3.75

    # Verify integer column
    assert schema_dict["int_col"]["type"] == "numeric"
    assert schema_dict["int_col"]["is_integer"] is True
    assert schema_dict["int_col"]["step"] == 1.0

    # Verify categorical column
    assert schema_dict["cat_col"]["type"] == "categorical"
    assert set(schema_dict["cat_col"]["options"]) == {"high", "low", "medium"}

    # Verify boolean column
    assert schema_dict["bool_col"]["type"] == "boolean"
    assert schema_dict["bool_col"]["default"] is True

    print("  -> Passed schema generation tests.")


def test_classification_prediction_and_probabilities():
    print("Testing classification predictions and class probabilities on fitted pipeline...")
    # Train actual AutoML classification pipeline on iris
    root = Path(__file__).resolve().parent.parent
    data = root / "sample_data"
    dfi = pd.read_csv(data / "iris_classification.csv")
    feats = [c for c in dfi.columns if c != "species"]

    res = run_automl(dfi, "species", feats, enable_tuning=False)
    pipeline = res["pipeline"]
    assert pipeline is not None, "Pipeline must be trained and returned"

    # Make single prediction with valid feature values
    input_values = {col: float(dfi[col].iloc[0]) for col in feats}
    pred_res = make_single_prediction(
        pipeline=pipeline,
        task="classification",
        input_values=input_values,
        feature_names=feats,
        target_name="species",
        classes=res.get("classes"),
    )

    assert pred_res["status"] == "success", f"Prediction failed: {pred_res.get('message')}"
    assert pred_res["task"] == "classification"
    assert pred_res["prediction"] in ["setosa", "versicolor", "virginica"]
    assert pred_res["prediction"] == "setosa", f"Iris setosa expected, got {pred_res['prediction']}"

    # Probabilities check
    if pred_res["has_probabilities"]:
        probs = pred_res["probabilities"]
        assert isinstance(probs, list) and len(probs) == 3
        # Ensure probabilities sum to ~1.0 (100%)
        total_p = sum(p["probability"] for p in probs)
        assert abs(total_p - 1.0) < 0.05, f"Probabilities should sum to ~1.0, got {total_p}"
        # Sorted descending
        assert probs[0]["probability"] >= probs[1]["probability"] >= probs[2]["probability"]
        assert probs[0]["class"] == pred_res["prediction"]

    # Explanation check
    exp = explain_prediction(
        task="classification",
        prediction=pred_res["prediction"],
        probabilities=pred_res.get("probabilities"),
        target_name="species",
        best_model=res.get("best_model"),
    )
    assert "species" in exp["headline"]
    assert "setosa" in exp["headline"]
    assert "estimates" in exp["disclaimer"].lower() and "guarantees" in exp["disclaimer"].lower()
    print("  -> Passed classification prediction and probability tests.")


def test_regression_prediction():
    print("Testing regression predictions on fitted pipeline...")
    root = Path(__file__).resolve().parent.parent
    data = root / "sample_data"
    dfd = pd.read_csv(data / "diabetes_regression.csv")
    feats = [c for c in dfd.columns if c != "target"]

    res = run_automl(dfd, "target", feats, enable_tuning=False)
    pipeline = res["pipeline"]

    # Sample input values using median
    input_values = {col: float(dfd[col].median()) for col in feats}

    pred_res = make_single_prediction(
        pipeline=pipeline,
        task="regression",
        input_values=input_values,
        feature_names=feats,
        target_name="target",
    )

    assert pred_res["status"] == "success", f"Prediction failed: {pred_res.get('message')}"
    assert pred_res["task"] == "regression"
    assert isinstance(pred_res["prediction"], (int, float))
    assert "formatted_prediction" in pred_res
    assert pred_res["has_probabilities"] is False
    assert pred_res["probabilities"] is None

    # Explanation check
    exp = explain_prediction(
        task="regression",
        prediction=pred_res["prediction"],
        target_name="target",
        best_model=res.get("best_model"),
    )
    assert "target" in exp["headline"]
    assert str(round(pred_res["prediction"], 2)) in exp["headline"] or f"{pred_res['prediction']:.4f}" in exp["headline"]
    assert "estimates" in exp["disclaimer"].lower() and "guarantees" in exp["disclaimer"].lower()
    print("  -> Passed regression prediction tests.")


def test_missing_values_imputation_and_robustness():
    print("Testing handling of missing values, invalid values, and pipeline integration...")
    # Build a pipeline with numeric imputer and categorical one-hot encoder
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")), ("scl", StandardScaler())]), ["num_a", "num_b"]),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(handle_unknown="ignore"))]), ["cat_c"]),
    ])
    clf = RandomForestClassifier(n_estimators=10, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", clf)])

    train_df = pd.DataFrame({
        "num_a": [1.0, 2.0, 3.0, 4.0, 5.0],
        "num_b": [10.0, 20.0, 30.0, 40.0, 50.0],
        "cat_c": ["cat", "dog", "cat", "dog", "bird"],
        "target": [0, 1, 0, 1, 0],
    })
    pipe.fit(train_df[["num_a", "num_b", "cat_c"]], train_df["target"])

    # 1. Test missing values in input record (should be imputed by pipeline without error)
    input_with_missing = {
        "num_a": np.nan,
        "num_b": 25.0,
        "cat_c": None,
    }
    pred_res = make_single_prediction(
        pipeline=pipe,
        task="classification",
        input_values=input_with_missing,
        feature_names=["num_a", "num_b", "cat_c"],
        target_name="target",
    )
    assert pred_res["status"] == "success", f"Pipeline should impute missing values: {pred_res.get('message')}"
    assert pred_res["prediction"] in ["0", "1", 0, 1]

    # 2. Test unseen category
    input_unseen_cat = {
        "num_a": 3.0,
        "num_b": 30.0,
        "cat_c": "elephant",  # Unseen category
    }
    pred_res2 = make_single_prediction(
        pipeline=pipe,
        task="classification",
        input_values=input_unseen_cat,
        feature_names=["num_a", "num_b", "cat_c"],
        target_name="target",
    )
    assert pred_res2["status"] == "success", f"Pipeline should handle unseen categories: {pred_res2.get('message')}"

    # 3. Test missing pipeline
    pred_no_pipe = make_single_prediction(
        pipeline=None,
        task="classification",
        input_values=input_unseen_cat,
        feature_names=["num_a"],
        target_name="target",
    )
    assert pred_no_pipe["status"] == "error"
    assert "No trained model pipeline" in pred_no_pipe["message"]

    # 4. Test target column leakage prevention
    # Even if target is supplied in input_values, it must not be passed to model
    input_with_target = {
        "num_a": 2.0,
        "num_b": 20.0,
        "cat_c": "dog",
        "target": 999,  # Malicious target value
    }
    pred_target_leak = make_single_prediction(
        pipeline=pipe,
        task="classification",
        input_values=input_with_target,
        feature_names=["num_a", "num_b", "cat_c"],
        target_name="target",
    )
    assert pred_target_leak["status"] == "success"
    assert "target" not in pred_target_leak["input_record"]

    # 5. Test model without predict_proba (e.g. SVC without probability=True)
    svc_pipe = Pipeline([
        ("pre", pre),
        ("model", SVC(probability=False)),
    ])
    svc_pipe.fit(train_df[["num_a", "num_b", "cat_c"]], train_df["target"])
    pred_svc = make_single_prediction(
        pipeline=svc_pipe,
        task="classification",
        input_values=input_unseen_cat,
        feature_names=["num_a", "num_b", "cat_c"],
        target_name="target",
    )
    assert pred_svc["status"] == "success"
    assert pred_svc["has_probabilities"] is False
    assert pred_svc["probabilities"] is None

    print("  -> Passed missing values and robustness tests.")


if __name__ == "__main__":
    test_schema_generation()
    test_classification_prediction_and_probabilities()
    test_regression_prediction()
    test_missing_values_imputation_and_robustness()
    print("\nALL PREDICTION PLAYGROUND TESTS PASSED!")

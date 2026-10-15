"""Focused test suite for Feature Importance Explorer calculations, methods, and explanations."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automl import evaluation as ev
from automl import plots
from automl.explain import explain_feature_importance


def test_tree_native_importance():
    print("Testing Random Forest native tree importance...")
    np.random.seed(42)
    X = pd.DataFrame({
        "num__feature_a": np.random.randn(100),
        "num__feature_b": np.random.randn(100) * 10,
    })
    y = (X["num__feature_b"] > 0).astype(int)

    pre = ColumnTransformer([("num", StandardScaler(), ["num__feature_a", "num__feature_b"])])
    model = RandomForestClassifier(n_estimators=50, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y)

    df_imp = ev.feature_importance(pipe)
    assert df_imp is not None, "feature_importance returned None for Random Forest"
    assert "feature" in df_imp.columns and "importance" in df_imp.columns
    # Check that prefixes were cleaned
    assert "feature_b" in list(df_imp["feature"]), f"Expected cleaned feature_b, got: {list(df_imp['feature'])}"
    assert (df_imp["importance"] >= 0).all(), "Importances should be non-negative"
    print("  -> Passed tree native importance.")


def test_linear_native_importance():
    print("Testing Logistic & Ridge native linear importance...")
    np.random.seed(42)
    X = pd.DataFrame({
        "x1": np.random.randn(100),
        "x2": np.random.randn(100) * 5,
    })
    y = (X["x2"] * 2 + X["x1"] > 0).astype(int)

    pre = ColumnTransformer([("num", StandardScaler(), ["x1", "x2"])])
    model = LogisticRegression()
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y)

    df_imp = ev.feature_importance(pipe)
    assert df_imp is not None, "feature_importance returned None for Logistic Regression"
    assert df_imp.iloc[0]["feature"] == "x2", f"x2 should be top feature, got {df_imp.iloc[0]['feature']}"
    assert (df_imp["importance"] >= 0).all(), "Importances should be non-negative"
    print("  -> Passed linear native importance.")


def test_permutation_importance_knn():
    print("Testing Permutation Importance for K-Nearest Neighbors (no native importance)...")
    np.random.seed(42)
    X = pd.DataFrame({
        "signal": np.random.randn(150),
        "noise": np.random.randn(150),
    })
    y = (X["signal"] > 0).astype(int)

    pre = ColumnTransformer([("num", StandardScaler(), ["signal", "noise"])])
    knn = KNeighborsClassifier(n_neighbors=5)
    pipe = Pipeline([("pre", pre), ("model", knn)])
    pipe.fit(X, y)

    # Native importance must be None for KNN
    assert ev.feature_importance(pipe) is None, "KNN should have no native feature importance"

    # Permutation importance must succeed
    df_perm = ev.permutation_importance_df(pipe, X, y, n_repeats=5)
    assert df_perm is not None, "permutation_importance_df returned None for KNN"
    assert "feature" in df_perm.columns and "importance" in df_perm.columns
    assert df_perm.iloc[0]["feature"] == "signal", f"signal should rank highest, got {df_perm.iloc[0]['feature']}"
    assert (df_perm["importance"] >= 0).all(), "Importances should be non-negative (clipped)"
    print("  -> Passed KNN permutation importance.")


def test_permutation_importance_hist_gb():
    print("Testing Permutation Importance for HistGradientBoosting (no native importance in sklearn)...")
    np.random.seed(42)
    X = pd.DataFrame({
        "key_feat": np.random.randn(100),
        "noise_feat": np.random.randn(100),
    })
    y = (X["key_feat"] > 0).astype(int)

    pre = ColumnTransformer([("num", StandardScaler(), ["key_feat", "noise_feat"])])
    gb = HistGradientBoostingClassifier(random_state=42)
    pipe = Pipeline([("pre", pre), ("model", gb)])
    pipe.fit(X, y)

    assert ev.feature_importance(pipe) is None, "HistGradientBoosting has no native feature_importances_"
    df_perm = ev.permutation_importance_df(pipe, X, y, n_repeats=5)
    assert df_perm is not None
    assert df_perm.iloc[0]["feature"] == "key_feat"
    print("  -> Passed HistGradientBoosting permutation importance.")


def test_computational_practicality_subsampling():
    print("Testing computational practicality with large dataset (> 500 rows)...")
    np.random.seed(42)
    n_rows = 2000
    X_large = pd.DataFrame({
        "f1": np.random.randn(n_rows),
        "f2": np.random.randn(n_rows),
        "f3": np.random.randn(n_rows),
    })
    y_large = (X_large["f1"] + X_large["f2"] > 0).astype(int)

    pre = ColumnTransformer([("num", StandardScaler(), ["f1", "f2", "f3"])])
    knn = KNeighborsClassifier()
    pipe = Pipeline([("pre", pre), ("model", knn)])
    pipe.fit(X_large, y_large)

    # Must complete fast because of max_samples=500 cap
    df_perm = ev.permutation_importance_df(pipe, X_large, y_large, max_samples=500, n_repeats=3)
    assert df_perm is not None
    assert len(df_perm) == 3
    print("  -> Passed computational practicality sub-sampling.")


def test_missing_values_safe_handling():
    print("Testing missing values handling in permutation importance...")
    np.random.seed(42)
    X = pd.DataFrame({
        "a": [1.0, 2.0, np.nan, 4.0, 5.0] * 20,
        "b": [np.nan, 2.0, 3.0, 4.0, 5.0] * 20,
    })
    y = np.array([0, 1, 0, 1, 0] * 20)

    pre = ColumnTransformer([("num", Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
    ]), ["a", "b"])])
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    pipe = Pipeline([("pre", pre), ("model", model)])
    pipe.fit(X, y)

    # Test permutation importance on X with NaNs
    df_perm = ev.permutation_importance_df(pipe, X, y, n_repeats=3)
    assert df_perm is not None
    assert len(df_perm) == 2
    print("  -> Passed missing values handling.")


def test_explanation_and_causation_warning():
    print("Testing explain_feature_importance and causation warning...")
    df_mock = pd.DataFrame({
        "feature": ["Income", "Age", "ZipCode"],
        "importance": [0.65, 0.25, 0.10],
    })

    exp = explain_feature_importance(df_mock, method="built-in", model_name="Random Forest")
    assert exp["has_features"] is True
    assert "causation_warning" in exp
    assert "not proof of real-world causation" in exp["causation_warning"].lower()
    assert "Mean Decrease in Impurity" in exp["method_title"]
    assert "beginner_explanation" in exp
    assert "Income" in exp["beginner_explanation"]
    assert "table_df" in exp
    assert len(exp["table_df"]) == 3
    assert "Share (%)" in exp["table_df"].columns
    assert "Cumulative (%)" in exp["table_df"].columns

    # Test empty / None handling
    exp_empty = explain_feature_importance(None, method="permutation", model_name="K-Nearest Neighbors")
    assert exp_empty["has_features"] is False
    assert "unavailable" in exp_empty["beginner_explanation"].lower()
    assert "causation_warning" in exp_empty
    print("  -> Passed explanation and causation warning tests.")


def test_plot_importance_rendering():
    print("Testing plot_importance figure creation...")
    df_mock = pd.DataFrame({
        "feature": ["FeatA", "FeatB", "FeatC"],
        "importance": [0.55, 0.30, 0.15],
    })
    fig = plots.plot_importance(df_mock, max_n=2)
    assert fig is not None

    # Empty test
    fig_empty = plots.plot_importance(None)
    assert fig_empty is not None
    print("  -> Passed plot_importance rendering.")


if __name__ == "__main__":
    test_tree_native_importance()
    test_linear_native_importance()
    test_permutation_importance_knn()
    test_permutation_importance_hist_gb()
    test_computational_practicality_subsampling()
    test_missing_values_safe_handling()
    test_explanation_and_causation_warning()
    test_plot_importance_rendering()
    print("\nALL FEATURE IMPORTANCE FOCUSED TESTS PASSED SUCCESSFULLY!")

"""Held-out evaluation metrics and feature-importance extraction."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)


def evaluate_classification(model, X_test, y_test, labels):
    y_pred = model.predict(X_test)
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1 (weighted)": f1_score(y_test, y_pred, average="weighted", zero_division=0),
        "Precision (weighted)": precision_score(y_test, y_pred, average="weighted", zero_division=0),
        "Recall (weighted)": recall_score(y_test, y_pred, average="weighted", zero_division=0),
    }
    try:
        if hasattr(model, "predict_proba"):
            proba = model.predict_proba(X_test)
            if len(labels) == 2:
                metrics["ROC AUC"] = roc_auc_score(y_test, proba[:, 1])
            else:
                metrics["ROC AUC"] = roc_auc_score(
                    y_test, proba, multi_class="ovr", average="weighted"
                )
    except Exception:
        pass  # ROC AUC is best-effort; skip if it can't be computed
    # Pass the full known label set so the matrix dimensions always match the
    # labels we display (otherwise classes absent from the test split desync them).
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    return metrics, cm, y_pred


def evaluate_regression(model, X_test, y_test):
    y_pred = model.predict(X_test)
    rmse = float(mean_squared_error(y_test, y_pred)) ** 0.5
    metrics = {
        "R²": r2_score(y_test, y_pred),
        "RMSE": rmse,
        "MAE": mean_absolute_error(y_test, y_pred),
    }
    return metrics, y_pred


def _clean_feature_name(name: str) -> str:
    """Clean pipeline transformer prefixes while preserving feature identity."""
    prefixes = ("num__", "cat_low__", "cat_high__", "remainder__")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if name.startswith(prefix):
                name = name[len(prefix):]
                changed = True
                break
    return name


def feature_importance(fitted_pipeline, max_n: int = 20):
    """Map transformed feature names to importances (tree) or |coef| (linear)."""
    try:
        pre = fitted_pipeline.named_steps["pre"]
        model = fitted_pipeline.named_steps["model"]
        names = pre.get_feature_names_out()
    except Exception:
        return None

    if hasattr(model, "feature_importances_"):
        imp = np.asarray(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_)
        imp = np.abs(coef).mean(axis=0) if coef.ndim > 1 else np.abs(coef)
    else:
        return None

    if len(imp) != len(names):
        return None

    # Clean transformer prefixes if all resulting feature names remain unique
    clean_names = [_clean_feature_name(str(n)) for n in names]
    if len(set(clean_names)) == len(clean_names):
        names = clean_names

    imp = np.nan_to_num(imp, nan=0.0)
    imp = np.maximum(imp, 0.0)

    df = pd.DataFrame({"feature": names, "importance": imp})
    return df.sort_values("importance", ascending=False).head(max_n).reset_index(drop=True)


def permutation_importance_df(
    fitted_pipeline,
    X_test,
    y_test,
    max_n: int = 20,
    n_repeats: int = 5,
    scoring=None,
    max_samples: int = 500,
):
    """Model-agnostic importance: how much the score drops when each input
    column is shuffled. Works for any model (KNN, gradient boosting, etc.).
    Reported per original input column."""
    if X_test is None or y_test is None or len(X_test) == 0:
        return None

    try:
        # Cap test sample size for computational practicality and responsiveness
        if len(X_test) > max_samples:
            sample_idx = np.random.RandomState(42).choice(len(X_test), size=max_samples, replace=False)
            X_eval = X_test.iloc[sample_idx] if hasattr(X_test, "iloc") else X_test[sample_idx]
            y_eval = y_test.iloc[sample_idx] if hasattr(y_test, "iloc") else y_test[sample_idx]
        else:
            X_eval = X_test
            y_eval = y_test

        kwargs = {"n_repeats": n_repeats, "random_state": 42, "n_jobs": -1}
        if scoring is not None:
            kwargs["scoring"] = scoring

        try:
            result = permutation_importance(fitted_pipeline, X_eval, y_eval, **kwargs)
        except Exception:
            # Fall back without custom scoring if scorer failed or is incompatible
            kwargs.pop("scoring", None)
            result = permutation_importance(fitted_pipeline, X_eval, y_eval, **kwargs)
    except Exception:
        return None

    importances = np.maximum(np.nan_to_num(result.importances_mean, nan=0.0), 0.0)
    features = list(X_test.columns) if hasattr(X_test, "columns") else [f"feature_{i}" for i in range(X_test.shape[1])]

    if len(features) != len(importances):
        return None

    df = pd.DataFrame({"feature": features, "importance": importances})
    if hasattr(result, "importances_std") and result.importances_std is not None:
        df["std"] = np.nan_to_num(result.importances_std, nan=0.0)
    return df.sort_values("importance", ascending=False).head(max_n).reset_index(drop=True)

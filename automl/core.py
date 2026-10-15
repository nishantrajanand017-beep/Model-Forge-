"""End-to-end AutoML orchestration: detect task -> split -> tournament -> tune -> evaluate.

Pure library code (no Streamlit). The single entry point is `run_automl`, which
accepts an optional `progress` callback so a UI can show live status.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import (
    train_test_split,
    cross_val_score,
    cross_validate,
    StratifiedKFold,
    KFold,
    RandomizedSearchCV,
)
from sklearn.metrics import (
    make_scorer,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    r2_score,
    mean_absolute_error,
    mean_squared_error,
)
from sklearn.pipeline import Pipeline

from .preprocessing import build_preprocessor
from .models import get_models, get_param_distributions, RANDOM_STATE
from . import evaluation as ev

# A numeric target with <= this many distinct values is treated as classification.
CLASSIF_MAX_UNIQUE = 20


def detect_task(y: pd.Series) -> str:
    """Heuristically decide 'classification' vs 'regression' from the target column."""
    y = y.dropna()
    if not pd.api.types.is_numeric_dtype(y) or pd.api.types.is_bool_dtype(y):
        return "classification"
    n_unique = y.nunique()
    if n_unique <= CLASSIF_MAX_UNIQUE and (pd.api.types.is_integer_dtype(y) or n_unique <= 10):
        return "classification"
    return "regression"


def _default_scoring(task: str, n_classes: int | None) -> str:
    if task == "regression":
        return "r2"
    if n_classes == 2:
        return "roc_auc"
    return "f1_weighted"


def run_automl(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    task_override: str = "auto",
    test_size: float = 0.2,
    cv_folds: int = 5,
    enable_tuning: bool = True,
    search_iters: int = 20,
    progress=None,
) -> dict:
    """Run the full pipeline and return a result dict consumed by the UI / report."""

    def report(msg: str, frac: float):
        if progress:
            progress(msg, frac)

    report("Preparing data", 0.05)
    data = df[list(features) + [target]].dropna(subset=[target])
    X = data[list(features)]
    y = data[target]

    task = detect_task(y) if task_override == "auto" else task_override
    classes = sorted(y.unique().tolist()) if task == "classification" else None
    n_classes = len(classes) if classes is not None else None

    report("Splitting train / test", 0.1)
    stratify = None
    if task == "classification" and y.value_counts().min() >= 2:
        stratify = y
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=stratify
    )

    pre, plan = build_preprocessor(df, list(features))
    scoring = _default_scoring(task, n_classes)

    if task == "classification":
        min_class = int(y_train.value_counts().min())
        n_splits = max(2, min(cv_folds, min_class))
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    else:
        cv = KFold(n_splits=cv_folds, shuffle=True, random_state=RANDOM_STATE)

    # --- model tournament (cross-validated) ---
    models = get_models(task)
    leaderboard = []
    n = len(models)
    for i, (name, est) in enumerate(models.items()):
        report(f"Cross-validating: {name}", 0.15 + 0.45 * (i / n))
        pipe = Pipeline([("pre", pre), ("model", est)])
        t0 = time.time()
        try:
            if task == "classification":
                scorers = {
                    "cv_score": scoring,
                    "Accuracy": make_scorer(accuracy_score),
                    "Precision": make_scorer(precision_score, average="weighted", zero_division=0),
                    "Recall": make_scorer(recall_score, average="weighted", zero_division=0),
                    "F1-score": make_scorer(f1_score, average="weighted", zero_division=0),
                }
            else:
                rmse_scorer = make_scorer(lambda y_t, y_p: float(mean_squared_error(y_t, y_p)) ** 0.5)
                scorers = {
                    "cv_score": scoring,
                    "MAE": make_scorer(mean_absolute_error),
                    "RMSE": rmse_scorer,
                    "R²": make_scorer(r2_score),
                }
            cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scorers, n_jobs=-1)
            scores = cv_res["test_cv_score"]
            row = {
                "model": name,
                "cv_score": float(np.mean(scores)),
                "cv_std": float(np.std(scores)),
                "seconds": round(time.time() - t0, 2),
            }
            if task == "classification":
                row["Accuracy"] = float(np.mean(cv_res["test_Accuracy"]))
                row["Precision"] = float(np.mean(cv_res["test_Precision"]))
                row["Recall"] = float(np.mean(cv_res["test_Recall"]))
                row["F1-score"] = float(np.mean(cv_res["test_F1-score"]))
            else:
                row["MAE"] = float(np.mean(cv_res["test_MAE"]))
                row["RMSE"] = float(np.mean(cv_res["test_RMSE"]))
                row["R²"] = float(np.mean(cv_res["test_R²"]))
            leaderboard.append(row)
        except Exception:
            # Fall back to single-metric cross_val_score if multi-metric scoring encountered an issue
            try:
                scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
                leaderboard.append({
                    "model": name,
                    "cv_score": float(np.mean(scores)),
                    "cv_std": float(np.std(scores)),
                    "seconds": round(time.time() - t0, 2),
                })
            except Exception as e:
                leaderboard.append({
                    "model": name,
                    "cv_score": float("nan"),
                    "cv_std": float("nan"),
                    "seconds": round(time.time() - t0, 2),
                    "error": str(e)[:200],
                })

    leaderboard.sort(key=lambda d: (np.isnan(d["cv_score"]), -(d["cv_score"] if not np.isnan(d["cv_score"]) else 0.0)))
    best_name = leaderboard[0]["model"]

    # --- tune (or just fit) the winner ---
    best_pipe = Pipeline([("pre", pre), ("model", get_models(task)[best_name])])
    tuned_params = {}
    dist = get_param_distributions(best_name) if enable_tuning else {}
    if dist:
        report(f"Tuning {best_name}", 0.7)
        search = RandomizedSearchCV(
            best_pipe, dist, n_iter=search_iters, cv=cv, scoring=scoring,
            n_jobs=-1, random_state=RANDOM_STATE, refit=True,
        )
        search.fit(X_train, y_train)
        best_pipe = search.best_estimator_
        tuned_params = {k.replace("model__", ""): v for k, v in search.best_params_.items()}
    else:
        report(f"Fitting {best_name}", 0.7)
        best_pipe.fit(X_train, y_train)

    # --- evaluate on held-out test set ---
    report("Evaluating on test set", 0.9)
    if task == "classification":
        metrics, cm, y_pred = ev.evaluate_classification(best_pipe, X_test, y_test, classes)
    else:
        metrics, y_pred = ev.evaluate_regression(best_pipe, X_test, y_test)
        cm = None
    imp = ev.feature_importance(best_pipe)
    importance_method = "built-in"
    if imp is None:
        report("Computing permutation importance", 0.95)
        imp = ev.permutation_importance_df(best_pipe, X_test, y_test, scoring=scoring)
        importance_method = "permutation" if imp is not None else "none"

    report("Done", 1.0)
    return {
        "task": task,
        "classes": classes,
        "n_classes": n_classes,
        "scoring": scoring,
        "preprocess_plan": plan,
        "leaderboard": leaderboard,
        "best_model": best_name,
        "tuned_params": tuned_params,
        "pipeline": best_pipe,
        "metrics": metrics,
        "confusion_matrix": cm,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "feature_importance": imp,
        "importance_method": importance_method,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "features": list(features),
        "target": target,
    }

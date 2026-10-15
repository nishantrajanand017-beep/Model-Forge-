"""Model portfolios and hyperparameter search spaces for the tournament."""
from __future__ import annotations

from scipy.stats import randint, loguniform
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

RANDOM_STATE = 42


def get_models(task: str) -> dict:
    """Return a fresh dict of {name: estimator} for the given task."""
    if task == "classification":
        return {
            "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
            "Random Forest": RandomForestClassifier(
                n_estimators=300, n_jobs=-1, class_weight="balanced", random_state=RANDOM_STATE
            ),
            "Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
            "K-Nearest Neighbors": KNeighborsClassifier(),
        }
    return {
        "Ridge Regression": Ridge(),
        "Random Forest": RandomForestRegressor(n_estimators=300, n_jobs=-1, random_state=RANDOM_STATE),
        "Gradient Boosting": HistGradientBoostingRegressor(random_state=RANDOM_STATE),
        "K-Nearest Neighbors": KNeighborsRegressor(),
    }


def get_param_distributions(name: str) -> dict:
    """RandomizedSearchCV distributions, keyed to the 'model' step of the pipeline."""
    grids = {
        "Logistic Regression": {
            "model__C": loguniform(1e-3, 1e2),
        },
        "Ridge Regression": {
            "model__alpha": loguniform(1e-3, 1e2),
        },
        "Random Forest": {
            "model__n_estimators": randint(100, 500),
            "model__max_depth": [None, 5, 10, 20, 40],
            "model__min_samples_leaf": randint(1, 10),
            "model__max_features": ["sqrt", "log2", None],
        },
        "Gradient Boosting": {
            "model__learning_rate": loguniform(1e-2, 3e-1),
            "model__max_iter": randint(100, 400),
            "model__max_leaf_nodes": randint(15, 63),
            "model__l2_regularization": loguniform(1e-3, 1e1),
        },
        "K-Nearest Neighbors": {
            "model__n_neighbors": randint(3, 30),
            "model__weights": ["uniform", "distance"],
        },
    }
    return grids.get(name, {})

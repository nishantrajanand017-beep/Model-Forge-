"""Headless end-to-end check of the engine on both sample datasets."""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from automl.core import run_automl  # noqa: E402

root = Path(__file__).resolve().parent.parent
data = root / "sample_data"

# --- classification ---
dfi = pd.read_csv(data / "iris_classification.csv")
feats = [c for c in dfi.columns if c != "species"]
r = run_automl(dfi, "species", feats, enable_tuning=True, search_iters=10)
print("CLASSIFICATION ->", r["task"], "| best:", r["best_model"], "| metrics:", r["metrics"])
assert r["task"] == "classification", "expected classification"
assert r["metrics"]["Accuracy"] > 0.85, "iris accuracy unexpectedly low"
assert r["feature_importance"] is not None, "expected feature importances"

# --- regression ---
dfd = pd.read_csv(data / "diabetes_regression.csv")
feats = [c for c in dfd.columns if c != "target"]
r2 = run_automl(dfd, "target", feats, enable_tuning=True, search_iters=10)
print("REGRESSION ->", r2["task"], "| best:", r2["best_model"], "| metrics:", r2["metrics"])
assert r2["task"] == "regression", "expected regression"
assert "R²" in r2["metrics"], "expected R2 metric"

print("\nSMOKE TEST PASSED")

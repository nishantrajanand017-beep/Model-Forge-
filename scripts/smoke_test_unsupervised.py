"""Small offline smoke test for the Unsupervised AutoML engine."""
import pathlib
import sys
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
from automl.unsupervised import run_unsupervised

rng = np.random.default_rng(11)
frame = pd.DataFrame({
    "measure_a": np.r_[rng.normal(0, 0.25, 35), rng.normal(3, 0.25, 35)],
    "measure_b": np.r_[rng.normal(0, 0.25, 35), rng.normal(3, 0.25, 35)],
    "category": ["group_a"] * 35 + ["group_b"] * 35,
})
frame.loc[0, "measure_a"] = np.nan
result = run_unsupervised(
    frame, list(frame.columns),
    ["Clustering", "Anomaly detection", "Dimensionality reduction"],
    n_clusters=2,
)
assert len(result["results"]) == len(frame)
assert {"K-Means", "DBSCAN", "Isolation Forest", "Local Outlier Factor", "One-Class SVM", "PCA"}.issubset(result["outputs"])
assert "PC1" in result["results"].columns
assert "Silhouette score" in result["leaderboard"].columns
print("Unsupervised AutoML smoke test passed.")
print(result["leaderboard"].to_string(index=False))

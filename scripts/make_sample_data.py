"""Generate tiny sample datasets (bundled with scikit-learn, no download needed)."""
from pathlib import Path

import pandas as pd
from sklearn.datasets import load_iris, load_diabetes

out = Path(__file__).resolve().parent.parent / "sample_data"
out.mkdir(exist_ok=True)

# Classification: iris with readable string labels
iris = load_iris(as_frame=True)
df_iris = iris.frame.drop(columns=["target"])
df_iris["species"] = iris.target_names[iris.target]
df_iris.to_csv(out / "iris_classification.csv", index=False)

# Regression: diabetes (continuous 'target')
dia = load_diabetes(as_frame=True)
dia.frame.to_csv(out / "diabetes_regression.csv", index=False)

print(f"Wrote sample data to {out}")

"""Automatic, leakage-safe preprocessing built on scikit-learn primitives.

Numeric columns          -> median impute + standard scale
Low-cardinality category -> most-frequent impute + one-hot encode
High-cardinality category-> most-frequent impute + ordinal encode

Everything is wrapped in a ColumnTransformer so it becomes part of a single
fitted Pipeline (no train/test leakage, one object to persist).
"""
from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

# A categorical column with <= this many distinct values is one-hot encoded;
# above it we fall back to ordinal encoding to avoid blowing up the feature space.
ONEHOT_MAX_CARDINALITY = 20


def classify_features(df: pd.DataFrame, features: list[str]):
    """Split feature columns into numeric / low-card categorical / high-card categorical."""
    numeric, cat_low, cat_high = [], [], []
    for col in features:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s):
            numeric.append(col)
        else:
            n_unique = s.nunique(dropna=True)
            if n_unique <= ONEHOT_MAX_CARDINALITY:
                cat_low.append(col)
            else:
                cat_high.append(col)
    return numeric, cat_low, cat_high


def build_preprocessor(df: pd.DataFrame, features: list[str]):
    """Return (ColumnTransformer, plan_dict) for the given feature columns."""
    numeric, cat_low, cat_high = classify_features(df, features)

    transformers = []
    if numeric:
        transformers.append((
            "num",
            Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]),
            numeric,
        ))
    if cat_low:
        transformers.append((
            "cat_low",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ]),
            cat_low,
        ))
    if cat_high:
        transformers.append((
            "cat_high",
            Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
            ]),
            cat_high,
        ))

    pre = ColumnTransformer(transformers, remainder="drop")
    plan = {
        "numeric (impute+scale)": numeric,
        "categorical one-hot": cat_low,
        "categorical ordinal": cat_high,
    }
    return pre, plan

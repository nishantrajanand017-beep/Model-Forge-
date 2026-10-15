"""Lightweight per-column data profile for the EDA panel."""
from __future__ import annotations

import pandas as pd


def profile(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in df.columns:
        s = df[col]
        rows.append({
            "column": col,
            "dtype": str(s.dtype),
            "missing %": round(float(s.isna().mean()) * 100, 1),
            "unique": int(s.nunique(dropna=True)),
            "example": s.dropna().iloc[0] if s.notna().any() else None,
        })
    return pd.DataFrame(rows)

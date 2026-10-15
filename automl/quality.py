"""Data Quality Inspector for Model Forge AutoML.

Inspects datasets for missing values, duplicate rows, data types, and potential
numerical outliers using the Interquartile Range (IQR) method.
Provides beginner-friendly quality ratings, recommendations, and explanations.
INSPECTION ONLY: Never mutates or alters the input dataset.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def detect_iqr_outliers(series: pd.Series) -> dict[str, Any]:
    """Detect potential numerical outliers using the Interquartile Range (IQR) method.

    Rule:
      - Q1 = 25th percentile
      - Q3 = 75th percentile
      - IQR = Q3 - Q1
      - Lower Bound = Q1 - 1.5 * IQR
      - Upper Bound = Q3 + 1.5 * IQR
      - Outlier: value < Lower Bound OR value > Upper Bound

    Important Note: Outliers detected this way are statistical extremes,
    not necessarily errors or corrupted data.
    """
    clean = series.dropna()
    is_num = pd.api.types.is_numeric_dtype(clean) and not pd.api.types.is_bool_dtype(clean)

    if clean.empty or not is_num:
        return {
            "count": 0,
            "percentage": 0.0,
            "q1": None,
            "q3": None,
            "iqr": None,
            "lower_bound": None,
            "upper_bound": None,
            "has_outliers": False,
        }

    q1 = float(clean.quantile(0.25))
    q3 = float(clean.quantile(0.75))
    iqr = float(q3 - q1)

    lower_bound = float(q1 - 1.5 * iqr)
    upper_bound = float(q3 + 1.5 * iqr)

    is_outlier = (clean < lower_bound) | (clean > upper_bound)
    outlier_count = int(is_outlier.sum())
    outlier_pct = round((outlier_count / len(clean)) * 100.0, 2) if len(clean) > 0 else 0.0

    return {
        "count": outlier_count,
        "percentage": outlier_pct,
        "q1": q1,
        "q3": q3,
        "iqr": iqr,
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "has_outliers": outlier_count > 0,
    }


def inspect_dataset_quality(df: pd.DataFrame | None) -> dict[str, Any]:
    """Perform a complete data quality inspection on a pandas DataFrame.

    Returns structured statistics, per-column diagnostics, overall health summary,
    actionable recommendations, and educational explanations.
    """
    if df is None or df.empty or len(df.columns) == 0:
        return {
            "status": "empty",
            "message": "Dataset is empty or contains no columns to inspect.",
            "n_rows": 0,
            "n_cols": 0,
            "n_duplicates": 0,
            "duplicate_pct": 0.0,
            "total_cells": 0,
            "total_missing_cells": 0,
            "missing_cells_pct": 0.0,
            "columns_report": [],
            "table_df": pd.DataFrame(),
            "rating": "No Data",
            "rating_badge": "EMPTY",
            "summary": "No data is loaded to inspect.",
            "recommendations": ["Upload or select a valid dataset with at least one row and column."],
            "outlier_summary": {},
        }

    n_rows = len(df)
    n_cols = len(df.columns)
    total_cells = n_rows * n_cols

    # Duplicate rows check (safe on unhashable or mixed types)
    try:
        n_duplicates = int(df.duplicated().sum())
    except Exception:
        n_duplicates = 0
    duplicate_pct = round((n_duplicates / n_rows) * 100.0, 2) if n_rows > 0 else 0.0

    # Total missing cells
    total_missing_cells = int(df.isna().sum().sum())
    missing_cells_pct = round((total_missing_cells / total_cells) * 100.0, 2) if total_cells > 0 else 0.0

    columns_report = []
    num_cols_count = 0
    cat_cols_count = 0
    bool_cols_count = 0
    total_outliers = 0
    cols_with_outliers = 0
    cols_with_missing = 0
    empty_cols = []

    for col in df.columns:
        s = df[col]
        missing_count = int(s.isna().sum())
        missing_pct = round(float(s.isna().mean()) * 100.0, 2) if n_rows > 0 else 0.0
        unique_count = int(s.nunique(dropna=True))
        dtype_str = str(s.dtype)

        is_bool = pd.api.types.is_bool_dtype(s)
        is_num = pd.api.types.is_numeric_dtype(s) and not is_bool

        if is_bool:
            col_type = "Boolean"
            bool_cols_count += 1
            outlier_info = detect_iqr_outliers(pd.Series([], dtype=float))
            outlier_display = "N/A (Boolean)"
            outlier_pct_display = "N/A"
        elif is_num:
            col_type = "Numerical"
            num_cols_count += 1
            outlier_info = detect_iqr_outliers(s)
            c_out = outlier_info["count"]
            total_outliers += c_out
            if c_out > 0:
                cols_with_outliers += 1
                outlier_display = f"{c_out:,}"
                outlier_pct_display = f"{outlier_info['percentage']:.1f}%"
            else:
                outlier_display = "0"
                outlier_pct_display = "0.0%"
        else:
            col_type = "Categorical"
            cat_cols_count += 1
            outlier_info = detect_iqr_outliers(pd.Series([], dtype=float))
            outlier_display = "N/A (Categorical)"
            outlier_pct_display = "N/A"

        if missing_count > 0:
            cols_with_missing += 1
        if missing_count == n_rows:
            empty_cols.append(str(col))

        # Status note
        if missing_count == n_rows:
            status = "Empty (100% missing)"
        elif missing_count > 0 and outlier_info["has_outliers"]:
            status = "Missing & Outliers"
        elif missing_count > 0:
            status = f"{missing_pct:.1f}% Missing"
        elif outlier_info["has_outliers"]:
            status = f"{outlier_info['count']} Outliers"
        else:
            status = "Clean"

        columns_report.append({
            "column": str(col),
            "col_type": col_type,
            "dtype": dtype_str,
            "missing_count": missing_count,
            "missing_pct": missing_pct,
            "unique_count": unique_count,
            "outlier_count": outlier_info["count"],
            "outlier_pct": outlier_info["percentage"],
            "outlier_display": outlier_display,
            "outlier_pct_display": outlier_pct_display,
            "lower_bound": outlier_info["lower_bound"],
            "upper_bound": outlier_info["upper_bound"],
            "status": status,
        })

    # Prepare user-facing summary DataFrame
    table_rows = []
    for c in columns_report:
        table_rows.append({
            "Column Name": c["column"],
            "Feature Type": c["col_type"],
            "Data Type": c["dtype"],
            "Missing Values": f"{c['missing_count']:,} ({c['missing_pct']:.1f}%)",
            "Unique Values": f"{c['unique_count']:,}",
            "Potential Outliers (1.5×IQR)": c["outlier_display"],
            "Outlier %": c["outlier_pct_display"],
            "Column Health": c["status"],
        })
    table_df = pd.DataFrame(table_rows)

    # Determine Overall Quality Rating
    recommendations = []

    if empty_cols:
        rating = "Attention Required"
        rating_badge = "REVIEW NEEDED"
        summary = (
            f"Dataset contains {len(empty_cols)} completely empty column(s). "
            f"Review recommended before model training."
        )
    elif missing_cells_pct > 15.0 or duplicate_pct > 10.0:
        rating = "Moderate Quality"
        rating_badge = "MODERATE"
        summary = (
            f"Dataset has noticeable missing values ({missing_cells_pct:.1f}%) or duplicates ({duplicate_pct:.1f}%). "
            f"Automated imputation will handle missing values, but review is suggested."
        )
    elif missing_cells_pct > 0 or n_duplicates > 0 or cols_with_outliers > 0:
        rating = "Good Quality"
        rating_badge = "GOOD"
        summary = (
            f"Dataset is in good operational shape. Minor missing values or statistical outliers "
            f"were found, which will be handled automatically by the fitted pipeline."
        )
    else:
        rating = "Excellent Quality"
        rating_badge = "EXCELLENT"
        summary = (
            "Dataset is clean and well-formed. No duplicate records, missing values, or extreme anomalies detected. "
            "Ready for modeling."
        )

    # Build beginner-friendly actionable recommendations
    if n_duplicates > 0:
        recommendations.append(
            f"**Duplicate Records ({n_duplicates:,} rows / {duplicate_pct:.1f}%):** "
            "Examine whether duplicate rows represent natural repeated events (e.g. repeated purchases) "
            "or unintentional duplicate submissions. Model Forge preserves all rows to avoid accidental data loss."
        )
    else:
        recommendations.append(
            "**No Duplicate Records:** Every row in this dataset is distinct across all attributes."
        )

    if empty_cols:
        recommendations.append(
            f"**Empty Columns:** Column(s) `{', '.join(empty_cols)}` contain 100% missing values. "
            "They supply zero predictive information and should be dropped or filled with external data."
        )
    elif total_missing_cells > 0:
        recommendations.append(
            f"**Missing Values ({total_missing_cells:,} cells / {missing_cells_pct:.1f}%):** "
            f"Found missing entries across {cols_with_missing} column(s). "
            "Our automated preprocessing pipeline handles this without data loss: numerical columns are imputed with "
            "the feature median, and categorical columns are imputed with the most frequent category."
        )
    else:
        recommendations.append(
            "**Zero Missing Values:** All rows and columns have complete data entries."
        )

    if cols_with_outliers > 0:
        recommendations.append(
            f"**Potential Outliers ({total_outliers:,} points across {cols_with_outliers} numerical columns):** "
            "These points fall outside 1.5× the interquartile range (IQR). Remember: statistical outliers are "
            "extreme values, not necessarily incorrect measurements (e.g., high-income earners or rare events). "
            "Our tree-based algorithms and standard scalers are designed to handle moderate extremes safely."
        )
    elif num_cols_count > 0:
        recommendations.append(
            "**No Outliers Detected:** Numerical feature values fall within standard interquartile bounds."
        )

    educational_guide = {
        "missing_values": (
            "**Missing Values (Nulls / NaNs):** Missing values happen when data was omitted, unavailable, or "
            "corrupted during collection. *Impact:* Most machine learning algorithms require complete numerical arrays "
            "and will error if fed a NaN. Model Forge resolves this automatically using median or mode imputation."
        ),
        "duplicate_rows": (
            "**Duplicate Rows:** Identical records where every single column matches another row. "
            "*Impact:* Duplicates can give certain observations double the influence during training, potentially "
            "causing the model to over-memorize those specific patterns."
        ),
        "outliers": (
            "**Statistical Outliers:** Values that sit unusually far away from the center of the distribution "
            "(defined by the Interquartile Range method as values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]). "
            "*Impact:* Outliers can pull linear models off-track, but they are often completely valid, high-value real-world "
            "observations (e.g. medical spikes, jackpot wins, or sensor surges). They should be understood, not deleted."
        ),
    }

    return {
        "status": "success",
        "n_rows": n_rows,
        "n_cols": n_cols,
        "num_cols_count": num_cols_count,
        "cat_cols_count": cat_cols_count,
        "bool_cols_count": bool_cols_count,
        "n_duplicates": n_duplicates,
        "duplicate_pct": duplicate_pct,
        "total_cells": total_cells,
        "total_missing_cells": total_missing_cells,
        "missing_cells_pct": missing_cells_pct,
        "cols_with_missing": cols_with_missing,
        "empty_cols": empty_cols,
        "total_outliers": total_outliers,
        "cols_with_outliers": cols_with_outliers,
        "columns_report": columns_report,
        "table_df": table_df,
        "rating": rating,
        "rating_badge": rating_badge,
        "summary": summary,
        "recommendations": recommendations,
        "educational_guide": educational_guide,
    }

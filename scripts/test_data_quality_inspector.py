"""Focused test suite for Data Quality Inspector:
missing-value counts, duplicate detection, IQR outlier detection,
categorical/boolean column types, empty/edge-case handling, and non-mutation guarantees.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automl.quality import detect_iqr_outliers, inspect_dataset_quality


def test_missing_value_detection():
    print("Testing missing value detection and percentage reporting...")
    df = pd.DataFrame({
        "col_a": [1.0, 2.0, np.nan, 4.0, 5.0],        # 1 missing (20%)
        "col_b": [np.nan, np.nan, np.nan, np.nan, np.nan],  # 5 missing (100%)
        "col_c": ["a", "b", "c", "d", "e"],          # 0 missing (0%)
    })

    res = inspect_dataset_quality(df)
    assert res["status"] == "success"
    assert res["n_rows"] == 5
    assert res["n_cols"] == 3
    assert res["total_cells"] == 15
    assert res["total_missing_cells"] == 6
    assert abs(res["missing_cells_pct"] - 40.0) < 0.01
    assert res["cols_with_missing"] == 2
    assert "col_b" in res["empty_cols"]

    col_map = {c["column"]: c for c in res["columns_report"]}
    assert col_map["col_a"]["missing_count"] == 1
    assert abs(col_map["col_a"]["missing_pct"] - 20.0) < 0.01

    assert col_map["col_b"]["missing_count"] == 5
    assert abs(col_map["col_b"]["missing_pct"] - 100.0) < 0.01
    assert "Empty" in col_map["col_b"]["status"]

    assert col_map["col_c"]["missing_count"] == 0
    assert col_map["col_c"]["missing_pct"] == 0.0

    print("  -> Passed missing value detection tests.")


def test_duplicate_detection():
    print("Testing duplicate row detection...")
    # 5 rows, row 0 and 2 are identical, row 1 and 3 are identical -> 2 duplicates
    df = pd.DataFrame({
        "num": [10, 20, 10, 20, 30],
        "cat": ["x", "y", "x", "y", "z"],
    })

    res = inspect_dataset_quality(df)
    assert res["status"] == "success"
    assert res["n_duplicates"] == 2
    assert abs(res["duplicate_pct"] - 40.0) < 0.01
    assert any("Duplicate Records" in r for r in res["recommendations"])

    # Test clean dataset with 0 duplicates
    df_clean = pd.DataFrame({
        "num": [1, 2, 3, 4],
        "cat": ["a", "b", "c", "d"],
    })
    res_clean = inspect_dataset_quality(df_clean)
    assert res_clean["n_duplicates"] == 0
    assert res_clean["duplicate_pct"] == 0.0
    assert any("No Duplicate Records" in r for r in res_clean["recommendations"])

    print("  -> Passed duplicate detection tests.")


def test_iqr_outlier_detection():
    print("Testing IQR outlier detection method...")
    # Known dataset: standard values [10, 11, 12, 10, 11, 12, 10, 12]
    # Q1 ~ 10.0, Q3 ~ 12.0, IQR ~ 2.0
    # Lower bound = 10 - 1.5*2 = 7.0
    # Upper bound = 12 + 1.5*2 = 15.0
    # Add an outlier at 100 and an outlier at -50
    vals = [10, 11, 12, 10, 11, 12, 10, 12, 100, -50]
    series = pd.Series(vals)

    iqr_res = detect_iqr_outliers(series)
    assert iqr_res["has_outliers"] is True
    assert iqr_res["count"] == 2
    assert abs(iqr_res["percentage"] - 20.0) < 0.01
    assert iqr_res["lower_bound"] is not None
    assert iqr_res["upper_bound"] is not None
    assert 100 > iqr_res["upper_bound"]
    assert -50 < iqr_res["lower_bound"]

    # Constant series (IQR == 0)
    const_series = pd.Series([5.0, 5.0, 5.0, 5.0])
    const_res = detect_iqr_outliers(const_series)
    assert const_res["iqr"] == 0.0
    assert const_res["count"] == 0
    assert const_res["has_outliers"] is False

    # Empty series or non-numeric
    assert detect_iqr_outliers(pd.Series([], dtype=float))["count"] == 0
    assert detect_iqr_outliers(pd.Series(["a", "b", "c"]))["count"] == 0

    print("  -> Passed IQR outlier detection tests.")


def test_categorical_and_boolean_columns():
    print("Testing distinction between numerical, categorical, and boolean columns...")
    df = pd.DataFrame({
        "num_col": [1.1, 2.2, 3.3, 4.4],
        "cat_col": ["red", "green", "blue", "yellow"],
        "bool_col": [True, False, True, False],
    })

    res = inspect_dataset_quality(df)
    assert res["status"] == "success"
    assert res["num_cols_count"] == 1
    assert res["cat_cols_count"] == 1
    assert res["bool_cols_count"] == 1

    col_map = {c["column"]: c for c in res["columns_report"]}
    assert col_map["num_col"]["col_type"] == "Numerical"
    assert col_map["cat_col"]["col_type"] == "Categorical"
    assert col_map["bool_col"]["col_type"] == "Boolean"

    # Outliers should be N/A for non-numerical
    assert col_map["cat_col"]["outlier_display"] == "N/A (Categorical)"
    assert col_map["bool_col"]["outlier_display"] == "N/A (Boolean)"
    assert col_map["num_col"]["outlier_display"] == "0"

    print("  -> Passed categorical and boolean column tests.")


def test_empty_and_edge_case_datasets():
    print("Testing empty and edge-case dataset handling...")
    # 1. None
    res_none = inspect_dataset_quality(None)
    assert res_none["status"] == "empty"
    assert res_none["n_rows"] == 0

    # 2. Empty DataFrame
    res_empty = inspect_dataset_quality(pd.DataFrame())
    assert res_empty["status"] == "empty"

    # 3. DataFrame with columns but 0 rows
    res_zero_rows = inspect_dataset_quality(pd.DataFrame(columns=["a", "b", "c"]))
    assert res_zero_rows["status"] == "empty"

    # 4. DataFrame with 1 row
    df_single = pd.DataFrame({"a": [42.0], "b": ["single"]})
    res_single = inspect_dataset_quality(df_single)
    assert res_single["status"] == "success"
    assert res_single["n_rows"] == 1
    assert res_single["n_cols"] == 2
    assert res_single["total_outliers"] == 0

    print("  -> Passed empty and edge-case tests.")


def test_non_mutation_guarantee():
    print("Testing that inspector never modifies the original dataset...")
    df = pd.DataFrame({
        "num": [10.0, np.nan, 30.0, 1000.0, 10.0],
        "cat": ["cat", "dog", None, "dog", "cat"],
    })
    df_copy = df.copy(deep=True)

    _ = inspect_dataset_quality(df)

    # Check df is strictly unchanged in values, dtypes, and shape
    pd.testing.assert_frame_equal(df, df_copy)
    print("  -> Passed non-mutation guarantee test.")


if __name__ == "__main__":
    test_missing_value_detection()
    test_duplicate_detection()
    test_iqr_outlier_detection()
    test_categorical_and_boolean_columns()
    test_empty_and_edge_case_datasets()
    test_non_mutation_guarantee()
    print("\nALL DATA QUALITY INSPECTOR TESTS PASSED!")

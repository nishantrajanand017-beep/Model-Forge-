"""Prediction Playground logic: dynamic input schema generation, inference on fitted pipelines,
input validation, and beginner-friendly prediction explanations."""
from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Any


def get_feature_input_schema(
    df: pd.DataFrame,
    feature_names: list[str],
    target_col: str | None = None,
) -> list[dict[str, Any]]:
    """Inspect dataset feature columns and return metadata needed to dynamically render input fields.
    Never includes the target column."""
    schema = []
    # Filter out target column strictly
    valid_features = [f for f in feature_names if f in df.columns and f != target_col]

    for col in valid_features:
        series = df[col]
        is_numeric = pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)
        is_bool = pd.api.types.is_bool_dtype(series)

        if is_numeric:
            clean_series = series.dropna()
            min_v = float(clean_series.min()) if not clean_series.empty else 0.0
            max_v = float(clean_series.max()) if not clean_series.empty else 100.0
            default_v = float(clean_series.median()) if not clean_series.empty else 0.0

            # Determine appropriate step
            is_int = pd.api.types.is_integer_dtype(series) or (clean_series == clean_series.round()).all()
            if is_int:
                step = 1.0
            else:
                rng = max_v - min_v
                step = round(rng / 100.0, 4) if rng > 0 else 0.1
                if step == 0:
                    step = 0.01

            schema.append({
                "name": col,
                "type": "numeric",
                "is_integer": bool(is_int),
                "min": min_v,
                "max": max_v,
                "default": default_v,
                "step": step,
            })
        elif is_bool:
            schema.append({
                "name": col,
                "type": "boolean",
                "default": bool(series.mode().iloc[0]) if not series.empty else True,
                "options": [True, False],
            })
        else:
            # Categorical / string
            clean_series = series.dropna().astype(str)
            unique_vals = sorted(list(clean_series.unique()))
            if not unique_vals:
                unique_vals = ["Unknown"]
            default_val = str(clean_series.mode().iloc[0]) if not clean_series.empty else unique_vals[0]

            schema.append({
                "name": col,
                "type": "categorical",
                "default": default_val,
                "options": unique_vals,
            })

    return schema


def make_single_prediction(
    pipeline: Any,
    task: str,
    input_values: dict[str, Any],
    feature_names: list[str],
    target_name: str | None = None,
    classes: list | None = None,
) -> dict[str, Any]:
    """Execute prediction on a single record using the already-fitted pipeline and preprocessing."""
    if pipeline is None:
        return {
            "status": "error",
            "message": "No trained model pipeline is available. Please run model training first.",
        }

    # Ensure target column is excluded from prediction features
    valid_features = [f for f in feature_names if f != target_name]
    if not valid_features:
        return {
            "status": "error",
            "message": "No valid input features specified for prediction.",
        }

    # Build 1-row DataFrame preserving expected column names and order
    row_data = {}
    for f in valid_features:
        val = input_values.get(f)
        row_data[f] = val

    try:
        input_df = pd.DataFrame([row_data], columns=valid_features)
    except Exception as e:
        return {
            "status": "error",
            "message": f"Failed to construct input record: {str(e)}",
        }

    try:
        # Predict using fitted pipeline (preprocessing + model)
        raw_pred = pipeline.predict(input_df)
        pred_val = raw_pred[0] if len(raw_pred) > 0 else raw_pred
    except Exception as e:
        return {
            "status": "error",
            "message": f"Prediction failed during pipeline transformation or inference: {str(e)}",
        }

    # Classification handling
    probabilities = None
    has_probabilities = False
    if task == "classification":
        # Check if predict_proba is available and reliable
        if hasattr(pipeline, "predict_proba"):
            try:
                proba_array = pipeline.predict_proba(input_df)[0]
                # Determine class labels
                model_classes = getattr(pipeline, "classes_", None)
                if model_classes is None and hasattr(pipeline, "named_steps") and "model" in pipeline.named_steps:
                    model_classes = getattr(pipeline.named_steps["model"], "classes_", None)
                if model_classes is None:
                    model_classes = classes

                if model_classes is not None and len(model_classes) == len(proba_array):
                    prob_items = []
                    for c_label, p_val in zip(model_classes, proba_array):
                        prob_items.append({
                            "class": str(c_label),
                            "probability": float(p_val),
                            "percentage": float(p_val * 100.0),
                        })
                    # Sort descending by probability
                    prob_items.sort(key=lambda x: x["probability"], reverse=True)
                    probabilities = prob_items
                    has_probabilities = True
            except Exception:
                probabilities = None
                has_probabilities = False

        return {
            "status": "success",
            "task": "classification",
            "prediction": str(pred_val),
            "probabilities": probabilities,
            "has_probabilities": has_probabilities,
            "input_record": row_data,
        }
    else:
        # Regression handling
        try:
            numeric_pred = float(pred_val)
        except Exception:
            numeric_pred = pred_val

        return {
            "status": "success",
            "task": "regression",
            "prediction": numeric_pred,
            "formatted_prediction": f"{numeric_pred:.4f}" if isinstance(numeric_pred, (int, float)) else str(numeric_pred),
            "probabilities": None,
            "has_probabilities": False,
            "input_record": row_data,
        }


def explain_prediction(
    task: str,
    prediction: Any,
    probabilities: list[dict[str, Any]] | None = None,
    target_name: str | None = None,
    best_model: str | None = None,
) -> dict[str, str]:
    """Generate beginner-friendly explanations that clarify what the prediction means
    and explicitly remind the user that predictions are estimates, not guarantees."""
    t_name = target_name or "target"
    m_name = best_model or "The trained model"

    if task == "classification":
        headline = f"The model predicts `{t_name}` is most likely **'{prediction}'**."
        if probabilities and len(probabilities) > 0:
            top_prob = probabilities[0]["percentage"]
            details = (
                f"{m_name} evaluated all learned class patterns for your input features and assigned "
                f"a **{top_prob:.1f}% confidence** to class **'{prediction}'**."
            )
        else:
            details = (
                f"{m_name} classified this input record as **'{prediction}'** based on its learned decision boundary."
            )
    else:
        headline = f"The estimated value for `{t_name}` is **{prediction:.4f}**." if isinstance(prediction, (int, float)) else f"The estimated value for `{t_name}` is **{prediction}**."
        details = (
            f"{m_name} applied its trained regression weights to your inputs to produce this continuous estimate."
        )

    disclaimer = (
        "Important: Machine learning predictions are statistical estimates based on historical training data, "
        "not real-world guarantees. The model cannot anticipate unmeasured factors or extreme scenarios outside "
        "the range of the training dataset. Always verify critical decisions with domain knowledge."
    )

    return {
        "headline": headline,
        "details": details,
        "disclaimer": disclaimer,
    }

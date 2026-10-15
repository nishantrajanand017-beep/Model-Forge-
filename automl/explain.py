"""Explain the pipeline's decisions and model evaluation results.

Pure logic with no Streamlit or external API dependencies.
Provides:
1. `build_decisions`: Architectural decision graph for data preparation & model selection.
2. `explain_results`: Unified entry point to generate plain-English explanations of supervised
   or unsupervised AutoML results.
3. Dedicated rule-based explainers for classification, regression, clustering, anomaly detection,
   PCA, and feature importance.
"""
from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


def _is_valid_number(val: Any) -> bool:
    """Check if a value is a finite, valid numerical value (excluding None, NaN, Inf, and booleans)."""
    if val is None or isinstance(val, bool):
        return False
    if isinstance(val, (int, float, np.integer, np.floating)):
        return not (np.isnan(val) or np.isinf(val))
    return False


def _model_reason(best: str, scoring: str, score: float) -> str:
    templates = {
        "Random Forest": "Bagged decision trees scored highest on cross-validated {s} ({v:.3f}); they capture non-linear feature interactions and tolerate unscaled, mixed-type inputs with low tuning sensitivity.",
        "Gradient Boosting": "Histogram gradient boosting scored highest on cross-validated {s} ({v:.3f}); sequential residual-fitting over binned features delivers strong accuracy with a favorable bias-variance tradeoff.",
        "Logistic Regression": "The L2-regularized linear classifier scored highest on cross-validated {s} ({v:.3f}); the class boundary is effectively linear here, so a high-bias, low-variance model generalizes best.",
        "Ridge Regression": "L2-regularized linear regression scored highest on cross-validated {s} ({v:.3f}); the target is well approximated by a linear combination of features, so shrinkage curbs variance without underfitting.",
        "K-Nearest Neighbors": "k-nearest neighbors scored highest on cross-validated {s} ({v:.3f}); the target is locally smooth in feature space, so non-parametric neighborhood voting beats the parametric candidates.",
    }
    t = templates.get(
        best, "This estimator scored highest on cross-validated {s} ({v:.3f}) among the candidates evaluated."
    )
    return t.format(s=scoring, v=score)


def build_decisions(result: dict) -> dict:
    """Return {'data': [...], 'model': [...]} where each item is
    {question, options:[(label, chosen_bool), ...], reason}."""
    plan = result["preprocess_plan"]
    has_numeric = bool(plan.get("numeric (impute+scale)"))
    has_onehot = bool(plan.get("categorical one-hot"))
    has_ordinal = bool(plan.get("categorical ordinal"))
    scoring = result["scoring"]
    task = result["task"]
    best = result["best_model"]

    data = []
    if has_numeric:
        data.append({
            "question": "Missing numeric values",
            "options": [("Drop rows", False), ("Mean", False), ("Median", True), ("KNN impute", False)],
            "reason": "Median imputation is robust to outliers and skew (unlike the mean) and fills in place, so no rows are discarded and the training sample size is preserved.",
        })
        data.append({
            "question": "Numeric feature scaling",
            "options": [("None", False), ("Min-Max", False), ("Standardize (z-score)", True), ("Robust", False)],
            "reason": "Standardizing to zero mean and unit variance stops large-magnitude columns from dominating distance- and gradient-based learners, which is critical for KNN and regularized models.",
        })
    if has_onehot or has_ordinal:
        data.append({
            "question": "Categorical encoding",
            "options": [("Drop", False), ("One-hot", has_onehot), ("Ordinal", has_ordinal), ("Target encode", False)],
            "reason": "Low-cardinality columns are one-hot encoded to avoid imposing a false ordinal ranking; high-cardinality columns use ordinal codes to keep the feature matrix compact and dense.",
        })

    lb = result["leaderboard"]
    best_row = next((r for r in lb if r["model"] == best), None)
    best_score = best_row["cv_score"] if best_row else float("nan")
    tuned = bool(result["tuned_params"])

    model = [
        {
            "question": "Algorithm",
            "options": [(r["model"], r["model"] == best) for r in lb],
            "reason": _model_reason(best, scoring, best_score),
        },
        {
            "question": "Hyperparameter search",
            "options": [
                ("No tuning", not tuned), ("Grid search", False),
                ("Randomized search", tuned), ("Bayesian (Optuna)", False),
            ],
            "reason": (
                "Randomized search samples the hyperparameter space far more efficiently than exhaustive "
                "grid search at a fixed budget, reaching near-optimal settings without combinatorial blow-up."
                if tuned else
                "The winning model exposes no hyperparameter space worth searching, so it was fit directly "
                "with well-tested library defaults."
            ),
        },
        {
            "question": "Validation scheme",
            "options": [
                ("Single holdout", False), ("k-fold CV", task == "regression"),
                ("Stratified k-fold", task == "classification"), ("Leave-one-out", False),
            ],
            "reason": (
                "Stratified k-fold preserves class proportions in every fold, giving a low-variance, "
                "unbiased estimate of out-of-sample " + scoring + " for model selection."
                if task == "classification" else
                "k-fold cross-validation averages " + scoring + " over several disjoint splits, giving a "
                "low-variance estimate of generalization used to rank the models."
            ),
        },
    ]

    return {"data": data, "model": model}


# ==============================================================================
# RULE-BASED METRIC EXPLAINERS (EXPLAIN MY RESULTS)
# ==============================================================================

def explain_metric(
    metric_name: str,
    value: Any,
    task: str | None = None,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Convert an individual metric name and value into an accurate plain-English explanation."""
    ctx = context or {}
    base_info: dict[str, Any] = {
        "metric": metric_name,
        "value": float(value) if _is_valid_number(value) else None,
        "formatted_value": f"{value:.4f}" if _is_valid_number(value) else "N/A",
        "level": "info",
        "label": "Informational",
        "meaning": f"Evaluation measure for {metric_name}.",
        "interpretation": "Metric value was not computed or is unavailable.",
    }

    if not _is_valid_number(value):
        return base_info

    val = float(value)

    # --- Classification Metrics ---
    if metric_name == "Accuracy":
        base_info["formatted_value"] = f"{val:.1%}"
        base_info["meaning"] = (
            "The percentage of test records where the model's predicted class exactly matched the true label."
        )
        if val >= 0.90:
            base_info["level"] = "excellent"
            base_info["label"] = "High Accuracy"
            base_info["interpretation"] = (
                f"The model correctly predicted {val:.1%} of all validation examples, "
                "demonstrating strong overall predictive reliability."
            )
        elif val >= 0.75:
            base_info["level"] = "good"
            base_info["label"] = "Solid Accuracy"
            base_info["interpretation"] = (
                f"The model correctly predicted {val:.1%} of examples, with minor misclassifications."
            )
        elif val >= 0.60:
            base_info["level"] = "moderate"
            base_info["label"] = "Moderate Accuracy"
            base_info["interpretation"] = (
                f"The model correctly predicted {val:.1%} of examples. "
                "While better than random chance, noticeable misclassifications occur."
            )
        else:
            base_info["level"] = "poor"
            base_info["label"] = "Low Accuracy"
            base_info["interpretation"] = (
                f"Accuracy is {val:.1%}. The model struggles to separate classes reliably on unseen data."
            )

    elif metric_name == "F1 (weighted)":
        base_info["formatted_value"] = f"{val:.4f}"
        base_info["meaning"] = (
            "The harmonic balance between Precision and Recall across all classes, "
            "weighted by class frequency. It penalizes extreme imbalances."
        )
        if val >= 0.85:
            base_info["level"] = "excellent"
            base_info["label"] = "Well-Balanced"
            base_info["interpretation"] = (
                f"High F1 score ({val:.4f}) indicates the model maintains strong, balanced detection "
                "across both common and rarer classes."
            )
        elif val >= 0.70:
            base_info["level"] = "good"
            base_info["label"] = "Good Balance"
            base_info["interpretation"] = (
                f"Solid F1 score ({val:.4f}) shows consistent identification across classes "
                "with acceptable trade-offs between precision and recall."
            )
        elif val >= 0.50:
            base_info["level"] = "moderate"
            base_info["label"] = "Moderate Balance"
            base_info["interpretation"] = (
                f"F1 score of {val:.4f} suggests some classes are predicted significantly more accurately than others."
            )
        else:
            base_info["level"] = "poor"
            base_info["label"] = "Imbalanced Fit"
            base_info["interpretation"] = (
                f"Low F1 score ({val:.4f}) reveals poor overall harmony between precision and recall across classes."
            )

    elif metric_name == "Precision (weighted)":
        base_info["formatted_value"] = f"{val:.1%}"
        base_info["meaning"] = (
            "When the model predicts a specific class, how often is it actually correct "
            "(minimizing false alarms and false positives)."
        )
        if val >= 0.85:
            base_info["level"] = "excellent"
            base_info["label"] = "High Precision"
            base_info["interpretation"] = (
                f"{val:.1%} of positive predictions were correct, meaning the model generates very few false alarms."
            )
        elif val >= 0.70:
            base_info["level"] = "good"
            base_info["label"] = "Good Precision"
            base_info["interpretation"] = (
                f"{val:.1%} of positive predictions were correct, reflecting good reliability when a label is assigned."
            )
        else:
            base_info["level"] = "moderate"
            base_info["label"] = "Moderate Precision"
            base_info["interpretation"] = (
                f"Precision is {val:.1%}. Approximately {1.0 - val:.1%} of positive predictions are false alarms."
            )

    elif metric_name == "Recall (weighted)":
        base_info["formatted_value"] = f"{val:.1%}"
        base_info["meaning"] = (
            "Out of all actual instances of each class, how many did the model successfully identify "
            "(minimizing missed cases and false negatives)."
        )
        if val >= 0.85:
            base_info["level"] = "excellent"
            base_info["label"] = "High Coverage"
            base_info["interpretation"] = (
                f"The model detected {val:.1%} of all true cases in the test set, missing very few positive instances."
            )
        elif val >= 0.70:
            base_info["level"] = "good"
            base_info["label"] = "Good Coverage"
            base_info["interpretation"] = (
                f"The model caught {val:.1%} of actual class occurrences, maintaining dependable sensitivity."
            )
        else:
            base_info["level"] = "moderate"
            base_info["label"] = "Moderate Recall"
            base_info["interpretation"] = (
                f"Recall is {val:.1%}. Approximately {1.0 - val:.1%} of true instances were missed or misattributed."
            )

    elif metric_name == "ROC AUC":
        base_info["formatted_value"] = f"{val:.4f}"
        base_info["meaning"] = (
            "Area Under the Receiver Operating Characteristic curve. Measures ability to rank positive cases "
            "above negative cases across all decision thresholds (0.50 is random guessing, 1.00 is perfect)."
        )
        if val >= 0.90:
            base_info["level"] = "excellent"
            base_info["label"] = "Outstanding Discrimination"
            base_info["interpretation"] = (
                f"ROC AUC of {val:.4f} demonstrates outstanding ability to distinguish between classes "
                "regardless of classification threshold."
            )
        elif val >= 0.80:
            base_info["level"] = "good"
            base_info["label"] = "Strong Discrimination"
            base_info["interpretation"] = (
                f"ROC AUC of {val:.4f} indicates solid confidence in distinguishing classes."
            )
        elif val >= 0.70:
            base_info["level"] = "moderate"
            base_info["label"] = "Acceptable Discrimination"
            base_info["interpretation"] = (
                f"ROC AUC of {val:.4f} shows fair discrimination, though some class overlap remains."
            )
        elif val >= 0.50:
            base_info["level"] = "poor"
            base_info["label"] = "Weak Discrimination"
            base_info["interpretation"] = (
                f"ROC AUC of {val:.4f} is close to random guessing (0.50), "
                "indicating difficulty ranking class likelihoods."
            )
        else:
            base_info["level"] = "poor"
            base_info["label"] = "Inverted Separation"
            base_info["interpretation"] = (
                f"ROC AUC of {val:.4f} is below 0.50, meaning class probability ranking is worse than random chance."
            )

    # --- Regression Metrics ---
    elif metric_name == "R²":
        base_info["formatted_value"] = f"{val:.4f}"
        base_info["meaning"] = (
            "The proportion of variance in the target variable explained by the model compared to simply predicting "
            "the mean (1.0 is perfect, 0.0 equals guessing the mean, negative values mean worse than guessing the mean)."
        )
        if val >= 0.80:
            base_info["level"] = "excellent"
            base_info["label"] = "Strong Explanatory Power"
            base_info["interpretation"] = (
                f"The model accounts for {val:.1%} of total target variation, demonstrating a strong predictive fit."
            )
        elif val >= 0.50:
            base_info["level"] = "good"
            base_info["label"] = "Moderate-to-Strong Fit"
            base_info["interpretation"] = (
                f"The model explains {val:.1%} of target variation. It captures dominant trends, "
                "though noticeable residual spread remains."
            )
        elif val >= 0.20:
            base_info["level"] = "moderate"
            base_info["label"] = "Modest Fit"
            base_info["interpretation"] = (
                f"The model explains {val:.1%} of target variation. It finds subtle signals, "
                "but most variation remains unexplained by the current features."
            )
        elif val >= 0.0:
            base_info["level"] = "poor"
            base_info["label"] = "Weak Fit"
            base_info["interpretation"] = (
                f"R² of {val:.4f} indicates minimal predictive connection; "
                "the model performs only slightly better than a flat average."
            )
        else:
            base_info["level"] = "poor"
            base_info["label"] = "Underperforming Baseline"
            base_info["interpretation"] = (
                f"Negative R² ({val:.4f}) indicates model predictions deviate more from actual values "
                "than simply using the dataset average."
            )

    elif metric_name == "RMSE":
        base_info["formatted_value"] = f"{val:.4f}"
        base_info["meaning"] = (
            "Root Mean Squared Error: typical deviation between predicted and actual values in target units. "
            "Heavily penalizes large outlier errors."
        )
        base_info["level"] = "info"
        base_info["label"] = "Prediction Deviation (Outlier Sensitive)"
        base_info["interpretation"] = (
            f"Predictions deviate by an average of {val:.2f} target units, "
            "with heightened sensitivity to larger mispredictions."
        )

    elif metric_name == "MAE":
        base_info["formatted_value"] = f"{val:.4f}"
        base_info["meaning"] = (
            "Mean Absolute Error: average magnitude of prediction errors in target units. "
            "Treats all error sizes linearly without amplifying outliers."
        )
        base_info["level"] = "info"
        base_info["label"] = "Average Prediction Error"
        base_info["interpretation"] = (
            f"On average, individual predictions differ from true values by {val:.2f} target units."
        )

    # --- Clustering Metrics ---
    elif metric_name == "Silhouette score":
        base_info["formatted_value"] = f"{val:.4f}"
        base_info["meaning"] = (
            "Measures how similar records are to their assigned cluster compared to neighboring clusters "
            "(-1.0 to +1.0; higher indicates clearer cluster separation)."
        )
        if val >= 0.70:
            base_info["level"] = "excellent"
            base_info["label"] = "Strong Cluster Separation"
            base_info["interpretation"] = (
                f"Silhouette score of {val:.4f} indicates distinct, dense, and well-isolated clusters."
            )
        elif val >= 0.50:
            base_info["level"] = "good"
            base_info["label"] = "Reasonable Structure"
            base_info["interpretation"] = (
                f"Silhouette score of {val:.4f} shows well-defined cluster boundaries with manageable overlap."
            )
        elif val >= 0.25:
            base_info["level"] = "moderate"
            base_info["label"] = "Weak Separation"
            base_info["interpretation"] = (
                f"Silhouette score of {val:.4f} indicates loose groupings with noticeable boundary overlap."
            )
        else:
            base_info["level"] = "poor"
            base_info["label"] = "Poor Clustering"
            base_info["interpretation"] = (
                f"Silhouette score of {val:.4f} suggests data points do not form natural separated clusters."
            )

    elif metric_name == "Davies-Bouldin index":
        base_info["formatted_value"] = f"{val:.4f}"
        base_info["meaning"] = (
            "Ratio of within-cluster scatter to between-cluster separation "
            "(lower is better; closer to 0 indicates tighter, more distinct clusters)."
        )
        if val < 0.8:
            base_info["level"] = "excellent"
            base_info["label"] = "Tightly Clustered"
            base_info["interpretation"] = (
                f"Low index ({val:.4f}) confirms clusters are compact relative to the distance between their centroids."
            )
        elif val < 1.5:
            base_info["level"] = "good"
            base_info["label"] = "Moderate Compactness"
            base_info["interpretation"] = (
                f"Davies-Bouldin index of {val:.4f} shows acceptable cluster density and centroid separation."
            )
        else:
            base_info["level"] = "moderate"
            base_info["label"] = "Diffuse Clusters"
            base_info["interpretation"] = (
                f"Higher index ({val:.4f}) indicates cluster spread is relatively wide compared to spacing between groups."
            )

    elif metric_name in {"Calinski-Harabasz", "Calinski-Harabasz score"}:
        base_info["formatted_value"] = f"{val:.1f}"
        base_info["meaning"] = (
            "Variance ratio criterion: ratio of between-cluster dispersion to within-cluster dispersion "
            "(higher scores indicate denser, well-separated clusters)."
        )
        base_info["level"] = "info"
        base_info["label"] = "Dispersion Ratio"
        base_info["interpretation"] = (
            f"Variance ratio score of {val:.1f}. Higher scores reflect denser clusters that stand apart from each other."
        )

    elif metric_name == "Clusters found":
        int_val = int(val)
        base_info["formatted_value"] = f"{int_val}"
        base_info["meaning"] = "The total number of valid non-noise clusters identified in the dataset."
        base_info["level"] = "info"
        base_info["label"] = "Cluster Count"
        base_info["interpretation"] = f"Identified {int_val} distinct group{'s' if int_val != 1 else ''} across the feature space."

    elif metric_name == "Noise points":
        int_val = int(val)
        base_info["formatted_value"] = f"{int_val:,}"
        base_info["meaning"] = "Data points rejected as unclustered noise or anomalous outliers."
        base_info["level"] = "info"
        base_info["label"] = "Unassigned Noise"
        base_info["interpretation"] = (
            f"{int_val:,} record{'s were' if int_val != 1 else ' was'} classified as noise outside cluster density thresholds."
        )

    # --- Anomaly Metrics ---
    elif metric_name == "Anomaly count":
        int_val = int(val)
        base_info["formatted_value"] = f"{int_val:,}"
        base_info["meaning"] = "Total number of records flagged as statistically unusual or outside expected density bounds."
        base_info["level"] = "info"
        base_info["label"] = "Flagged Outliers"
        base_info["interpretation"] = (
            f"{int_val:,} record{'s were' if int_val != 1 else ' was'} flagged under the configured sensitivity threshold."
        )

    elif metric_name == "Anomaly rate":
        base_info["formatted_value"] = f"{val:.1%}"
        base_info["meaning"] = (
            "Proportion of total records flagged as anomalies. Governed by the configured expected contamination prior."
        )
        base_info["level"] = "info"
        base_info["label"] = "Outlier Proportion"
        base_info["interpretation"] = (
            f"{val:.1%} of records were flagged. Note: This rate reflects your chosen contamination baseline assumption "
            "rather than ground-truth error."
        )

    # --- PCA Metrics ---
    elif metric_name == "Explained variance":
        base_info["formatted_value"] = f"{val:.1%}"
        base_info["meaning"] = (
            "Cumulative proportion of total multi-feature variation retained in the lower-dimensional projection."
        )
        if val >= 0.80:
            base_info["level"] = "excellent"
            base_info["label"] = "High Information Retention"
            base_info["interpretation"] = (
                f"The 2D projection captures {val:.1%} of all variance in the original features, "
                "preserving nearly all primary relationships."
            )
        elif val >= 0.50:
            base_info["level"] = "good"
            base_info["label"] = "Good Retention"
            base_info["interpretation"] = (
                f"The projection retains {val:.1%} of overall variance, conveying the dominant structure "
                "with modest compression loss."
            )
        else:
            base_info["level"] = "moderate"
            base_info["label"] = "Compressed Representation"
            base_info["interpretation"] = (
                f"The projection captures {val:.1%} of variance. Significant patterns remain in higher dimensions not shown in 2D."
            )

    elif metric_name == "PC1 variance":
        base_info["formatted_value"] = f"{val:.1%}"
        base_info["meaning"] = (
            "Proportion of total variance captured along the single most dominant direction of variation (Principal Component 1)."
        )
        base_info["level"] = "info"
        base_info["label"] = "Primary Component Share"
        base_info["interpretation"] = f"PC1 accounts for {val:.1%} of total dataset variation on its own."

    return base_info


def explain_classification_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Explain each calculated classification metric in plain English."""
    order = ["Accuracy", "F1 (weighted)", "Precision (weighted)", "Recall (weighted)", "ROC AUC"]
    explanations = []
    seen = set()

    for k in order:
        if k in metrics:
            seen.add(k)
            explanations.append(explain_metric(k, metrics[k], task="classification"))

    # Any additional metrics not in the standard order
    for k, v in metrics.items():
        if k not in seen:
            explanations.append(explain_metric(k, v, task="classification"))

    return explanations


def explain_regression_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Explain each calculated regression metric in plain English."""
    order = ["R²", "RMSE", "MAE"]
    explanations = []
    seen = set()

    for k in order:
        if k in metrics:
            seen.add(k)
            explanations.append(explain_metric(k, metrics[k], task="regression"))

    for k, v in metrics.items():
        if k not in seen:
            explanations.append(explain_metric(k, v, task="regression"))

    # Comparative analysis between RMSE and MAE if both are available
    rmse = metrics.get("RMSE")
    mae = metrics.get("MAE")
    if _is_valid_number(rmse) and _is_valid_number(mae) and float(mae) > 0:
        ratio = float(rmse) / float(mae)
        if ratio >= 1.35:
            for item in explanations:
                if item["metric"] == "RMSE":
                    item["interpretation"] += (
                        f" The ratio between RMSE ({float(rmse):.2f}) and MAE ({float(mae):.2f}) is {ratio:.2f}, "
                        "indicating that a subset of unusually large errors is driving up the squared error."
                    )

    return explanations


def explain_feature_importance(
    importance_df: Any,
    method: str = "built-in",
    model_name: str | None = None,
    max_features: int = 5,
) -> dict[str, Any]:
    """Explain the most influential input features derived from feature importance analysis."""
    # Method resolution and beginner-friendly metadata
    if method == "permutation":
        method_title = "Permutation Importance"
        method_badge = "Model-Agnostic Validation Test"
        method_detail = (
            "Permutation importance measures how much model prediction accuracy drops when each "
            "feature's values are randomly shuffled on held-out validation data. "
            "Features causing the steepest accuracy drops when scrambled are ranked as having the highest predictive influence."
        )
    elif model_name and any(kw in model_name for kw in ("Forest", "Tree")):
        method_title = "Mean Decrease in Impurity"
        method_badge = "Built-in Gini / Variance Impurity Reduction"
        method_detail = (
            "Tree-based importance calculates how much each feature split reduced node impurity "
            "(Gini impurity for classification or variance for regression) across all decision trees in the ensemble."
        )
    elif model_name and any(kw in model_name for kw in ("Logistic", "Ridge", "Linear", "Regression")):
        method_title = "Standardized Coefficient Magnitude"
        method_badge = "Built-in Linear Model Weights (|coef|)"
        method_detail = (
            "Because numeric features are standardized to unit variance during preprocessing, "
            "the absolute values of the linear model's learned coefficients indicate how strongly each feature drives predictions."
        )
    else:
        method_title = "Intrinsic Model Feature Importance"
        method_badge = "Built-in Model Weights"
        method_detail = (
            "Derived directly from the trained algorithm's internal parameter weights and feature utilization metrics."
        )

    causation_warning = (
        "Important Note on Causation vs. Correlation: Feature importance indicates mathematical influence on predictions, "
        "not proof of real-world causation. A top-ranked feature is strongly associated with the target in this dataset, "
        "but changing this feature in reality will not necessarily cause the outcome to change. Always verify domain factors and confounding variables before making operational decisions."
    )

    empty_table = pd.DataFrame(columns=["Rank", "Feature", "Importance", "Share (%)", "Cumulative (%)"])

    if importance_df is None or not isinstance(importance_df, pd.DataFrame) or importance_df.empty:
        return {
            "has_features": False,
            "method": method,
            "method_title": method_title,
            "method_badge": method_badge,
            "method_detail": method_detail,
            "causation_warning": causation_warning,
            "summary": "Feature importance was not computed or is unavailable for this model architecture.",
            "beginner_explanation": (
                f"Feature importance is currently unavailable for {model_name or 'the winning model'}. "
                "The model architecture does not provide intrinsic weights, and permutation testing could not be completed. "
                "However, all validation metrics, predictions, and diagnostic plots remain completely valid."
            ),
            "top_features": [],
            "table_df": empty_table,
        }

    df_sorted = importance_df.sort_values("importance", ascending=False).reset_index(drop=True)
    total_imp = float(df_sorted["importance"].sum())
    top_n = df_sorted.head(max_features)

    items = []
    table_rows = []
    running_sum = 0.0

    for idx, row in df_sorted.iterrows():
        imp = float(row["importance"])
        pct = (imp / total_imp * 100.0) if total_imp > 0 else 0.0
        running_sum += pct
        rank = idx + 1

        table_rows.append({
            "Rank": rank,
            "Feature": str(row["feature"]),
            "Importance": f"{imp:.4f}",
            "Share (%)": f"{pct:.1f}%",
            "Cumulative (%)": f"{min(100.0, running_sum):.1f}%",
        })

        if idx < max_features:
            items.append({
                "feature": str(row["feature"]),
                "importance": imp,
                "percentage": pct,
            })

    table_df = pd.DataFrame(table_rows)

    method_desc = (
        "model-agnostic permutation tests on validation data"
        if method == "permutation"
        else "intrinsic model weights/coefficients"
    )

    if items:
        lead = items[0]
        cum_top = sum(it["percentage"] for it in items[:min(3, len(items))])

        if len(items) == 1:
            summary = (
                f"Prediction decisions are overwhelmingly driven by '{lead['feature']}' "
                f"({lead['percentage']:.1f}% relative importance), measured via {method_desc}."
            )
            beginner_explanation = (
                f"The horizontal bar chart ranks the input features by how strongly the model relies on them. "
                f"The leading predictor is **'{lead['feature']}'**, contributing **{lead['percentage']:.1f}%** of the total predictive power."
            )
        else:
            names = ", ".join(f"'{it['feature']}' ({it['percentage']:.1f}%)" for it in items[:2])
            summary = (
                f"The top predictive drivers are {names}, measured via {method_desc}."
            )
            top_3_names = ", ".join(f"'{it['feature']}'" for it in items[:min(3, len(items))])
            beginner_explanation = (
                f"The horizontal bar chart ranks the input features by how heavily the trained model depends on each one when generating predictions. "
                f"The single most influential feature is **'{lead['feature']}'**, accounting for **{lead['percentage']:.1f}%** of overall feature importance. "
                f"Together, the top {min(3, len(items))} features ({top_3_names}) drive **{cum_top:.1f}%** of all predictive decisions."
            )
    else:
        summary = "No decisive features could be identified."
        beginner_explanation = "The model did not identify strong individual feature drivers in this dataset."

    return {
        "has_features": bool(items),
        "method": method,
        "method_title": method_title,
        "method_badge": method_badge,
        "method_detail": method_detail,
        "causation_warning": causation_warning,
        "summary": summary,
        "beginner_explanation": beginner_explanation,
        "top_features": items,
        "table_df": table_df,
    }


def explain_supervised_results(result: dict[str, Any] | None) -> dict[str, Any]:
    """Produce a complete plain-English explanation of a supervised AutoML tournament run."""
    if not result or not isinstance(result, dict):
        return {
            "status": "empty",
            "task": "unknown",
            "best_model": "None",
            "verdict": "No Results",
            "headline": "No supervised model results are currently loaded.",
            "summary": "Upload a dataset and execute the tournament to generate result explanations.",
            "metrics_explanations": [],
            "feature_insights": {"has_features": False, "summary": "None", "top_features": []},
            "key_takeaways": ["Run the Supervised Tournament to inspect results."],
        }

    task = result.get("task", "classification")
    best_model = result.get("best_model", "Selected Model")
    scoring = result.get("scoring", "objective")
    metrics = result.get("metrics") or {}
    n_train = result.get("n_train")
    n_test = result.get("n_test")
    target = result.get("target", "target")

    # Generate metric explanations
    if task == "classification":
        metrics_exp = explain_classification_metrics(metrics)
        primary_val = metrics.get("Accuracy")
        primary_f1 = metrics.get("F1 (weighted)")
        # Assess overall verdict
        if _is_valid_number(primary_val) and float(primary_val) >= 0.90:
            verdict = "High Confidence"
        elif _is_valid_number(primary_val) and float(primary_val) >= 0.75:
            verdict = "Solid Performance"
        elif _is_valid_number(primary_val) and float(primary_val) >= 0.60:
            verdict = "Moderate Performance"
        else:
            verdict = "Needs Review"

        headline = (
            f"**{best_model}** won the tournament for predicting `{target}`. "
            f"It achieved {primary_val:.1%} accuracy on unseen test data."
            if _is_valid_number(primary_val)
            else f"**{best_model}** was selected for predicting `{target}`."
        )
    else:
        metrics_exp = explain_regression_metrics(metrics)
        primary_val = metrics.get("R²")
        if _is_valid_number(primary_val) and float(primary_val) >= 0.80:
            verdict = "High Confidence"
        elif _is_valid_number(primary_val) and float(primary_val) >= 0.50:
            verdict = "Solid Performance"
        elif _is_valid_number(primary_val) and float(primary_val) >= 0.20:
            verdict = "Moderate Performance"
        else:
            verdict = "Needs Review"

        headline = (
            f"**{best_model}** won the tournament for estimating `{target}`. "
            f"It explains {primary_val:.1%} of target variance (R² = {primary_val:.4f})."
            if _is_valid_number(primary_val)
            else f"**{best_model}** was selected for estimating `{target}`."
        )

    # Feature insights
    feature_insights = explain_feature_importance(
        result.get("feature_importance"),
        method=result.get("importance_method", "built-in"),
        model_name=best_model,
    )

    # Algorithm selection context
    lb = result.get("leaderboard") or []
    best_row = next((r for r in lb if r.get("model") == best_model), None)
    best_cv_score = best_row.get("cv_score") if best_row else float("nan")
    algorithm_rationale = (
        _model_reason(best_model, scoring, float(best_cv_score))
        if _is_valid_number(best_cv_score)
        else f"{best_model} performed best across cross-validation splits."
    )

    # Build concise actionable takeaways
    takeaways = []
    takeaways.append(f"Model Champion: {best_model} was chosen because it achieved the highest cross-validated {scoring}.")
    if task == "classification":
        acc = metrics.get("Accuracy")
        f1 = metrics.get("F1 (weighted)")
        if _is_valid_number(acc):
            takeaways.append(f"Accuracy: Correctly classified {float(acc):.1%} of held-out test records.")
        if _is_valid_number(acc) and _is_valid_number(f1) and abs(float(acc) - float(f1)) > 0.08:
            takeaways.append("Class Imbalance Notice: Gap between Accuracy and F1 suggests minority classes have lower detection rates.")
    else:
        r2 = metrics.get("R²")
        rmse = metrics.get("RMSE")
        if _is_valid_number(r2):
            takeaways.append(f"Fit Quality: Explains {float(r2):.1%} of target variation on unseen data.")
        if _is_valid_number(rmse):
            takeaways.append(f"Expected Error: Typical prediction error is ±{float(rmse):.2f} target units.")

    if feature_insights["has_features"] and feature_insights["top_features"]:
        top_name = feature_insights["top_features"][0]["feature"]
        top_pct = feature_insights["top_features"][0]["percentage"]
        takeaways.append(f"Key Driver: Feature '{top_name}' carries the highest relative importance ({top_pct:.1f}%).")

    if n_train and n_test:
        takeaways.append(f"Generalization: Evaluated on {n_test:,} held-out records after training on {n_train:,} records.")

    summary = (
        f"{headline} Across {len(metrics_exp)} evaluation metrics, the pipeline demonstrates {verdict.lower()} behavior. "
        f"{algorithm_rationale}"
    )

    return {
        "status": "success",
        "task": task,
        "best_model": best_model,
        "verdict": verdict,
        "headline": headline,
        "summary": summary,
        "algorithm_rationale": algorithm_rationale,
        "metrics_explanations": metrics_exp,
        "feature_insights": feature_insights,
        "key_takeaways": takeaways,
    }


def explain_clustering_metrics(
    metrics: dict[str, Any],
    algorithm_name: str | None = None,
) -> list[dict[str, Any]]:
    """Explain clustering quality and structural metrics in plain English."""
    order = ["Clusters found", "Noise points", "Silhouette score", "Davies-Bouldin index", "Calinski-Harabasz score"]
    explanations = []
    seen = set()

    for k in order:
        val = metrics.get(k)
        if val is not None:
            seen.add(k)
            explanations.append(explain_metric(k, val))

    for k, v in metrics.items():
        if k not in seen and v is not None:
            explanations.append(explain_metric(k, v))

    return explanations


def explain_anomaly_metrics(
    metrics: dict[str, Any],
    algorithm_name: str | None = None,
    contamination: float | None = None,
) -> list[dict[str, Any]]:
    """Explain anomaly detection output counts and sensitivity assumptions."""
    order = ["Anomaly count", "Anomaly rate"]
    explanations = []
    for k in order:
        val = metrics.get(k)
        if val is not None:
            explanations.append(explain_metric(k, val))
    return explanations


def explain_pca_metrics(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    """Explain dimensionality reduction variance retention and compression quality."""
    order = ["Explained variance", "PC1 variance"]
    explanations = []
    for k in order:
        val = metrics.get(k)
        if val is not None:
            explanations.append(explain_metric(k, val))
    return explanations


def explain_clustering_visualizer(
    algorithm_name: str,
    labels: Any,
    n_features: int,
    used_pca: bool = True,
    has_centroids: bool = False,
) -> dict[str, Any]:
    """Provide plain-English explanations, centroid descriptions, PCA axis notes,
    and structured partition breakdown for the Clustering Visualizer."""
    clustering_concept = (
        "Clustering is an unsupervised machine learning method that automatically discovers natural groups or partitions "
        "in your data without needing predefined answers or target labels. Data points in the same cluster share similar "
        "characteristics and feature patterns, while points in different clusters are mathematically distinct."
    )

    alg_lower = (algorithm_name or "").lower()
    if "kmeans" in alg_lower or "k-means" in alg_lower:
        alg_desc = (
            "K-Means partitions data into a chosen number of compact clusters by assigning each point to its nearest cluster center (centroid)."
        )
    elif "agglomerative" in alg_lower or "hierarchical" in alg_lower:
        alg_desc = (
            "Agglomerative Clustering is a hierarchical bottom-up method that starts with every data point in its own individual group "
            "and progressively merges the closest clusters together based on linkage distance."
        )
    elif "dbscan" in alg_lower:
        alg_desc = (
            "DBSCAN (Density-Based Spatial Clustering) identifies clusters based on the local density of data points. "
            "It groups points packed closely together and flags points in sparse regions as unclustered noise/outliers."
        )
    elif "gaussian" in alg_lower or "gmm" in alg_lower or "mixture" in alg_lower:
        alg_desc = (
            "Gaussian Mixture Models (GMM) assume data points are generated from a mixture of bell-shaped probability distributions, "
            "allowing flexible elliptical cluster shapes and probabilistic cluster membership."
        )
    else:
        alg_desc = f"{algorithm_name} partitions the feature space into natural subsets based on mathematical similarity."

    # Centroid explanation (Requirement 7)
    if has_centroids or "kmeans" in alg_lower or "k-means" in alg_lower:
        centroid_explanation = (
            "**What is a Centroid?** A centroid (marked with a bold **'X'** on the scatter plot) is the mathematical center or 'anchor point' "
            "of a cluster. It represents the average (mean) value of all input features for the data points assigned to that group. "
            "K-Means works by iteratively adjusting these centroids until every point is as close as possible to its assigned center."
        )
    else:
        centroid_explanation = None

    # PCA axes explanation (Requirement 8)
    if used_pca:
        pca_axes_explanation = (
            f"**About the 2D Visualization (PCA Projection):** Because your dataset contains {n_features} features, it cannot be drawn "
            "directly on a flat 2D screen. Model Forge uses Principal Component Analysis (PCA) to compress all dimensions into 2 visual axes (PC1 and PC2). "
            "**Important:** These axes do not represent individual original columns; each axis is a mathematical combination (weighted blend) of all your "
            "original input features, oriented to capture the greatest amount of variation across your dataset."
        )
    else:
        pca_axes_explanation = (
            "**About the 2D Visualization:** The scatter plot directly maps your two input features onto the horizontal and vertical axes "
            "without dimensional reduction."
        )

    # Empty or invalid labels handling
    if labels is None or len(labels) == 0:
        empty_df = pd.DataFrame(columns=["Cluster", "Data Points (N)", "Share (%)", "Classification"])
        return {
            "clustering_concept": clustering_concept,
            "algorithm_description": alg_desc,
            "centroid_explanation": centroid_explanation,
            "pca_axes_explanation": pca_axes_explanation,
            "noise_explanation": None,
            "breakdown_df": empty_df,
            "cluster_counts": {},
            "total_points": 0,
            "n_clusters": 0,
            "noise_count": 0,
        }

    labels_arr = np.asarray(labels)
    total_pts = len(labels_arr)
    unique_labels = sorted(list(set(labels_arr.tolist())), key=lambda v: (str(v) == "-1", v))

    rows = []
    cluster_counts = {}
    noise_count = 0

    for lbl in unique_labels:
        mask = labels_arr == lbl
        pts = int(np.sum(mask))
        pct = (pts / total_pts * 100.0) if total_pts > 0 else 0.0

        if str(lbl) == "-1":
            noise_count = pts
            cluster_name = "Noise / Outlier"
            classification = "Noise Point (Unclustered)"
            cluster_counts["Noise"] = pts
        else:
            cluster_name = f"Cluster {lbl}"
            classification = "Dense Cluster"
            cluster_counts[f"Cluster {lbl}"] = pts

        rows.append({
            "Cluster": cluster_name,
            "Data Points (N)": pts,
            "Share (%)": f"{pct:.1f}%",
            "Classification": classification,
        })

    breakdown_df = pd.DataFrame(rows)

    # Noise explanation (Requirement 4)
    if noise_count > 0:
        noise_pct = (noise_count / total_pts * 100.0) if total_pts > 0 else 0.0
        noise_explanation = (
            f"**Noise Points Detected:** DBSCAN identified **{noise_count:,} data point{'s' if noise_count != 1 else ''}** "
            f"({noise_pct:.1f}% of data) in low-density regions that could not be assigned to any dense cluster. "
            "These are clearly displayed on the scatter plot with distinct **'x'** markers."
        )
    else:
        noise_explanation = None

    return {
        "clustering_concept": clustering_concept,
        "algorithm_description": alg_desc,
        "centroid_explanation": centroid_explanation,
        "pca_axes_explanation": pca_axes_explanation,
        "noise_explanation": noise_explanation,
        "breakdown_df": breakdown_df,
        "cluster_counts": cluster_counts,
        "total_points": total_pts,
        "n_clusters": len([lbl for lbl in unique_labels if str(lbl) != "-1"]),
        "noise_count": noise_count,
    }


def explain_unsupervised_results(
    unsupervised_result: dict[str, Any] | None,
    selected_model: str | None = None,
    selected_anomaly: str | None = None,
) -> dict[str, Any]:
    """Produce a complete plain-English explanation of an unsupervised AutoML run."""
    if not unsupervised_result or not isinstance(unsupervised_result, dict):
        return {
            "status": "empty",
            "headline": "No unsupervised analysis results are currently loaded.",
            "summary": "Upload a dataset and run unsupervised analysis to generate insights.",
            "clustering": None,
            "anomaly": None,
            "pca": None,
            "key_takeaways": ["Run Unsupervised Analysis to inspect results."],
        }

    outputs = unsupervised_result.get("outputs") or {}
    leaderboard = unsupervised_result.get("leaderboard")
    n_rows = unsupervised_result.get("n_rows", 0)

    clustering_info = None
    anomaly_info = None
    pca_info = None
    takeaways = []

    # 1. Clustering analysis
    cluster_names = [n for n, o in outputs.items() if o.get("kind") == "clustering"]
    if cluster_names:
        # Determine top candidate by silhouette score if leaderboard is available
        winner_name = cluster_names[0]
        winner_sil = float("nan")
        if isinstance(leaderboard, pd.DataFrame) and not leaderboard.empty:
            c_board = leaderboard[
                (leaderboard["Analysis"] == "Clustering") & leaderboard["Silhouette score"].notna()
            ]
            if not c_board.empty:
                winner_row = c_board.sort_values("Silhouette score", ascending=False).iloc[0]
                winner_name = str(winner_row["Algorithm"])
                winner_sil = float(winner_row["Silhouette score"])

        active_name = selected_model if (selected_model and selected_model in outputs) else winner_name
        active_out = outputs[active_name]
        c_metrics = active_out.get("metrics") or {}
        c_exps = explain_clustering_metrics(c_metrics, algorithm_name=active_name)

        sil_val = c_metrics.get("Silhouette score")
        if _is_valid_number(sil_val):
            c_summary = (
                f"**{active_name}** achieved a Silhouette score of {float(sil_val):.4f}. "
                "Recommendation is based strictly on cluster compactness and distinct boundary separation."
            )
            takeaways.append(
                f"Clustering Structure: {active_name} found {c_metrics.get('Clusters found', 'multiple')} clusters with {float(sil_val):.4f} silhouette separation."
            )
        else:
            c_summary = f"**{active_name}** grouped data into {c_metrics.get('Clusters found', 'multiple')} clusters."
            takeaways.append(f"Clustering: Executed {active_name} partitioning.")

        clustering_info = {
            "selected_algorithm": active_name,
            "top_candidate": winner_name,
            "summary": c_summary,
            "metrics_explanations": c_exps,
        }

    # 2. Anomaly detection analysis
    anomaly_names = [n for n, o in outputs.items() if o.get("kind") == "anomaly"]
    if anomaly_names:
        active_anom = selected_anomaly if (selected_anomaly and selected_anomaly in anomaly_names) else (
            selected_model if (selected_model and selected_model in anomaly_names) else anomaly_names[0]
        )
        anom_out = outputs[active_anom]
        a_metrics = anom_out.get("metrics") or {}
        a_exps = explain_anomaly_metrics(a_metrics, algorithm_name=active_anom)
        count = a_metrics.get("Anomaly count", 0)
        rate = a_metrics.get("Anomaly rate", 0.0)

        a_summary = (
            f"**{active_anom}** flagged {count:,} records ({rate:.1%}) as statistical outliers. "
            "These records deviate most noticeably from the primary data density."
        )
        anomaly_info = {
            "selected_detector": active_anom,
            "flagged_count": count,
            "flagged_rate": rate,
            "summary": a_summary,
            "metrics_explanations": a_exps,
        }
        takeaways.append(
            f"Anomaly Detection: Flagged {count:,} records ({rate:.1%}) under the prior sensitivity setting."
        )

    # 3. PCA analysis
    if "PCA" in outputs:
        pca_out = outputs["PCA"]
        p_metrics = pca_out.get("metrics") or {}
        p_exps = explain_pca_metrics(p_metrics)
        exp_var = p_metrics.get("Explained variance", 0.0)

        p_summary = (
            f"Principal Component Analysis condensed all features into 2 dimensions while preserving "
            f"{exp_var:.1%} of overall dataset variance."
        )
        pca_info = {
            "explained_variance": exp_var,
            "summary": p_summary,
            "metrics_explanations": p_exps,
        }
        takeaways.append(f"Dimensionality Reduction: 2D projection retains {exp_var:.1%} of original information.")

    headline = "Unsupervised analysis successfully identified underlying data patterns without target labels."
    summary = (
        f"{headline} Executed algorithms generated diagnostic profiles for "
        f"{'clustering, ' if clustering_info else ''}"
        f"{'anomaly detection, ' if anomaly_info else ''}"
        f"{'and dimensionality reduction' if pca_info else ''}."
    )

    return {
        "status": "success",
        "headline": headline,
        "summary": summary,
        "clustering": clustering_info,
        "anomaly": anomaly_info,
        "pca": pca_info,
        "key_takeaways": takeaways,
    }


def explain_results(result: dict[str, Any] | None) -> dict[str, Any]:
    """Unified entry point that safely inspects and explains any AutoML result dict (supervised or unsupervised)."""
    if not result or not isinstance(result, dict):
        return {
            "status": "empty",
            "headline": "No results available.",
            "summary": "Run an AutoML tournament or unsupervised analysis to generate explanations.",
            "metrics_explanations": [],
            "key_takeaways": ["Upload data and run an analysis to see explanations."],
        }

    # Detect if supervised
    if "task" in result and "metrics" in result:
        return explain_supervised_results(result)

    # Detect if unsupervised
    if "outputs" in result or "leaderboard" in result:
        return explain_unsupervised_results(result)

    return {
        "status": "unknown",
        "headline": "Unrecognized result structure.",
        "summary": "The provided result object does not match known supervised or unsupervised formats.",
        "metrics_explanations": [],
        "key_takeaways": [],
    }


def explain_why_model_won(result: dict[str, Any] | None) -> dict[str, Any]:
    """Explain why the winning model finished first in the supervised tournament.

    Explains:
    1. Scoring metric objective (maximized vs minimized).
    2. Exact cross-validation leaderboard ranking and score comparison against competitors.
    3. Margin over the runner-up and whether it was a close contest.
    4. Algorithmic strengths that suited the data.
    5. Hyperparameter tuning impact.
    6. Honest limitations of cross-validation ranking.
    """
    if not result or not isinstance(result, dict) or "task" not in result:
        return {
            "status": "empty",
            "headline": "No supervised tournament results are available.",
            "summary": "Run a supervised tournament to see why the winning model was chosen.",
            "comparison_items": [],
            "honest_limitations": [],
        }

    task = result.get("task", "classification")
    best_model = result.get("best_model", "Selected Model")
    scoring = result.get("scoring", "objective")
    leaderboard = result.get("leaderboard") or []
    tuned_params = result.get("tuned_params") or {}

    # 1. Scoring metric objective
    scoring_explanations = {
        "roc_auc": {
            "name": "ROC AUC (Receiver Operating Characteristic Area Under Curve)",
            "goal": "maximized",
            "direction_text": "Higher is better (range 0.0 to 1.0; 0.50 is random chance).",
            "description": (
                "Measures the probability that the classifier ranks a randomly chosen positive instance "
                "higher than a randomly chosen negative one across all decision thresholds."
            ),
        },
        "f1_weighted": {
            "name": "Weighted F1 Score",
            "goal": "maximized",
            "direction_text": "Higher is better (range 0.0 to 1.0).",
            "description": (
                "Balances Precision (minimizing false alarms) and Recall (minimizing missed cases), "
                "weighted by each class's frequency to protect against majority-class bias."
            ),
        },
        "r2": {
            "name": "R² (Coefficient of Determination)",
            "goal": "maximized",
            "direction_text": "Higher is better (1.0 is perfect fit; 0.0 equals predicting the mean).",
            "description": (
                "Measures the proportion of variance in the target variable explained by the model's predictions. "
                "Scores can be negative if predictions deviate more than a simple horizontal mean baseline."
            ),
        },
    }
    metric_info = scoring_explanations.get(
        scoring,
        {
            "name": scoring,
            "goal": "maximized",
            "direction_text": "Higher is better.",
            "description": f"Optimized using cross-validated {scoring}.",
        },
    )

    # 2. Leaderboard comparisons
    valid_rows = []
    failed_rows = []
    for r in leaderboard:
        score = r.get("cv_score")
        if _is_valid_number(score):
            valid_rows.append(r)
        else:
            failed_rows.append(r)

    # Sort valid rows descending
    valid_rows.sort(key=lambda x: float(x.get("cv_score", 0.0)), reverse=True)

    winner_row = next((r for r in valid_rows if r.get("model") == best_model), valid_rows[0] if valid_rows else None)
    best_score = float(winner_row["cv_score"]) if winner_row and _is_valid_number(winner_row.get("cv_score")) else float("nan")
    best_std = float(winner_row["cv_std"]) if winner_row and _is_valid_number(winner_row.get("cv_std")) else float("nan")

    comparison_items = []
    runner_up_name = None
    runner_up_score = None
    is_close_contest = False

    for idx, r in enumerate(valid_rows):
        m_name = r.get("model", f"Model {idx+1}")
        m_score = float(r["cv_score"])
        m_std = float(r["cv_std"]) if _is_valid_number(r.get("cv_std")) else 0.0
        diff = best_score - m_score if _is_valid_number(best_score) else 0.0
        sec = float(r.get("seconds", 0.0))

        if idx == 0 or m_name == best_model:
            role = "Winner"
        elif idx == 1:
            role = "Runner-up"
            runner_up_name = m_name
            runner_up_score = m_score
        else:
            role = "Contender"

        comparison_items.append({
            "model": m_name,
            "role": role,
            "score": m_score,
            "std": m_std,
            "diff_from_best": diff,
            "time_seconds": sec,
        })

    for r in failed_rows:
        comparison_items.append({
            "model": r.get("model", "Unknown Model"),
            "role": "Failed",
            "score": None,
            "std": None,
            "diff_from_best": None,
            "error": r.get("error", "Evaluation failed"),
            "time_seconds": float(r.get("seconds", 0.0)),
        })

    # Check if close contest
    margin_summary = ""
    if runner_up_name and _is_valid_number(runner_up_score) and _is_valid_number(best_score):
        diff = best_score - runner_up_score
        if diff < 0.015:
            is_close_contest = True
            margin_summary = (
                f"Close contest: **{best_model}** narrowly led runner-up **{runner_up_name}** "
                f"by {diff:.4f} ({best_score:.4f} vs. {runner_up_score:.4f}). "
                f"Given cross-validation variance (±{best_std:.4f}), both models performed at very similar levels."
            )
        else:
            margin_summary = (
                f"Clear winner: **{best_model}** outperformed runner-up **{runner_up_name}** "
                f"by a margin of +{diff:.4f} ({best_score:.4f} vs. {runner_up_score:.4f})."
            )
    elif len(valid_rows) == 1:
        margin_summary = f"**{best_model}** was the sole candidate successfully cross-validated."
    elif not valid_rows:
        margin_summary = "No valid model scores could be computed."

    # 3. Architectural reasons (reusing _model_reason)
    strengths_reason = _model_reason(best_model, scoring, best_score if _is_valid_number(best_score) else 0.0)

    # 4. Tuning impact
    if tuned_params:
        param_list = ", ".join(f"`{k}` = {v}" for k, v in list(tuned_params.items())[:4])
        tuning_impact = (
            f"After winning the initial tournament, **{best_model}** underwent randomized hyperparameter tuning. "
            f"Optimized parameters: {param_list}."
        )
    else:
        tuning_impact = (
            f"**{best_model}** was trained using robust library defaults without additional hyperparameter tuning."
        )

    # 5. Honest limitations
    limitations = [
        "Cross-validation ranking is specific to this dataset and train/test split. A model that scored slightly higher on these folds may perform differently if the real-world data distribution shifts.",
        f"The tournament optimized strictly for `{scoring}`. A different objective (such as maximizing Recall or minimizing MAE) could favor a different model architecture.",
    ]
    if is_close_contest and runner_up_name:
        limitations.append(
            f"The performance gap between {best_model} and {runner_up_name} is small. "
            "In production, operational factors like inference latency, model interpretability, and memory footprint may matter more than this narrow score difference."
        )

    headline = (
        f"**{best_model}** earned the highest rank in the tournament with a cross-validated "
        f"`{scoring}` score of **{best_score:.4f}** (±{best_std:.4f})."
        if _is_valid_number(best_score)
        else f"**{best_model}** was selected for this task."
    )

    return {
        "status": "success",
        "best_model": best_model,
        "task": task,
        "scoring": scoring,
        "metric_info": metric_info,
        "headline": headline,
        "margin_summary": margin_summary,
        "is_close_contest": is_close_contest,
        "comparison_items": comparison_items,
        "strengths_reason": strengths_reason,
        "tuning_impact": tuning_impact,
        "honest_limitations": limitations,
    }


def explain_unsupervised_selection(
    unsupervised_result: dict[str, Any] | None,
    selected_model: str | None = None,
) -> dict[str, Any]:
    """Explain how unsupervised algorithms were evaluated and compared, without falsely declaring a global 'winner'.

    Explains:
    1. Why unsupervised learning cannot declare a universal winning model (absence of ground truth).
    2. How clustering candidates were compared via Silhouette score (higher is better).
    3. Why anomaly detection models cannot be ranked against each other by accuracy.
    4. PCA's role in variance retention rather than algorithm competition.
    5. Inconclusive or missing comparisons explained honestly.
    """
    if not unsupervised_result or not isinstance(unsupervised_result, dict):
        return {
            "status": "empty",
            "headline": "No unsupervised analysis results are currently available.",
            "summary": "Run unsupervised analysis to inspect how algorithms were evaluated.",
            "clustering": None,
            "anomaly": None,
            "pca": None,
            "honest_guidance": [],
        }

    outputs = unsupervised_result.get("outputs") or {}
    clustering_comp = None
    anomaly_comp = None
    pca_comp = None

    # 1. Clustering comparison
    cluster_names = [n for n, o in outputs.items() if o.get("kind") == "clustering"]
    if cluster_names:
        c_items = []
        valid_sil = []
        for name in cluster_names:
            m = outputs[name].get("metrics") or {}
            sil = m.get("Silhouette score")
            n_clusters = m.get("Clusters found")
            noise = m.get("Noise points", 0)
            is_valid = _is_valid_number(sil)
            if is_valid:
                valid_sil.append((name, float(sil), n_clusters, noise))
            c_items.append({
                "algorithm": name,
                "silhouette": float(sil) if is_valid else None,
                "clusters": n_clusters,
                "noise": noise,
            })

        valid_sil.sort(key=lambda x: x[1], reverse=True)

        if valid_sil:
            top_name, top_score, top_k, top_noise = valid_sil[0]
            summary = (
                f"Among the {len(cluster_names)} clustering algorithm(s) tested, **{top_name}** achieved the highest "
                f"Silhouette score (**{top_score:.4f}**), forming {top_k} clusters."
            )
            caveat = (
                "Objective: **Silhouette score is maximized (higher is better, scale -1.0 to +1.0)**. "
                "It measures how compact clusters are internally versus how well they are separated from other clusters. "
                "However, a higher silhouette score does not mean this clustering is objectively 'true'—it reflects "
                "separation under Euclidean distance assumptions. Non-spherical or overlapping clusters in reality may "
                "be better captured by density-based (DBSCAN) or probabilistic (Gaussian Mixture) models despite lower silhouette scores."
            )
        else:
            top_name = cluster_names[0]
            top_score = None
            summary = "No clustering algorithms produced a defined Silhouette score (e.g., single cluster or all noise points)."
            caveat = "Silhouette scores require at least two distinct, non-noise clusters to compute."

        clustering_comp = {
            "top_algorithm": top_name,
            "top_score": top_score,
            "summary": summary,
            "caveat": caveat,
            "candidates": c_items,
        }

    # 2. Anomaly detection comparison
    anomaly_names = [n for n, o in outputs.items() if o.get("kind") == "anomaly"]
    if anomaly_names:
        a_items = []
        for name in anomaly_names:
            m = outputs[name].get("metrics") or {}
            a_items.append({
                "detector": name,
                "count": m.get("Anomaly count", 0),
                "rate": m.get("Anomaly rate", 0.0),
            })
        anomaly_comp = {
            "detectors": a_items,
            "explanation": (
                "Anomaly detectors **cannot be ranked by a 'winning' accuracy score** because the data has no ground-truth "
                "labels. Each algorithm flagged outliers according to its mathematical model:\n"
                "- **Isolation Forest**: Separates anomalies by random feature slicing (outliers require fewer splits).\n"
                "- **Local Outlier Factor**: Flags records with significantly lower local density than their neighbors.\n"
                "- **One-Class SVM**: Fits a boundary enclosing the densest data region.\n\n"
                "Differences in flagged counts show differing perspectives on what constitutes an outlier, not superior accuracy."
            ),
        }

    # 3. PCA context
    if "PCA" in outputs:
        pca_m = outputs["PCA"].get("metrics") or {}
        exp_var = pca_m.get("Explained variance", 0.0)
        pca_comp = {
            "explained_variance": exp_var,
            "explanation": (
                f"PCA is evaluated by **Explained Variance** ({exp_var:.1%} retained in 2 components). "
                "Higher explained variance is better because it preserves more original information in 2D, "
                "but PCA does not compete with clustering or anomaly detection—it provides coordinate compression."
            ),
        }

    honest_guidance = [
        "Unsupervised models have no target labels, so there is no objective 'ground-truth' benchmark to prove one algorithm is universally correct.",
        "Clustering rankings reflect geometric compactness (Silhouette score), which favors spherical, well-separated groupings.",
        "Anomaly detection outputs reflect the configured prior contamination setting rather than verified errors.",
    ]

    headline = (
        "In unsupervised learning, models are evaluated on structural properties (cluster separation, variance retention) "
        "rather than a single 'winning' prediction accuracy."
    )

    return {
        "status": "success",
        "headline": headline,
        "clustering": clustering_comp,
        "anomaly": anomaly_comp,
        "pca": pca_comp,
        "honest_guidance": honest_guidance,
    }


def recommend_next_steps_supervised(result: dict[str, Any] | None) -> dict[str, Any]:
    """Generate concise, actionable next-step recommendations for supervised AutoML.

    Based strictly on available test metrics, confusion matrix, feature importance, and task type.
    Never invents uncalculated metrics or presumes unproven failure modes like overfitting.
    """
    if not result or not isinstance(result, dict) or "task" not in result:
        return {
            "status": "empty",
            "headline": "No supervised recommendations available.",
            "recommendations": [],
            "disclaimer": "Recommendations are analytical guidance based on calculated results, not absolute guarantees.",
        }

    task = result.get("task", "classification")
    metrics = result.get("metrics") or {}
    best_model = result.get("best_model", "Selected Model")
    y_test = result.get("y_test")
    n_test = result.get("n_test", len(y_test) if y_test is not None else 0)
    feat_imp = result.get("feature_importance")
    has_cm = result.get("confusion_matrix") is not None

    recommendations: list[dict[str, str]] = []

    if task == "classification":
        acc = metrics.get("Accuracy")
        f1 = metrics.get("F1 (weighted)")
        prec = metrics.get("Precision (weighted)")
        rec = metrics.get("Recall (weighted)")

        has_acc = _is_valid_number(acc)
        has_f1 = _is_valid_number(f1)
        has_prec = _is_valid_number(prec)
        has_rec = _is_valid_number(rec)

        imbalance_evidence = False
        imbalance_details = ""
        if y_test is not None and hasattr(y_test, "value_counts"):
            try:
                vc = y_test.value_counts()
                if len(vc) >= 2:
                    min_c = int(vc.min())
                    max_c = int(vc.max())
                    if max_c > 0 and (min_c / max_c) <= 0.40:
                        imbalance_evidence = True
                        imbalance_details = f"Test distribution shows class counts ranging from {min_c} to {max_c}."
            except Exception:
                pass

        if has_acc and has_f1 and (float(acc) - float(f1) >= 0.05 or imbalance_evidence):
            recommendations.append({
                "title": "Evaluate Precision, Recall, and Minority Classes",
                "advice": (
                    f"Overall Accuracy is {float(acc):.1%}, but F1 (weighted) is {float(f1):.4f}. "
                    + (f"{imbalance_details} " if imbalance_evidence else "")
                    + "When class distributions are uneven, high accuracy can mask missed instances in smaller classes. "
                    "Review Precision and Recall specifically for low-frequency classes rather than relying solely on accuracy."
                ),
            })
        elif has_acc and float(acc) < 0.70:
            recommendations.append({
                "title": "Review Precision and Recall Trade-offs",
                "advice": (
                    f"Accuracy is currently {float(acc):.1%}. Review whether your application tolerates false positives "
                    f"(monitored by Precision{f' = {float(prec):.1%}' if has_prec else ''}) more easily than missed cases "
                    f"(monitored by Recall{f' = {float(rec):.1%}' if has_rec else ''}) to determine which threshold adjustments are necessary."
                ),
            })
        else:
            acc_text = f"{float(acc):.1%}" if has_acc else "N/A"
            f1_text = f"{float(f1):.4f}" if has_f1 else "N/A"
            recommendations.append({
                "title": "Confirm Operational Thresholds",
                "advice": (
                    f"The model demonstrated consistent test metrics (Accuracy: {acc_text}, "
                    f"F1: {f1_text}). Determine whether your real-world use case prioritizes "
                    "avoiding false alarms (requiring high Precision) or avoiding missed detections (requiring high Recall)."
                ),
            })

        if has_cm:
            recommendations.append({
                "title": "Inspect Confusion Matrix and Per-Class Metrics",
                "advice": (
                    "Review the Confusion Matrix and per-class breakdown in the Diagnostic Plots tab above to identify "
                    "precisely which class pairs experience cross-classification, and whether errors are symmetric or "
                    "concentrated in specific categories."
                ),
            })

    else:
        r2 = metrics.get("R²")
        rmse = metrics.get("RMSE")
        mae = metrics.get("MAE")

        has_r2 = _is_valid_number(r2)
        has_rmse = _is_valid_number(rmse)
        has_mae = _is_valid_number(mae)

        if has_r2 and float(r2) < 0.30:
            if float(r2) < 0.0:
                recommendations.append({
                    "title": "Address Negative R² Score",
                    "advice": (
                        f"R² is negative ({float(r2):.4f}), indicating predictions deviate more than simply predicting "
                        "the mean target value. Check for extreme target outliers, uninformative features, or severe data scale mismatch."
                    ),
                })
            else:
                recommendations.append({
                    "title": "Investigate Low Target Variance Explained",
                    "advice": (
                        f"R² is {float(r2):.4f} ({float(r2):.1%} of variance explained). Current features have modest predictive power. "
                        "Consider gathering additional domain variables, combining existing features, or exploring non-linear interactions."
                    ),
                })
        elif has_r2 and float(r2) >= 0.70:
            recommendations.append({
                "title": "Inspect Residual Distribution",
                "advice": (
                    f"The model explains {float(r2):.1%} of target variance. Inspect the Predicted vs. Actual scatter plot "
                    "in Diagnostic Plots to verify that residuals are evenly scattered around the 45-degree diagonal without "
                    "systematic over-prediction at low values or under-prediction at high values."
                ),
            })

        if has_rmse and has_mae and float(mae) > 0:
            ratio = float(rmse) / float(mae)
            if ratio >= 1.35:
                recommendations.append({
                    "title": "Investigate Outlier Prediction Errors",
                    "advice": (
                        f"RMSE ({float(rmse):.2f}) is substantially higher than MAE ({float(mae):.2f}) by a factor of {ratio:.2f}. "
                        "Because RMSE squares errors, this divergence indicates that a small subset of test records suffered from "
                        "exceptionally large errors. Inspect the Predicted vs. Actual scatter plot to isolate these specific data points."
                    ),
                })
            else:
                recommendations.append({
                    "title": "Review Typical Prediction Error",
                    "advice": (
                        f"Predictions deviate by an average of ±{float(mae):.2f} target units (MAE), with relatively uniform error spread "
                        f"(RMSE = {float(rmse):.2f}). Verify whether this error margin satisfies domain accuracy requirements."
                    ),
                })

    if isinstance(feat_imp, pd.DataFrame) and not feat_imp.empty:
        try:
            top_row = feat_imp.sort_values("importance", ascending=False).iloc[0]
            top_feature = str(top_row["feature"])
            total_imp = float(feat_imp["importance"].sum())
            top_share = (float(top_row["importance"]) / total_imp) if total_imp > 0 else 0.0

            if top_share >= 0.45:
                recommendations.append({
                    "title": f"Verify Feature Integrity for '{top_feature}'",
                    "advice": (
                        f"The feature '{top_feature}' accounts for {top_share:.1%} of overall feature importance. "
                        "Verify that this column is legitimately available at prediction time in production and does not "
                        "contain future knowledge (target leakage)."
                    ),
                })
            else:
                recommendations.append({
                    "title": "Explore Additional Feature Engineering",
                    "advice": (
                        "Predictive influence is distributed across multiple features without a single dominant driver. "
                        "Domain-specific feature ratios or interaction terms may help strengthen signal further."
                    ),
                })
        except Exception:
            pass

    recommendations.append({
        "title": "Validate on Fresh Out-of-Sample Records",
        "advice": (
            f"Validation was performed on {n_test:,} held-out records. Before relying on model predictions in operational workflows, "
            "test the serialized `.joblib` pipeline on newly collected records from different time periods or sources to confirm stability."
        ),
    })

    return {
        "status": "success",
        "headline": f"Actionable recommendations based on validation diagnostics for **{best_model}**.",
        "recommendations": recommendations,
        "disclaimer": "These suggestions are analytical guidelines based on observed validation results, not definitive guarantees.",
    }


def recommend_next_steps_unsupervised(
    unsupervised_result: dict[str, Any] | None,
    selected_model: str | None = None,
) -> dict[str, Any]:
    """Generate concise, actionable next-step recommendations for unsupervised AutoML.

    Based strictly on calculated clustering metrics, anomaly counts, and PCA variance.
    Never invents uncalculated data properties or assumes a particular cluster count is correct.
    """
    if not unsupervised_result or not isinstance(unsupervised_result, dict):
        return {
            "status": "empty",
            "headline": "No unsupervised recommendations available.",
            "recommendations": [],
            "disclaimer": "Recommendations are analytical guidelines based on calculated diagnostics, not absolute guarantees.",
        }

    outputs = unsupervised_result.get("outputs") or {}
    recommendations: list[dict[str, str]] = []

    # 1. Clustering recommendations
    cluster_names = [n for n, o in outputs.items() if o.get("kind") == "clustering"]
    if cluster_names:
        c_models_with_sil = []
        for name in cluster_names:
            m = outputs[name].get("metrics") or {}
            sil = m.get("Silhouette score")
            if _is_valid_number(sil):
                c_models_with_sil.append((name, float(sil), m.get("Clusters found", 0), m.get("Noise points", 0)))

        c_models_with_sil.sort(key=lambda x: x[1], reverse=True)

        if c_models_with_sil:
            top_name, top_sil, n_clusters, noise = c_models_with_sil[0]
            if top_sil < 0.45:
                recommendations.append({
                    "title": "Review Cluster Separation and Count",
                    "advice": (
                        f"The highest Silhouette score achieved was {top_sil:.4f} ({top_name}, {n_clusters} clusters), "
                        "indicating weak separation and substantial overlap between groups. "
                        "Do not assume that this cluster count is definitively correct. Consider experimenting with different "
                        "cluster counts in Section 2, or verifying whether the underlying data behaves more like a continuous spectrum than discrete segments."
                    ),
                })
            else:
                recommendations.append({
                    "title": "Validate Cluster Profiles with Domain Experts",
                    "advice": (
                        f"**{top_name}** achieved a solid Silhouette score of {top_sil:.4f} across {n_clusters} clusters. "
                        "Inspect the assigned cluster preview table and column statistics to verify whether the grouped profiles "
                        "correspond to meaningful real-world cohorts or business categories."
                    ),
                })

            # Check cluster sizes if labels are available
            top_labels = outputs.get(top_name, {}).get("labels")
            if top_labels is not None:
                try:
                    s_series = pd.Series(top_labels)
                    valid_counts = s_series[s_series != -1].value_counts().sort_index()
                    if not valid_counts.empty:
                        sizes_str = ", ".join(f"Cluster {c}: {cnt:,}" for c, cnt in valid_counts.items())
                        recommendations.append({
                            "title": "Review Cluster Size Distribution",
                            "advice": (
                                f"Review the distribution of records across clusters ({sizes_str}). "
                                "Check whether clusters are balanced enough to be actionable, or if one group disproportionately absorbs most data."
                            ),
                        })
                except Exception:
                    pass

            if noise > 0:
                recommendations.append({
                    "title": "Review Noise and Unassigned Points",
                    "advice": (
                        f"DBSCAN flagged {noise:,} records as unclustered noise points. "
                        "If too many legitimate records are excluded, consider increasing the neighborhood radius (`eps`) "
                        "or lowering minimum samples in Configuration."
                    ),
                })
        else:
            recommendations.append({
                "title": "Evaluate Feature Scale and Informational Content",
                "advice": (
                    "None of the evaluated clustering algorithms produced a defined multi-cluster Silhouette score. "
                    "Review whether input features have sufficient variance or whether additional informative columns are needed."
                ),
            })

    # 2. Anomaly detection recommendations
    anomaly_names = [n for n, o in outputs.items() if o.get("kind") == "anomaly"]
    if anomaly_names:
        active_anom = selected_model if (selected_model and selected_model in anomaly_names) else anomaly_names[0]
        a_m = outputs[active_anom].get("metrics") or {}
        a_count = a_m.get("Anomaly count", 0)
        a_rate = a_m.get("Anomaly rate", 0.0)

        recommendations.append({
            "title": f"Investigate Flagged Outliers ({active_anom})",
            "advice": (
                f"**{active_anom}** flagged {a_count:,} records ({a_rate:.1%}) as statistical outliers. "
                "These flagged rows represent unusual positions in feature space, but are not necessarily errors, defects, or fraudulent entries. "
                "Review the flagged records table in Tab 2 with domain specialists to understand the real-world cause of the deviations."
            ),
        })

        if len(anomaly_names) > 1:
            recommendations.append({
                "title": "Cross-Reference Outliers Across Detectors",
                "advice": (
                    f"Multiple anomaly detectors were run ({', '.join(anomaly_names)}). "
                    "Cross-referencing which records are flagged concurrently by both Isolation Forest and Local Outlier Factor "
                    "can help isolate high-priority, multi-dimensional anomalies with greater confidence."
                ),
            })

    # 3. PCA recommendations
    if "PCA" in outputs:
        p_m = outputs["PCA"].get("metrics") or {}
        exp_var = p_m.get("Explained variance", 0.0)
        if _is_valid_number(exp_var):
            var_val = float(exp_var)
            if var_val >= 0.75:
                recommendations.append({
                    "title": "Leverage 2D Projection for Visual Analysis",
                    "advice": (
                        f"The 2D PCA projection retains {var_val:.1%} of overall variance, meaning the 2D scatter plot "
                        "is a highly faithful representation of multi-dimensional groupings and distance relationships."
                    ),
                })
            else:
                recommendations.append({
                    "title": "Account for Higher-Dimensional Information",
                    "advice": (
                        f"The 2D PCA projection captures {var_val:.1%} of total variance, leaving {1.0 - var_val:.1%} "
                        "in higher dimensions. While the scatter plot is helpful for visual exploration, do not base critical "
                        "decisions solely on 2D visual separation."
                    ),
                })

    headline = "Actionable suggestions based on diagnostic metrics and pattern structures observed in your data."

    return {
        "status": "success",
        "headline": headline,
        "recommendations": recommendations,
        "disclaimer": "These suggestions are analytical guidelines based on observed unsupervised metrics, not definitive guarantees.",
    }


def explain_limitations_supervised(result: dict[str, Any] | None) -> dict[str, Any]:
    """Explain model limitations, operational boundaries, and trust considerations for supervised AutoML.

    Covers classification accuracy limitations, regression out-of-sample caveats,
    evaluation setup constraints (only when evidence is available), and feature sensitivities.
    """
    if not result or not isinstance(result, dict) or "task" not in result:
        return {
            "status": "empty",
            "headline": "No supervised results are currently available to assess limitations.",
            "items": [],
            "summary": "Execute the tournament to evaluate model boundaries and trust considerations.",
        }

    task = result.get("task", "classification")
    metrics = result.get("metrics") or {}
    best_model = result.get("best_model", "Selected Model")
    n_train = result.get("n_train")
    n_test = result.get("n_test")
    y_test = result.get("y_test")
    feat_imp = result.get("feature_importance")

    items: list[dict[str, str]] = []

    if task == "classification":
        acc = metrics.get("Accuracy")
        f1 = metrics.get("F1 (weighted)")
        has_acc = _is_valid_number(acc)
        has_f1 = _is_valid_number(f1)

        imbalance_note = ""
        if y_test is not None and hasattr(y_test, "value_counts"):
            try:
                vc = y_test.value_counts()
                if len(vc) >= 2:
                    min_c = int(vc.min())
                    max_c = int(vc.max())
                    if max_c > 0 and (min_c / max_c) <= 0.40:
                        imbalance_note = f" (test class frequencies range from {min_c} to {max_c})"
            except Exception:
                pass

        acc_str = f"{float(acc):.1%}" if has_acc else "reported"
        f1_str = f", F1: {float(f1):.4f}" if has_f1 else ""
        items.append({
            "title": "High Accuracy Does Not Guarantee Reliability",
            "text": (
                f"Even with an accuracy of {acc_str}{f1_str}, high accuracy alone does not guarantee reliable real-world predictions, "
                f"especially when class distributions are imbalanced{imbalance_note}. A model can attain high overall accuracy by "
                "consistently predicting majority classes while failing to identify critical minority cases. Evaluate precision and "
                "recall on individual target classes before relying on automated predictions."
            ),
        })
    else:
        r2 = metrics.get("R²")
        mae = metrics.get("MAE")
        rmse = metrics.get("RMSE")
        has_r2 = _is_valid_number(r2)
        has_mae = _is_valid_number(mae)
        has_rmse = _is_valid_number(rmse)

        r2_str = f" (R² = {float(r2):.4f})" if has_r2 else ""
        err_str = f"an average error of ±{float(mae):.2f} target units (MAE)" if has_mae else "the reported test error"
        items.append({
            "title": "Metrics Reflect Evaluated Data, Not Future Guarantees",
            "text": (
                f"The reported metrics{r2_str} measure performance strictly on the evaluated test partition, showing {err_str}. "
                "They do not guarantee identical accuracy on future real-world data. Future observations may experience concept drift, "
                "seasonal variation, or external shifts that increase prediction error beyond what was observed during evaluation."
            ),
        })

        items.append({
            "title": "Extrapolation and Out-of-Bounds Inputs",
            "text": (
                "Regression algorithms cannot reliably predict values for inputs outside the numeric ranges observed during training. "
                "Predictions for extreme feature combinations or unprecedented scenarios should be treated with skepticism."
            ),
        })

        if has_rmse and has_mae and float(mae) > 0 and (float(rmse) / float(mae)) >= 1.35:
            items.append({
                "title": "Potential for Occasional Large Error Spikes",
                "text": (
                    f"RMSE ({float(rmse):.2f}) is substantially higher than MAE ({float(mae):.2f}). While typical errors remain near MAE, "
                    "a subset of test records suffered from disproportionately large errors, indicating vulnerability to outlier misses."
                ),
            })

    # Evaluation setup evidence
    if n_train and n_test:
        test_pct = (n_test / (n_train + n_test) * 100.0) if (n_train + n_test) > 0 else 0.0
        sample_warning = (
            f" With only {n_test:,} test instances, metric confidence intervals are wider and sensitive to small sample shifts."
            if n_test < 50
            else ""
        )
        items.append({
            "title": "Evaluation Setup & Sample Scope",
            "text": (
                f"Evaluation was performed using a held-out test split of {n_test:,} records ({test_pct:.0f}% of data) "
                f"after fitting on {n_train:,} training records.{sample_warning} These validation scores assume that future "
                "operational records will be generated by the exact same underlying mechanism as the training sample."
            ),
        })

    # Feature reliance evidence
    if isinstance(feat_imp, pd.DataFrame) and not feat_imp.empty:
        try:
            top_row = feat_imp.sort_values("importance", ascending=False).iloc[0]
            top_feature = str(top_row["feature"])
            total_imp = float(feat_imp["importance"].sum())
            top_pct = (float(top_row["importance"]) / total_imp * 100.0) if total_imp > 0 else 0.0
            if top_pct >= 45.0:
                items.append({
                    "title": f"Concentrated Feature Dependency ('{top_feature}')",
                    "text": (
                        f"The model derives {top_pct:.1f}% of its predictive importance from a single column: '{top_feature}'. "
                        "If this feature suffers from measurement drift, missing values, or latency in production, model accuracy "
                        "will degrade significantly."
                    ),
                })
        except Exception:
            pass

    headline = f"Key operational boundaries and trust considerations for **{best_model}**."
    summary = "Validation metrics are evidence of past fit on static sample data, not unconditional guarantees for future operations."

    return {
        "status": "success",
        "headline": headline,
        "items": items,
        "summary": summary,
    }


def explain_limitations_unsupervised(
    unsupervised_result: dict[str, Any] | None,
    selected_model: str | None = None,
) -> dict[str, Any]:
    """Explain unsupervised learning limitations, heuristic assumptions, and trust considerations.

    Covers geometric clustering assumptions, absence of ground-truth labels,
    anomaly false-alarm nuances, and PCA dimensionality information loss.
    """
    if not unsupervised_result or not isinstance(unsupervised_result, dict):
        return {
            "status": "empty",
            "headline": "No unsupervised results are currently available to assess limitations.",
            "items": [],
            "summary": "Run unsupervised analysis to evaluate structural boundaries and trust factors.",
        }

    outputs = unsupervised_result.get("outputs") or {}
    items: list[dict[str, str]] = []

    # 1. Clustering limitations
    cluster_names = [n for n, o in outputs.items() if o.get("kind") == "clustering"]
    if cluster_names:
        c_models = []
        for name in cluster_names:
            m = outputs[name].get("metrics") or {}
            sil = m.get("Silhouette score")
            if _is_valid_number(sil):
                c_models.append((name, float(sil), m.get("Clusters found", 0), m.get("Noise points", 0)))
        c_models.sort(key=lambda x: x[1], reverse=True)

        sil_detail = f" (e.g. Silhouette score of {c_models[0][1]:.4f} for {c_models[0][0]})" if c_models else ""
        items.append({
            "title": "Clustering Metrics Do Not Prove Real-World Meaning",
            "text": (
                f"Favorable clustering metrics{sil_detail} indicate that groupings are geometrically compact and separated "
                "under Euclidean distance metrics. However, mathematical separation does not prove that the discovered groups "
                "represent meaningful, valid, or useful categories in the real world. Discovered clusters must always be reviewed "
                "and validated by domain experts before guiding operational strategies."
            ),
        })

        items.append({
            "title": "Sensitivity to Scale and Distance Assumptions",
            "text": (
                "Clustering algorithms are sensitive to feature scales, normalizations, and geometric assumptions (e.g., spherical clusters in K-Means). "
                "Adding, removing, or rescaling input variables can substantially alter group assignments."
            ),
        })

    # 2. Anomaly detection limitations
    anomaly_names = [n for n, o in outputs.items() if o.get("kind") == "anomaly"]
    if anomaly_names:
        active_anom = selected_model if (selected_model and selected_model in anomaly_names) else anomaly_names[0]
        a_m = outputs[active_anom].get("metrics") or {}
        a_count = a_m.get("Anomaly count")
        a_rate = a_m.get("Anomaly rate")

        rate_detail = f" ({a_count:,} records or {a_rate:.1%})" if (_is_valid_number(a_count) and _is_valid_number(a_rate)) else ""
        items.append({
            "title": "Flagged Records Are Outliers, Not Confirmed Errors or Fraud",
            "text": (
                f"Records flagged as anomalous by {active_anom}{rate_detail} are statistically unusual observations in feature space. "
                "They are not automatically errors, defects, fraud, or malicious incidents. Many flagged points represent legitimate, "
                "infrequent business activities or edge cases. Domain specialists must investigate flagged rows before taking punitive "
                "or automated actions."
            ),
        })

        items.append({
            "title": "Anomaly Rates Are Driven by Sensitivity Assumptions",
            "text": (
                "Unsupervised anomaly detectors lack verified labels to benchmark false-positive versus false-negative trade-offs. "
                "The number of flagged records is largely governed by the user-configured contamination setting rather than an "
                "empirically measured failure rate."
            ),
        })

    # 3. PCA limitations
    if "PCA" in outputs:
        p_m = outputs["PCA"].get("metrics") or {}
        exp_var = p_m.get("Explained variance")
        if _is_valid_number(exp_var):
            var_val = float(exp_var)
            lost_val = max(0.0, 1.0 - var_val)
            items.append({
                "title": "Dimensionality Reduction Discards Information",
                "text": (
                    f"PCA compresses multi-dimensional relationships into 2 components, retaining {var_val:.1%} of overall variance. "
                    f"The remaining {lost_val:.1%} of variance is discarded from the 2D projection. Visual closeness or distance "
                    "in 2D space does not capture all nuances present in the full feature space."
                ),
            })
        else:
            items.append({
                "title": "Dimensionality Reduction Discards Information",
                "text": (
                    "Projecting high-dimensional data down to 2 principal components inevitably discards variance and higher-order "
                    "relationships. Visual proximity in the 2D plot should be treated as an exploratory aid rather than complete truth."
                ),
            })

    # General absence of ground truth
    items.append({
        "title": "Absence of Ground-Truth Benchmarks",
        "text": (
            "Because unsupervised methods operate without target labels, there is no objective 'accuracy' score to guarantee correctness. "
            "All findings represent mathematical patterns in the uploaded sample and should be treated as hypotheses for investigation."
        ),
    })

    headline = "Important heuristic boundaries, assumptions, and trust guidelines for unsupervised findings."
    summary = "Unsupervised patterns reflect geometric and statistical density properties, not verified real-world facts."

    return {
        "status": "success",
        "headline": headline,
        "items": items,
        "summary": summary,
    }


def build_model_comparison_arena(result: dict[str, Any] | None) -> dict[str, Any]:
    """Prepare structured data for the Model Comparison Arena.

    Evaluates and formats all candidate algorithms into one clear table,
    identifies the recommended model with simple-language rationale,
    provides objective metric guidance, and safely handles missing or failed models.
    """
    if not result or not isinstance(result, dict) or "task" not in result:
        return {
            "status": "empty",
            "task": "unknown",
            "best_model": "None",
            "table": pd.DataFrame(),
            "recommendation_headline": "No evaluation results are currently loaded.",
            "recommendation_explanation": "Execute the tournament to populate the Model Comparison Arena.",
            "metric_guidance": "",
            "failed_models": [],
        }

    task = result.get("task", "classification")
    best_model = result.get("best_model", "Selected Model")
    scoring = result.get("scoring", "objective")
    leaderboard = result.get("leaderboard") or []
    metrics = result.get("metrics") or {}

    # Scoring human-readable description
    scoring_names = {
        "f1_weighted": "Weighted F1-score (balancing precision and recall across all classes)",
        "roc_auc": "ROC AUC (area under the ROC curve across decision thresholds)",
        "accuracy": "Accuracy (proportion of correct classifications)",
        "r2": "R² (proportion of target variance explained by predictions)",
        "neg_mean_squared_error": "Negative Mean Squared Error",
        "neg_mean_absolute_error": "Negative Mean Absolute Error",
    }
    scoring_desc = scoring_names.get(scoring, f"{scoring} score")

    # 1. Identify winner details & runner up
    valid_rows = [r for r in leaderboard if _is_valid_number(r.get("cv_score"))]
    valid_rows.sort(key=lambda x: float(x.get("cv_score", 0.0)), reverse=True)
    failed_rows = [r for r in leaderboard if not _is_valid_number(r.get("cv_score"))]

    best_row = next((r for r in valid_rows if r.get("model") == best_model), valid_rows[0] if valid_rows else None)
    best_score = float(best_row["cv_score"]) if best_row and _is_valid_number(best_row.get("cv_score")) else None
    best_std = float(best_row["cv_std"]) if best_row and _is_valid_number(best_row.get("cv_std")) else None

    runner_up_row = None
    if len(valid_rows) > 1:
        runner_up_row = valid_rows[1] if valid_rows[0].get("model") == best_model else valid_rows[0]

    # Simple language explanation of why the recommended model was selected
    score_str = f"{best_score:.4f}" if best_score is not None else "highest"
    std_str = f" (±{best_std:.4f})" if best_std is not None else ""

    reasons = [
        f"**{best_model}** is the recommended model because it achieved the strongest generalization score "
        f"on cross-validation ({score_str}{std_str}), evaluated using {scoring_desc}."
    ]

    if runner_up_row and best_score is not None and _is_valid_number(runner_up_row.get("cv_score")):
        r_score = float(runner_up_row["cv_score"])
        diff = best_score - r_score
        r_name = runner_up_row.get("model", "the runner-up")
        if abs(diff) < 0.005:
            reasons.append(
                f"It won a very close contest against **{r_name}** ({r_score:.4f}), edging it out by {diff:+.4f} points."
            )
        else:
            reasons.append(
                f"It demonstrated a clear lead over the runner-up **{r_name}** ({r_score:.4f}), outperforming it by {diff:.4f} points."
            )

    # Architectural rationale from _model_reason
    arch_reason = _model_reason(best_model, scoring, best_score if best_score is not None else 0.0)
    reasons.append(arch_reason)
    reasons.append(
        "Cross-validation was used to ensure the recommendation reflects genuine out-of-sample performance rather than memorization of a single data split."
    )

    recommendation_explanation = " ".join(reasons)

    # 2. Build one unified comparison table
    table_rows = []
    for r in leaderboard:
        m_name = r.get("model", "Unknown")
        is_best = (m_name == best_model)
        has_score = _is_valid_number(r.get("cv_score"))

        status = "Recommended" if is_best else ("Evaluated" if has_score else "Failed")

        cv_val = float(r["cv_score"]) if has_score else np.nan
        std_val = float(r["cv_std"]) if _is_valid_number(r.get("cv_std")) else np.nan
        sec_val = float(r["seconds"]) if _is_valid_number(r.get("seconds")) else np.nan

        row_dict: dict[str, Any] = {
            "Algorithm": m_name,
            "Status": status,
        }

        if task == "classification":
            # Accuracy
            acc = r.get("Accuracy")
            if not _is_valid_number(acc) and is_best and _is_valid_number(metrics.get("Accuracy")):
                acc = metrics.get("Accuracy")
            row_dict["Accuracy"] = round(float(acc), 4) if _is_valid_number(acc) else np.nan

            # Precision
            prec = r.get("Precision")
            if not _is_valid_number(prec) and is_best and _is_valid_number(metrics.get("Precision (weighted)")):
                prec = metrics.get("Precision (weighted)")
            row_dict["Precision"] = round(float(prec), 4) if _is_valid_number(prec) else np.nan

            # Recall
            rec = r.get("Recall")
            if not _is_valid_number(rec) and is_best and _is_valid_number(metrics.get("Recall (weighted)")):
                rec = metrics.get("Recall (weighted)")
            row_dict["Recall"] = round(float(rec), 4) if _is_valid_number(rec) else np.nan

            # F1-score
            f1 = r.get("F1-score") or r.get("F1")
            if not _is_valid_number(f1) and is_best and _is_valid_number(metrics.get("F1 (weighted)")):
                f1 = metrics.get("F1 (weighted)")
            row_dict["F1-score"] = round(float(f1), 4) if _is_valid_number(f1) else np.nan

        else:
            # MAE
            mae = r.get("MAE")
            if not _is_valid_number(mae) and is_best and _is_valid_number(metrics.get("MAE")):
                mae = metrics.get("MAE")
            row_dict["MAE"] = round(float(mae), 4) if _is_valid_number(mae) else np.nan

            # RMSE
            rmse = r.get("RMSE")
            if not _is_valid_number(rmse) and is_best and _is_valid_number(metrics.get("RMSE")):
                rmse = metrics.get("RMSE")
            row_dict["RMSE"] = round(float(rmse), 4) if _is_valid_number(rmse) else np.nan

            # R²
            r2 = r.get("R²") or r.get("R2")
            if not _is_valid_number(r2) and is_best and _is_valid_number(metrics.get("R²")):
                r2 = metrics.get("R²")
            row_dict["R²"] = round(float(r2), 4) if _is_valid_number(r2) else np.nan

        row_dict[f"CV Score ({scoring})"] = round(cv_val, 4) if _is_valid_number(cv_val) else np.nan
        row_dict["CV Std (±)"] = round(std_val, 4) if _is_valid_number(std_val) else np.nan
        row_dict["Time (s)"] = round(sec_val, 2) if _is_valid_number(sec_val) else np.nan

        table_rows.append(row_dict)

    comparison_df = pd.DataFrame(table_rows)

    # 3. Metric guidance note
    if task == "classification":
        metric_guidance = (
            "Metric Guidance: Accuracy measures the overall fraction of correct predictions across all classes. "
            "Precision measures the proportion of positive identifications that were actually correct (limiting false alarms). "
            "Recall measures the proportion of actual positives that were correctly identified (limiting missed cases). "
            "F1-score balances Precision and Recall, especially on uneven class distributions. "
            "No single metric is universally best; choose the metric that best aligns with whether false positives or false negatives carry greater real-world cost."
        )
    else:
        metric_guidance = (
            "Metric Guidance: MAE (Mean Absolute Error) measures average prediction error magnitude in target units. "
            "RMSE (Root Mean Squared Error) penalizes large outlier errors more heavily through squaring. "
            "R² (Coefficient of Determination) reflects the proportion of target variance captured by the model. "
            "No single metric is universally best; select based on whether occasional large prediction errors are disproportionately harmful."
        )

    # 4. Failed models list for safe reporting
    failed_models = []
    for fr in failed_rows:
        failed_models.append({
            "model": fr.get("model", "Unknown Model"),
            "error": fr.get("error", "Failed during cross-validation"),
        })

    return {
        "status": "success",
        "task": task,
        "best_model": best_model,
        "scoring": scoring,
        "scoring_desc": scoring_desc,
        "table": comparison_df,
        "recommendation_headline": f"Recommended Model: **{best_model}**",
        "recommendation_explanation": recommendation_explanation,
        "metric_guidance": metric_guidance,
        "failed_models": failed_models,
    }






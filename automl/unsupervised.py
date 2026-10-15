"""Unsupervised AutoML for clustering, anomaly detection and dimensionality reduction."""
from __future__ import annotations

import time
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score

RANDOM_STATE = 42


def prepare_features(df: pd.DataFrame, features: list[str]):
    """Create a fitted preprocessing pipeline for mixed numeric/categorical data."""
    if not features:
        raise ValueError("Select at least one input feature.")
    X = df[features].copy()
    if len(X) < 3:
        raise ValueError("At least 3 rows are required for unsupervised analysis.")
    # Drop columns that contain no information.
    usable = [c for c in features if X[c].nunique(dropna=True) > 1]
    if not usable:
        raise ValueError("All selected columns are constant or empty. Choose informative features.")
    X = X[usable]
    numeric = X.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical = [c for c in X.columns if c not in numeric]
    transformers = []
    if numeric:
        transformers.append(("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), numeric))
    if categorical:
        transformers.append(("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
            ("scaler", StandardScaler()),
        ]), categorical))
    preprocessor = ColumnTransformer(transformers, remainder="drop")
    matrix = preprocessor.fit_transform(X)
    matrix = np.asarray(matrix, dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("The selected features contain values that could not be converted to finite numbers.")
    return X, matrix, preprocessor, usable


def _cluster_metrics(matrix, labels):
    labels = np.asarray(labels)
    unique = set(labels.tolist())
    # DBSCAN's -1 label represents noise; exclude it from cluster-quality metrics.
    valid = labels != -1
    clean_labels = labels[valid]
    clean_matrix = matrix[valid]
    n_clusters = len(set(clean_labels.tolist()))
    metrics = {"Clusters found": n_clusters, "Noise points": int(np.sum(labels == -1))}
    if n_clusters >= 2 and n_clusters < len(clean_labels) and len(clean_labels) >= 3:
        # Silhouette sampling keeps large uploads responsive.
        sample_size = min(5000, len(clean_labels))
        try:
            metrics["Silhouette score"] = float(silhouette_score(
                clean_matrix, clean_labels, sample_size=sample_size, random_state=RANDOM_STATE
            ))
            metrics["Davies-Bouldin index"] = float(davies_bouldin_score(clean_matrix, clean_labels))
            metrics["Calinski-Harabasz score"] = float(calinski_harabasz_score(clean_matrix, clean_labels))
        except Exception:
            pass
    else:
        metrics["Silhouette score"] = float("nan")
    return metrics


def run_unsupervised(df: pd.DataFrame, features: list[str], methods: list[str],
                     n_clusters: int = 3, contamination: float = 0.05,
                     eps: float = 0.5, min_samples: int = 5, progress=None) -> dict:
    """Fit selected unsupervised families and return outputs plus comparable diagnostics."""
    def report(message, fraction):
        if progress:
            progress(message, fraction)

    report("Preparing features", 0.05)
    X, matrix, preprocessor, usable = prepare_features(df, features)
    n_rows = len(X)
    if n_rows > 100_000:
        raise ValueError("This local version is designed for up to 100,000 rows per run. Please sample the dataset first.")
    leaderboard, outputs, fitted = [], {}, {}
    tasks = [m for m in methods if m in {"Clustering", "Anomaly detection", "Dimensionality reduction"}]
    if not tasks:
        raise ValueError("Choose at least one analysis type.")
    total_steps = 0
    if "Clustering" in tasks:
        total_steps += 4
    if "Anomaly detection" in tasks:
        total_steps += 3
    if "Dimensionality reduction" in tasks:
        total_steps += 1
    step = 0

    def tick(name):
        nonlocal step
        step += 1
        report(f"Running {name}", 0.1 + 0.8 * step / max(1, total_steps))

    if "Clustering" in tasks:
        clusterers = {
            "K-Means": KMeans(n_clusters=min(n_clusters, n_rows - 1), n_init=10, random_state=RANDOM_STATE),
            "Agglomerative": AgglomerativeClustering(n_clusters=min(n_clusters, n_rows - 1)),
            "DBSCAN": DBSCAN(eps=eps, min_samples=min_samples),
            "Gaussian Mixture": GaussianMixture(n_components=min(n_clusters, n_rows - 1), random_state=RANDOM_STATE),
        }
        for name, model in clusterers.items():
            tick(name)
            start = time.time()
            try:
                if name == "Gaussian Mixture":
                    labels = model.fit_predict(matrix)
                else:
                    labels = model.fit_predict(matrix)
                metrics = _cluster_metrics(matrix, labels)
                outputs[name] = {"kind": "clustering", "labels": labels, "metrics": metrics}
                fitted[name] = {"preprocessor": preprocessor, "model": model, "features": usable, "kind": "clustering"}
                leaderboard.append({"Analysis": "Clustering", "Algorithm": name,
                                    "Silhouette score": metrics.get("Silhouette score", np.nan),
                                    "Clusters": metrics.get("Clusters found", np.nan),
                                    "Noise points": metrics.get("Noise points", 0),
                                    "Seconds": round(time.time() - start, 2),
                                    "Selection note": "Higher silhouette is generally better when defined"})
            except Exception as exc:
                leaderboard.append({"Analysis": "Clustering", "Algorithm": name, "Silhouette score": np.nan,
                                    "Clusters": np.nan, "Noise points": np.nan,
                                    "Seconds": round(time.time() - start, 2), "Selection note": f"Failed: {str(exc)[:120]}"})

    if "Anomaly detection" in tasks:
        anomaly_models = {
            "Isolation Forest": IsolationForest(contamination=contamination, random_state=RANDOM_STATE, n_estimators=150),
            "Local Outlier Factor": LocalOutlierFactor(n_neighbors=min(20, max(2, n_rows - 1)), contamination=contamination),
            "One-Class SVM": None,
        }
        # Import lazily alongside other sklearn estimators.
        from sklearn.svm import OneClassSVM
        anomaly_models["One-Class SVM"] = OneClassSVM(nu=contamination, gamma="scale")
        for name, model in anomaly_models.items():
            tick(name)
            start = time.time()
            try:
                labels = model.fit_predict(matrix) if name == "Local Outlier Factor" else model.fit(matrix).predict(matrix)
                anomaly_mask = np.asarray(labels) == -1
                fraction = float(anomaly_mask.mean())
                outputs[name] = {"kind": "anomaly", "labels": np.asarray(labels), "anomaly_mask": anomaly_mask,
                                 "metrics": {"Anomaly count": int(anomaly_mask.sum()), "Anomaly rate": fraction}}
                fitted[name] = {"preprocessor": preprocessor, "model": model, "features": usable, "kind": "anomaly"}
                leaderboard.append({"Analysis": "Anomaly detection", "Algorithm": name,
                                    "Silhouette score": np.nan, "Clusters": np.nan, "Noise points": int(anomaly_mask.sum()),
                                    "Seconds": round(time.time() - start, 2),
                                    "Selection note": f"Flagged {fraction:.1%}; rate is not an accuracy score"})
            except Exception as exc:
                leaderboard.append({"Analysis": "Anomaly detection", "Algorithm": name, "Silhouette score": np.nan,
                                    "Clusters": np.nan, "Noise points": np.nan,
                                    "Seconds": round(time.time() - start, 2), "Selection note": f"Failed: {str(exc)[:120]}"})

    if "Dimensionality reduction" in tasks:
        tick("PCA")
        n_components = min(2, matrix.shape[1], matrix.shape[0])
        pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
        transformed = pca.fit_transform(matrix)
        coords = pd.DataFrame(transformed, columns=[f"PC{i+1}" for i in range(transformed.shape[1])], index=X.index)
        outputs["PCA"] = {"kind": "pca", "coordinates": coords, "metrics": {
            "Components": n_components, "Explained variance": float(pca.explained_variance_ratio_.sum()),
            "PC1 variance": float(pca.explained_variance_ratio_[0]),
        }}
        fitted["PCA"] = {"preprocessor": preprocessor, "model": pca, "features": usable, "kind": "pca"}
        leaderboard.append({"Analysis": "Dimensionality reduction", "Algorithm": "PCA",
                            "Silhouette score": np.nan, "Clusters": np.nan, "Noise points": np.nan,
                            "Seconds": 0.0, "Selection note": f"Retained {pca.explained_variance_ratio_.sum():.1%} of variance in {n_components} component(s)"})

    report("Preparing result tables", 0.98)
    # Ensure 2D projection coordinates and PCA model are always available for visualization
    n_vis_components = min(2, matrix.shape[1], matrix.shape[0])
    if "PCA" in outputs and "coordinates" in outputs["PCA"]:
        coords_2d = outputs["PCA"]["coordinates"]
        pca_2d = fitted["PCA"]["model"]
    elif n_vis_components >= 1:
        pca_2d = PCA(n_components=n_vis_components, random_state=RANDOM_STATE)
        transformed_2d = pca_2d.fit_transform(matrix)
        coords_2d = pd.DataFrame(
            transformed_2d,
            columns=[f"PC{i+1}" for i in range(transformed_2d.shape[1])],
            index=X.index,
        )
    else:
        pca_2d = None
        coords_2d = pd.DataFrame(index=X.index)

    rows = []
    for name, output in outputs.items():
        if output["kind"] in {"clustering", "anomaly"}:
            rows.append((name, output["labels"]))
    results = df.copy()
    for name, labels in rows:
        col = f"{name} result".replace(" ", "_")
        results[col] = np.where(labels == -1, "Noise / anomaly", labels.astype(str)) if name in {"DBSCAN", "Isolation Forest", "Local Outlier Factor", "One-Class SVM"} else labels
    if "PCA" in outputs:
        for col in outputs["PCA"]["coordinates"].columns:
            results[col] = outputs["PCA"]["coordinates"][col].values
    report("Done", 1.0)
    return {
        "outputs": outputs,
        "fitted": fitted,
        "leaderboard": pd.DataFrame(leaderboard),
        "results": results,
        "features": usable,
        "preprocessor": preprocessor,
        "matrix": matrix,
        "coords_2d": coords_2d,
        "pca_2d": pca_2d,
        "n_rows": n_rows,
    }


def get_2d_projection(unsup_result: dict, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame | None, Any | None, bool]:
    """Retrieve or compute 2D visual projection coordinates and PCA model."""
    if not unsup_result or not isinstance(unsup_result, dict):
        return None, None, False

    coords = unsup_result.get("coords_2d")
    pca_model = unsup_result.get("pca_2d")
    if coords is not None and isinstance(coords, pd.DataFrame) and len(coords.columns) >= 2:
        return coords, pca_model, True

    if "PCA" in unsup_result.get("outputs", {}):
        coords = unsup_result["outputs"]["PCA"].get("coordinates")
        pca_model = unsup_result.get("fitted", {}).get("PCA", {}).get("model")
        if coords is not None and isinstance(coords, pd.DataFrame) and len(coords.columns) >= 2:
            return coords, pca_model, True

    matrix = unsup_result.get("matrix")
    if matrix is not None and len(matrix) > 0 and matrix.shape[1] >= 2:
        pca = PCA(n_components=2, random_state=RANDOM_STATE)
        tr = pca.fit_transform(matrix)
        coords = pd.DataFrame(tr, columns=["PC1", "PC2"], index=df.index if df is not None else None)
        return coords, pca, True

    return coords, pca_model, False

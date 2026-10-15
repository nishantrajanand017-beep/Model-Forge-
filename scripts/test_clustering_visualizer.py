"""Focused test suite for the Clustering Visualizer: cluster labels, counts, noise handling,
PCA projection, centroids, explanations, and plot generation.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from automl.unsupervised import run_unsupervised, get_2d_projection
from automl import plots
from automl.explain import explain_clustering_visualizer


def test_clustering_algorithms_labels_and_counts():
    print("Testing clustering algorithms (K-Means, Agglomerative, DBSCAN, Gaussian Mixture)...")
    np.random.seed(42)
    # 3 distinct blobs in 4-dimensional space
    n_per_blob = 40
    blob1 = np.random.normal(loc=[-5, -5, 0, 1], scale=0.5, size=(n_per_blob, 4))
    blob2 = np.random.normal(loc=[5, 5, 2, -1], scale=0.5, size=(n_per_blob, 4))
    blob3 = np.random.normal(loc=[0, 8, -3, 2], scale=0.5, size=(n_per_blob, 4))
    data = np.vstack([blob1, blob2, blob3])
    
    df = pd.DataFrame(data, columns=["feat_w", "feat_x", "feat_y", "feat_z"])
    features = list(df.columns)

    res = run_unsupervised(
        df, features,
        methods=["Clustering"],
        n_clusters=3,
        eps=1.2,
        min_samples=4,
    )

    outputs = res["outputs"]
    assert "K-Means" in outputs, "K-Means should be present in outputs"
    assert "Agglomerative" in outputs, "Agglomerative should be present in outputs"
    assert "DBSCAN" in outputs, "DBSCAN should be present in outputs"
    assert "Gaussian Mixture" in outputs, "Gaussian Mixture should be present in outputs"

    # Check 2D projection
    coords_2d, pca_model, used_pca = get_2d_projection(res, df)
    assert coords_2d is not None, "2D coordinates should be generated"
    assert len(coords_2d) == len(df), "Coordinates should match row count"
    assert len(coords_2d.columns) == 2, "Coordinates should be 2-dimensional"
    assert used_pca is True, "PCA should be used for 4D input features"

    # Check K-Means labels and centroids
    km_out = outputs["K-Means"]
    km_labels = km_out["labels"]
    assert len(km_labels) == len(df)
    unique_km = set(km_labels)
    assert len(unique_km) == 3, f"Expected 3 K-Means clusters, got {len(unique_km)}"

    km_model = res["fitted"]["K-Means"]["model"]
    assert hasattr(km_model, "cluster_centers_")
    assert km_model.cluster_centers_.shape == (3, 4), "Centroids should be in 4D feature space"
    centroids_2d = pca_model.transform(km_model.cluster_centers_)
    assert centroids_2d.shape == (3, 2), "Centroids should project into 2D visual space"

    # Test plot generation for K-Means with centroids
    fig_km = plots.plot_clustering(coords_2d, km_labels, centroids_2d=centroids_2d, title="K-Means Test")
    assert fig_km is not None
    print("  -> Passed K-Means labels, centroids, and plot.")


def test_dbscan_noise_handling():
    print("Testing DBSCAN noise points handling...")
    np.random.seed(42)
    # Dense cluster + sparse outlier points
    dense = np.random.normal(loc=0.0, scale=0.2, size=(80, 3))
    outliers = np.array([
        [15.0, 15.0, 15.0],
        [-18.0, -18.0, -18.0],
        [25.0, -25.0, 20.0],
    ])
    data = np.vstack([dense, outliers])
    df = pd.DataFrame(data, columns=["a", "b", "c"])

    res = run_unsupervised(
        df, list(df.columns),
        methods=["Clustering"],
        eps=0.8,
        min_samples=5,
    )

    db_out = res["outputs"]["DBSCAN"]
    labels = db_out["labels"]
    noise_count = int(np.sum(labels == -1))
    assert noise_count > 0, "DBSCAN should flag outliers with label -1"

    coords_2d, _, used_pca = get_2d_projection(res, df)

    # Test explanation of DBSCAN with noise
    exp = explain_clustering_visualizer(
        "DBSCAN",
        labels,
        n_features=3,
        used_pca=used_pca,
        has_centroids=False,
    )

    assert exp["noise_count"] == noise_count
    assert exp["noise_explanation"] is not None
    assert "Noise Points Detected" in exp["noise_explanation"]
    
    # Check breakdown table
    breakdown = exp["breakdown_df"]
    noise_row = breakdown[breakdown["Cluster"] == "Noise / Outlier"]
    assert not noise_row.empty, "Breakdown table should explicitly label 'Noise / Outlier'"
    assert noise_row.iloc[0]["Classification"] == "Noise Point (Unclustered)"
    assert noise_row.iloc[0]["Data Points (N)"] == noise_count

    # Test plot with noise points
    fig_db = plots.plot_clustering(coords_2d, labels, title="DBSCAN Noise Test")
    assert fig_db is not None
    print("  -> Passed DBSCAN noise points handling and explanation.")


def test_clustering_explanations():
    print("Testing beginner explanations, centroid explanations, and PCA axis notes...")
    labels = np.array([0, 0, 1, 1, 2, 2])
    
    # K-Means explanation
    exp_km = explain_clustering_visualizer(
        "K-Means",
        labels,
        n_features=5,
        used_pca=True,
        has_centroids=True,
    )
    assert "Clustering is an unsupervised machine learning method" in exp_km["clustering_concept"]
    assert exp_km["centroid_explanation"] is not None
    assert "What is a Centroid?" in exp_km["centroid_explanation"]
    assert "About the 2D Visualization (PCA Projection)" in exp_km["pca_axes_explanation"]
    assert "combinations of original features" in exp_km["pca_axes_explanation"].lower() or "weighted blend" in exp_km["pca_axes_explanation"].lower()

    # Agglomerative (no centroids)
    exp_agg = explain_clustering_visualizer(
        "Agglomerative",
        labels,
        n_features=2,
        used_pca=False,
        has_centroids=False,
    )
    assert exp_agg["centroid_explanation"] is None
    assert "without dimensional reduction" in exp_agg["pca_axes_explanation"]

    # Empty labels / safe fallback handling
    exp_empty = explain_clustering_visualizer("K-Means", None, n_features=3)
    assert exp_empty["total_points"] == 0
    assert exp_empty["breakdown_df"].empty
    print("  -> Passed clustering explanations and fallback handling.")


def test_plot_clustering_edge_cases():
    print("Testing plot_clustering edge cases...")
    # None coordinates
    fig1 = plots.plot_clustering(None, np.array([]))
    assert fig1 is not None

    # Single cluster
    coords = pd.DataFrame({"PC1": [1.0, 2.0, 3.0], "PC2": [4.0, 5.0, 6.0]})
    fig2 = plots.plot_clustering(coords, np.array([0, 0, 0]))
    assert fig2 is not None

    # All noise points
    fig3 = plots.plot_clustering(coords, np.array([-1, -1, -1]))
    assert fig3 is not None
    print("  -> Passed plot_clustering edge cases.")


if __name__ == "__main__":
    test_clustering_algorithms_labels_and_counts()
    test_dbscan_noise_handling()
    test_clustering_explanations()
    test_plot_clustering_edge_cases()
    print("\nALL CLUSTERING VISUALIZER TESTS PASSED SUCCESSFULLY!")

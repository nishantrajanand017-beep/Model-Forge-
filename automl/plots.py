"""Matplotlib figures for the results view. Uses the headless Agg backend with a monochrome theme."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import ConfusionMatrixDisplay  # noqa: E402

PRIMARY_BLACK = "#18181B"
CHARCOAL = "#27272A"
MID_GREY = "#71717A"
LIGHT_BORDER = "#E4E4E7"
GRID_GREY = "#F4F4F5"


def _style_axis(ax, title: str | None = None, xlabel: str | None = None, ylabel: str | None = None):
    """Apply consistent, professional monochrome styling to an axis."""
    ax.set_facecolor("#FFFFFF")
    ax.figure.patch.set_facecolor("#FFFFFF")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(LIGHT_BORDER)
    ax.spines["bottom"].set_color(LIGHT_BORDER)
    ax.spines["left"].set_linewidth(1.0)
    ax.spines["bottom"].set_linewidth(1.0)
    ax.tick_params(colors=CHARCOAL, labelsize=8.5, width=0.8, length=3)
    ax.grid(True, linestyle="--", linewidth=0.65, color=GRID_GREY, alpha=1.0)
    if title:
        ax.set_title(title, color=PRIMARY_BLACK, fontsize=10.5, fontweight="600", pad=10)
    if xlabel:
        ax.set_xlabel(xlabel, color=MID_GREY, fontsize=9, labelpad=6)
    if ylabel:
        ax.set_ylabel(ylabel, color=MID_GREY, fontsize=9, labelpad=6)


def plot_target_distribution(y, task: str):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    if task == "classification":
        vc = y.value_counts()
        title = "Target class distribution"
        if len(vc) > 20:
            vc = vc.head(20)
            title += " (top 20)"
        bars = ax.bar(vc.index.astype(str), vc.values, color=PRIMARY_BLACK, edgecolor=PRIMARY_BLACK, width=0.55)
        _style_axis(ax, title=title, ylabel="Count")
        ax.tick_params(axis="x", rotation=35)
    else:
        ax.hist(y.dropna(), bins=25, color=CHARCOAL, edgecolor="#FFFFFF", linewidth=0.8)
        _style_axis(ax, title="Target distribution", xlabel="Value", ylabel="Count")
    fig.tight_layout()
    return fig


def plot_confusion(cm, labels):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(ax=ax, cmap="Greys", colorbar=False)
    _style_axis(ax, title="Confusion matrix", xlabel="Predicted label", ylabel="True label")
    fig.tight_layout()
    return fig


def plot_pred_vs_actual(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(5, 3.8))
    ax.scatter(y_true, y_pred, alpha=0.55, edgecolor=PRIMARY_BLACK, facecolor=CHARCOAL, s=28, linewidth=0.6)
    lo = float(min(np.min(y_true), np.min(y_pred)))
    hi = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([lo, hi], [lo, hi], linestyle="--", color=MID_GREY, linewidth=1.2, label="Perfect fit")
    _style_axis(ax, title="Predicted vs. Actual", xlabel="Actual", ylabel="Predicted")
    fig.tight_layout()
    return fig


def plot_importance(imp_df, max_n: int | None = None, title: str = "Feature Importance Ranking"):
    """Horizontal bar chart ranking input features by importance with monochrome theme."""
    if imp_df is None or not isinstance(imp_df, pd.DataFrame) or imp_df.empty:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(
            0.5, 0.5, "No feature importance data available",
            ha="center", va="center", color=MID_GREY, fontsize=9.5,
            transform=ax.transAxes
        )
        _style_axis(ax, title=title)
        fig.tight_layout()
        return fig

    # Take top max_n features if requested
    df = imp_df.sort_values("importance", ascending=False).copy()
    if max_n is not None and max_n > 0:
        df = df.head(max_n)

    # For horizontal barh, sort ascending so the top feature is plotted at the top
    df = df.sort_values("importance", ascending=True).reset_index(drop=True)

    n_bars = len(df)
    fig_height = max(3.0, 0.34 * n_bars + 0.6)
    fig, ax = plt.subplots(figsize=(6.5, fig_height))

    y_pos = np.arange(n_bars)
    importances = df["importance"].to_numpy(dtype=float)
    features = [str(f) for f in df["feature"]]

    bars = ax.barh(y_pos, importances, color=PRIMARY_BLACK, height=0.58, edgecolor=PRIMARY_BLACK)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features, fontsize=9, color=CHARCOAL)

    # Annotate bar values cleanly
    max_val = float(np.max(importances)) if len(importances) > 0 else 1.0
    total_val = float(np.sum(importances)) if len(importances) > 0 else 0.0

    offset = max(max_val * 0.015, 0.002) if max_val > 0 else 0.01
    for bar in bars:
        w = bar.get_width()
        pct_str = f" ({w / total_val * 100:.1f}%)" if total_val > 0 else ""
        ax.text(
            w + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{w:.3f}{pct_str}",
            va="center",
            ha="left",
            fontsize=8,
            color=CHARCOAL,
            fontweight="500",
        )

    # Ensure annotations fit comfortably
    ax.set_xlim(0, max_val * 1.25 if max_val > 0 else 1.0)
    _style_axis(ax, title=title, xlabel="Relative Importance Score (Share %)")
    fig.tight_layout()
    return fig


def plot_pca(coords: pd.DataFrame, labels: np.ndarray | None = None, title: str = "PCA Projection (PC1 vs. PC2)"):
    """Monochrome 2D PCA scatter plot with distinct markers for groups."""
    fig, ax = plt.subplots(figsize=(6, 4))
    if len(coords.columns) >= 2:
        x = coords["PC1"]
        y = coords["PC2"]
        if labels is not None:
            labels_arr = np.asarray(labels)
            unique_labels = sorted(list(set(labels_arr.tolist())))
            markers = ["o", "s", "^", "D", "v", "X", "P", "*"]
            shades = [PRIMARY_BLACK, CHARCOAL, MID_GREY, "#A1A1AA", "#D4D4D8", "#52525B", "#3F3F46", "#27272A"]
            for idx, label in enumerate(unique_labels):
                mask = labels_arr == label
                marker = markers[idx % len(markers)]
                color = shades[idx % len(shades)]
                lbl_name = f"Cluster {label}" if str(label).isdigit() or isinstance(label, (int, np.integer)) else str(label)
                if str(label) == "-1":
                    lbl_name = "Noise / Anomaly"
                ax.scatter(
                    x[mask], y[mask],
                    label=lbl_name,
                    marker=marker,
                    facecolor=color,
                    edgecolor=PRIMARY_BLACK,
                    s=32,
                    alpha=0.75,
                    linewidth=0.6,
                )
            if len(unique_labels) <= 10:
                legend = ax.legend(frameon=True, facecolor="#FFFFFF", edgecolor=LIGHT_BORDER, fontsize=8, loc="best")
                legend.get_frame().set_linewidth(0.8)
        else:
            ax.scatter(x, y, facecolor=CHARCOAL, edgecolor=PRIMARY_BLACK, s=28, alpha=0.65, linewidth=0.6)
        _style_axis(ax, title=title, xlabel="Principal Component 1 (PC1)", ylabel="Principal Component 2 (PC2)")
    else:
        ax.plot(coords.iloc[:, 0], color=PRIMARY_BLACK, linewidth=1.2)
        _style_axis(ax, title="PCA Component 1", xlabel="Record index", ylabel="Score")
    fig.tight_layout()
    return fig


def plot_clustering(
    coords: pd.DataFrame,
    labels: np.ndarray,
    centroids_2d: np.ndarray | None = None,
    title: str = "Cluster Partitions (2D Projection)",
    xlabel: str = "Principal Component 1 (PC1)",
    ylabel: str = "Principal Component 2 (PC2)",
) -> plt.Figure:
    """Monochrome 2D scatter plot visualizing clusters, noise points, and centroids."""
    if coords is None or not isinstance(coords, pd.DataFrame) or coords.empty:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(
            0.5, 0.5, "No 2D coordinates available for visualization",
            ha="center", va="center", color=MID_GREY, fontsize=9.5,
            transform=ax.transAxes,
        )
        _style_axis(ax, title=title)
        fig.tight_layout()
        return fig

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    x_col = coords.columns[0]
    y_col = coords.columns[1] if len(coords.columns) > 1 else coords.columns[0]
    x = coords[x_col].to_numpy(dtype=float)
    y = coords[y_col].to_numpy(dtype=float) if len(coords.columns) > 1 else np.zeros_like(x)

    labels_arr = np.asarray(labels)
    total_pts = len(labels_arr)

    # Distinct monochrome markers and shades
    markers = ["o", "s", "^", "D", "v", "P", "*"]
    shades = [PRIMARY_BLACK, CHARCOAL, "#52525B", MID_GREY, "#A1A1AA", "#D4D4D8", "#3F3F46"]

    # Identify regular clusters and noise points (-1)
    unique_labels = sorted(list(set(labels_arr.tolist())), key=lambda v: (str(v) == "-1", v))
    regular_clusters = [lbl for lbl in unique_labels if str(lbl) != "-1"]
    has_noise = -1 in labels_arr or "-1" in [str(lbl) for lbl in unique_labels]

    # 1. Plot regular clusters
    for idx, lbl in enumerate(regular_clusters):
        mask = labels_arr == lbl
        pts_count = int(np.sum(mask))
        pct = (pts_count / total_pts * 100) if total_pts > 0 else 0.0
        marker = markers[idx % len(markers)]
        color = shades[idx % len(shades)]
        lbl_name = f"Cluster {lbl} ({pts_count} pts · {pct:.1f}%)"
        ax.scatter(
            x[mask], y[mask],
            label=lbl_name,
            marker=marker,
            facecolor=color,
            edgecolor=PRIMARY_BLACK,
            s=36,
            alpha=0.75,
            linewidth=0.6,
            zorder=3,
        )

    # 2. Plot noise points (DBSCAN -1) with distinct marker and styling
    if has_noise:
        noise_mask = (labels_arr == -1) | (labels_arr == "-1")
        noise_count = int(np.sum(noise_mask))
        noise_pct = (noise_count / total_pts * 100) if total_pts > 0 else 0.0
        ax.scatter(
            x[noise_mask], y[noise_mask],
            label=f"Noise / Outlier ({noise_count} pts · {noise_pct:.1f}%)",
            marker="x",
            color=MID_GREY,
            s=44,
            alpha=0.75,
            linewidth=1.2,
            zorder=2,
        )

    # 3. Plot K-Means Centroids if provided
    if centroids_2d is not None and len(centroids_2d) > 0:
        c_x = centroids_2d[:, 0]
        c_y = centroids_2d[:, 1] if centroids_2d.shape[1] > 1 else np.zeros_like(c_x)
        ax.scatter(
            c_x, c_y,
            label="Centroids (Anchor points)",
            marker="X",
            s=140,
            facecolor=PRIMARY_BLACK,
            edgecolor="#FFFFFF",
            linewidth=1.6,
            zorder=10,
        )

    _style_axis(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    legend = ax.legend(
        frameon=True,
        facecolor="#FFFFFF",
        edgecolor=LIGHT_BORDER,
        fontsize=8,
        loc="best",
    )
    if legend:
        legend.get_frame().set_linewidth(0.8)

    fig.tight_layout()
    return fig


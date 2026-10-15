"""Model Forge: a local, on-device AutoML platform for tabular data.

Supports both Supervised and Unsupervised learning workflows entirely on-device
using scikit-learn. No APIs, no cloud dependencies, no network calls.
"""
from __future__ import annotations

import io
import pathlib
import matplotlib

matplotlib.use("Agg")
import pandas as pd  # noqa: E402
import numpy as np  # noqa: E402
import streamlit as st  # noqa: E402
import joblib  # noqa: E402

from automl.core import run_automl, detect_task  # noqa: E402
from automl.unsupervised import run_unsupervised, get_2d_projection  # noqa: E402
from automl.profiling import profile  # noqa: E402
from automl.quality import inspect_dataset_quality  # noqa: E402
from automl.report import build_report_md  # noqa: E402
from automl.explain import (  # noqa: E402
    build_decisions,
    explain_supervised_results,
    explain_unsupervised_results,
    explain_why_model_won,
    explain_unsupervised_selection,
    recommend_next_steps_supervised,
    recommend_next_steps_unsupervised,
    explain_limitations_supervised,
    explain_limitations_unsupervised,
    build_model_comparison_arena,
    explain_feature_importance,
    explain_clustering_visualizer,
)
from automl.prediction import (  # noqa: E402
    get_feature_input_schema,
    make_single_prediction,
    explain_prediction,
)
from automl import plots  # noqa: E402
from automl import ui  # noqa: E402

st.set_page_config(page_title="Model Forge · AutoML", layout="wide")
ui.inject_css()

if "result" not in st.session_state:
    st.session_state.result = None
if "unsupervised_result" not in st.session_state:
    st.session_state.unsupervised_result = None
if "data_signature" not in st.session_state:
    st.session_state.data_signature = None


@st.cache_data(show_spinner=False)
def load_df(name: str, data: bytes) -> pd.DataFrame:
    if name.lower().endswith((".xlsx", ".xls")):
        return pd.read_excel(io.BytesIO(data))
    return pd.read_csv(io.BytesIO(data))


# Header pinned to top
header_placeholder = st.empty()

# ---- Mode Selector ----
mode = st.radio(
    "Learning Mode",
    ["Supervised AutoML", "Unsupervised AutoML"],
    horizontal=True,
    help="Supervised requires a target label column; Unsupervised discovers patterns without labels.",
)

# ---- Section 1 · Dataset ----
ui.render_section("Section 1", "Dataset", "Upload a CSV or Excel spreadsheet to begin analysis.")
col_upload, col_sample = st.columns([3, 1])
with col_upload:
    uploaded = st.file_uploader(
        "Upload dataset (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        label_visibility="collapsed",
    )
with col_sample:
    sample_name = st.selectbox(
        "Sample dataset",
        ["Select sample...", "iris_classification.csv", "diabetes_regression.csv"],
        label_visibility="collapsed",
        help="Quickly test with bundled sample datasets",
    )

df: pd.DataFrame | None = None
active_dataset_name: str | None = None

if uploaded is not None:
    try:
        raw_bytes = uploaded.getvalue()
        df = load_df(uploaded.name, raw_bytes)
        active_dataset_name = uploaded.name
        current_sig = f"{uploaded.name}:{len(raw_bytes)}:{hash(raw_bytes)}"
        if st.session_state.data_signature != current_sig:
            st.session_state.result = None
            st.session_state.unsupervised_result = None
            st.session_state.data_signature = current_sig
    except Exception as exc:
        st.error(f"Error loading dataset: {exc}. Please verify the file is a valid CSV or Excel spreadsheet.")
        df = None
elif sample_name and sample_name != "Select sample...":
    try:
        sample_path = pathlib.Path(__file__).resolve().parent / "sample_data" / sample_name
        if sample_path.exists():
            df = pd.read_csv(sample_path)
            active_dataset_name = sample_name
            current_sig = f"sample:{sample_name}"
            if st.session_state.data_signature != current_sig:
                st.session_state.result = None
                st.session_state.unsupervised_result = None
                st.session_state.data_signature = current_sig
    except Exception as exc:
        st.error(f"Could not load sample dataset: {exc}")
        df = None

# Update top header with current dataset status
with header_placeholder:
    if df is not None and active_dataset_name:
        ui.render_header(active_dataset_name, len(df), df.shape[1])
    else:
        ui.render_header(None, None, None)

if df is None:
    ui.render_flowchart()
    st.stop()

# Dataset statistics summary
c1, c2, c3, c4 = st.columns(4)
c1.metric("Rows", f"{len(df):,}")
c2.metric("Columns", f"{df.shape[1]:,}")
c3.metric("Missing cells", f"{int(df.isna().sum().sum()):,}")
num_cols = len(df.select_dtypes(include=[np.number, "bool"]).columns)
cat_cols = df.shape[1] - num_cols
c4.metric("Features", f"{num_cols} num / {cat_cols} cat")

st.dataframe(df.head(25), use_container_width=True)
with st.expander("Column Profile & Diagnostics", expanded=False):
    st.dataframe(profile(df), use_container_width=True)

with st.expander("Data Quality Inspector", expanded=False):
    quality_res = inspect_dataset_quality(df)
    if quality_res["status"] == "success":
        # Overview Metric Cards
        qc1, qc2, qc3, qc4, qc5 = st.columns(5)
        qc1.metric("Rows", f"{quality_res['n_rows']:,}")
        qc2.metric("Columns", f"{quality_res['n_cols']:,}")
        qc3.metric(
            "Duplicate Rows",
            f"{quality_res['n_duplicates']:,}",
            f"{quality_res['duplicate_pct']:.1f}% of data",
        )
        qc4.metric(
            "Missing Values",
            f"{quality_res['total_missing_cells']:,}",
            f"{quality_res['missing_cells_pct']:.1f}% of cells",
        )
        qc5.metric(
            "Potential Outliers",
            f"{quality_res['total_outliers']:,}",
            f"{quality_res['cols_with_outliers']} num col(s)",
        )

        # Overall Health Rating Banner
        st.markdown(
            f"""
            <div style="border: 1px solid #18181B; border-radius: 6px; padding: 1rem 1.25rem; background: #FAFAFA; margin: 1rem 0 1.25rem 0;">
                <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #71717A; font-weight: 600; margin-bottom: 0.25rem;">Overall Data Quality Rating · {quality_res['rating_badge']}</div>
                <div style="font-size: 1.15rem; font-weight: 600; color: #18181B; margin-bottom: 0.4rem;">{quality_res['rating']}</div>
                <div style="font-size: 0.85rem; color: #27272A; line-height: 1.5;">{quality_res['summary']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        # Per-Column Health Breakdown Table
        st.markdown("**Per-Column Health Breakdown**")
        st.dataframe(quality_res["table_df"], use_container_width=True, hide_index=True)

        # Actionable Quality Recommendations
        st.markdown("**Actionable Quality Recommendations**")
        for rec in quality_res["recommendations"]:
            st.markdown(f"- {rec}")

        # Note on statistical outliers
        st.info(
            "ℹ️ **Note on Outliers:** Potential outliers are detected using the standard 1.5×IQR "
            "(Interquartile Range) method: values outside [Q1 - 1.5×IQR, Q3 + 1.5×IQR]. These are "
            "statistical extremes relative to the distribution, not necessarily errors or bad data. "
            "Model Forge inspects and reports data quality only — it never silently alters or deletes your data."
        )

        # Educational Guide
        with st.expander("Understanding Data Quality: Missing vs. Duplicates vs. Outliers", expanded=False):
            guide = quality_res["educational_guide"]
            st.markdown(guide["missing_values"])
            st.markdown(guide["duplicate_rows"])
            st.markdown(guide["outliers"])
    else:
        st.warning(quality_res.get("message", "Unable to inspect dataset."))


# ==============================================================================
# UNSUPERVISED AUTOML WORKFLOW
# ==============================================================================
if mode == "Unsupervised AutoML":
    all_cols = list(df.columns)
    default_features = [c for c in all_cols if df[c].nunique(dropna=True) > 1]

    # ---- Section 2 · Configuration ----
    ui.render_section(
        "Section 2", "Configuration",
        "Select feature columns and configure parameters for clustering, anomaly detection, and dimensionality reduction."
    )
    features_unsup = st.multiselect(
        "Input features (no target column required)",
        all_cols,
        default=default_features,
        help="Zero-variance and constant columns are omitted automatically.",
    )
    methods = st.multiselect(
        "Analysis types",
        ["Clustering", "Anomaly detection", "Dimensionality reduction"],
        default=["Clustering", "Anomaly detection", "Dimensionality reduction"],
    )

    a1, a2, a3 = st.columns(3)
    with a1:
        n_clusters = st.slider("Clusters (K-Means & Gaussian Mixture)", 2, 12, 3)
    with a2:
        contamination = st.slider(
            "Expected anomaly fraction (prior)",
            0.01, 0.25, 0.05, 0.01,
            help="Configured baseline proportion of records expected to be anomalous. Used as a prior assumption.",
        )
    with a3:
        eps = st.slider("DBSCAN neighborhood radius (eps)", 0.1, 5.0, 0.5, 0.1)
    min_samples = st.slider("DBSCAN minimum samples", 2, 20, 5)

    # ---- Section 3 · Analysis ----
    ui.render_section("Section 3", "Analysis", "Execute algorithm comparisons and generate diagnostics.")
    run_unsup = st.button(
        "Run Unsupervised Analysis",
        type="primary",
        disabled=not features_unsup or not methods,
    )

    if run_unsup:
        prog = st.progress(0.0)
        status = st.empty()

        def unsup_cb(msg: str, frac: float):
            status.text(msg)
            prog.progress(min(max(frac, 0.0), 1.0))

        try:
            with st.spinner("Executing unsupervised algorithms..."):
                st.session_state.unsupervised_result = run_unsupervised(
                    df, features_unsup, methods,
                    n_clusters=n_clusters, contamination=contamination,
                    eps=eps, min_samples=min_samples, progress=unsup_cb,
                )
            status.text("Analysis complete.")
        except Exception as exc:
            st.error(f"Could not complete unsupervised analysis: {exc}")

    unsup_result = st.session_state.get("unsupervised_result")

    # ---- Section 4 · Results ----
    if unsup_result is not None:
        ui.render_section(
            "Section 4", "Results",
            "Review diagnostic metrics, compare algorithm performance, and inspect clustered partitions or flagged anomalies."
        )

        board = unsup_result["leaderboard"]
        cluster_board = board[(board["Analysis"] == "Clustering") & board["Silhouette score"].notna()]
        if not cluster_board.empty:
            winner = cluster_board.sort_values("Silhouette score", ascending=False).iloc[0]
            st.success(
                f"Top clustering candidate: **{winner['Algorithm']}** · Silhouette score **{winner['Silhouette score']:.4f}**. "
                "Recommendation is based strictly on cluster compactness and separation."
            )

        st.dataframe(board, use_container_width=True)
        st.caption(
            "Clustering algorithms are compared using Silhouette score when defined (higher indicates clearer cluster separation). "
            "Anomaly detection scores reflect records flagged under the configured prior and are not ground-truth accuracy. "
            "PCA is evaluated by retained explained variance."
        )

        tabs = st.tabs(["Clustering", "Anomaly Detection", "Dimensionality Reduction (PCA)"])

        # Tab 1: Clustering
        with tabs[0]:
            cluster_names = [n for n, o in unsup_result["outputs"].items() if o["kind"] == "clustering"]
            if cluster_names:
                selected_cluster = st.selectbox("Select clustering algorithm", cluster_names, key="clustering_alg_selector")
                output = unsup_result["outputs"][selected_cluster]

                mcols = st.columns(5)
                mcols[0].metric("Clusters found", output["metrics"].get("Clusters found", "N/A"))
                mcols[1].metric("Noise points", output["metrics"].get("Noise points", 0))

                sil = output["metrics"].get("Silhouette score")
                mcols[2].metric("Silhouette score", f"{sil:.4f}" if isinstance(sil, (float, np.floating)) and np.isfinite(sil) else "N/A")

                db = output["metrics"].get("Davies-Bouldin index")
                mcols[3].metric("Davies-Bouldin", f"{db:.4f}" if isinstance(db, (float, np.floating)) and np.isfinite(db) else "N/A")

                ch = output["metrics"].get("Calinski-Harabasz", output["metrics"].get("Calinski-Harabasz score"))
                mcols[4].metric("Calinski-Harabasz", f"{ch:.1f}" if isinstance(ch, (float, np.floating)) and np.isfinite(ch) else "N/A")

                # ---- Clustering Visualizer ----
                st.markdown("---")
                st.markdown(f"**Clustering Visualizer · {selected_cluster}**")

                coords_2d, pca_model, used_pca = get_2d_projection(unsup_result, df)
                labels = output.get("labels")

                # Project centroids if K-Means
                centroids_2d = None
                if selected_cluster == "K-Means" and "K-Means" in unsup_result.get("fitted", {}):
                    km_model = unsup_result["fitted"]["K-Means"].get("model")
                    if km_model is not None and hasattr(km_model, "cluster_centers_"):
                        if pca_model is not None:
                            try:
                                centroids_2d = pca_model.transform(km_model.cluster_centers_)
                            except Exception:
                                centroids_2d = None
                        elif coords_2d is not None and km_model.cluster_centers_.shape[1] == 2:
                            centroids_2d = km_model.cluster_centers_

                if coords_2d is not None and labels is not None and len(labels) == len(coords_2d):
                    # 1. 2D Scatter Plot
                    st.pyplot(
                        plots.plot_clustering(
                            coords_2d,
                            labels,
                            centroids_2d=centroids_2d,
                            title=f"{selected_cluster} Partitions (2D Projection)",
                            xlabel="Principal Component 1 (PC1)" if used_pca else (coords_2d.columns[0] if len(coords_2d.columns) > 0 else "X"),
                            ylabel="Principal Component 2 (PC2)" if used_pca else (coords_2d.columns[1] if len(coords_2d.columns) > 1 else "Y"),
                        )
                    )

                    # 2. Explanations & Context
                    n_feats = len(unsup_result.get("features", []))
                    exp_vis = explain_clustering_visualizer(
                        selected_cluster,
                        labels,
                        n_features=n_feats,
                        used_pca=used_pca,
                        has_centroids=(centroids_2d is not None),
                    )

                    st.markdown("**Understanding This Clustering**")
                    st.write(exp_vis["clustering_concept"])
                    st.caption(exp_vis["algorithm_description"])

                    if exp_vis.get("centroid_explanation"):
                        st.info(exp_vis["centroid_explanation"])

                    st.caption(exp_vis["pca_axes_explanation"])

                    if exp_vis.get("noise_explanation"):
                        st.warning(exp_vis["noise_explanation"])

                    # 3. Cluster Partitions Breakdown Table
                    st.markdown("**Cluster Partitions Breakdown**")
                    st.dataframe(exp_vis["breakdown_df"], use_container_width=True, hide_index=True)
                else:
                    st.warning(
                        f"2D visualization is unavailable for {selected_cluster} due to insufficient coordinates or record mismatch."
                    )

                # 4. Row assignments preview & CSV download
                st.markdown("**Cluster Assignments Preview**")
                cluster_df = pd.DataFrame({"Row": df.index, "Assigned Cluster": output["labels"]})
                st.dataframe(cluster_df.head(100), use_container_width=True)

                csv_clusters = cluster_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label=f"Download {selected_cluster} Assignments (CSV)",
                    data=csv_clusters,
                    file_name=f"{selected_cluster.lower().replace(' ', '_')}_cluster_assignments.csv",
                    mime="text/csv",
                    key=f"btn_download_{selected_cluster.lower().replace(' ', '_')}",
                )
            else:
                st.info("Clustering was not included in this run. Enable 'Clustering' under Configuration to compare grouping algorithms.")

        # Tab 2: Anomaly Detection
        with tabs[1]:
            anomaly_names = [n for n, o in unsup_result["outputs"].items() if o["kind"] == "anomaly"]
            if anomaly_names:
                selected_anomaly = st.selectbox("Select anomaly detector", anomaly_names)
                output = unsup_result["outputs"][selected_anomaly]

                acols = st.columns(2)
                acols[0].metric("Flagged records", f"{output['metrics']['Anomaly count']:,}")
                acols[1].metric("Flagged fraction", f"{output['metrics']['Anomaly rate']:.1%}")

                st.caption(
                    f"Notice: The flagged rate reflects the configured contamination threshold ({contamination:.1%}). "
                    "In unsupervised anomaly detection without labels, flagged rows are outliers relative to expected distribution."
                )

                mask = output["anomaly_mask"]
                flagged_df = df.loc[mask]
                if not flagged_df.empty:
                    st.write(f"**Flagged anomalous records ({len(flagged_df):,})**")
                    st.dataframe(flagged_df, use_container_width=True)
                else:
                    st.write("No anomalous records flagged under current sensitivity settings.")
            else:
                st.info("Anomaly detection was not included in this run. Enable 'Anomaly detection' under Configuration to inspect outliers.")

        # Tab 3: Dimensionality Reduction (PCA)
        with tabs[2]:
            pca_output = unsup_result["outputs"].get("PCA")
            if pca_output:
                coords = pca_output["coordinates"]
                pcols = st.columns(2)
                pcols[0].metric("Total variance retained", f"{pca_output['metrics']['Explained variance']:.1%}")
                pcols[1].metric("PC1 variance", f"{pca_output['metrics']['PC1 variance']:.1%}")

                # If clustering was also run, overlay cluster labels on the monochrome scatter chart
                active_labels = None
                cluster_names = [n for n, o in unsup_result["outputs"].items() if o["kind"] == "clustering"]
                if cluster_names:
                    active_labels = unsup_result["outputs"][cluster_names[0]]["labels"]

                st.pyplot(plots.plot_pca(coords, labels=active_labels, title="PCA Projection (PC1 vs. PC2)"))
                st.write("**Transformed coordinates**")
                st.dataframe(coords.head(50), use_container_width=True)
            else:
                st.info("Dimensionality reduction was not included in this run. Enable 'Dimensionality reduction' to project features onto principal components.")

        # ---- Explain My Results ----
        with st.expander("Explain My Results", expanded=False):
            active_cluster = selected_cluster if "selected_cluster" in locals() else None
            active_anomaly = selected_anomaly if "selected_anomaly" in locals() else None
            exp = explain_unsupervised_results(
                unsup_result, selected_model=active_cluster, selected_anomaly=active_anomaly
            )
            if exp.get("status") == "success":
                st.markdown(f"**Overview:** {exp['headline']}")

                c_info = exp.get("clustering")
                if c_info:
                    st.write(f"**Clustering Analysis ({c_info['selected_algorithm']})**")
                    st.markdown(c_info["summary"])
                    for m in c_info.get("metrics_explanations", []):
                        st.markdown(
                            f"- **{m['metric']}** (`{m['formatted_value']}`) · "
                            f"*{m['label']}*: {m['interpretation']}"
                        )

                a_info = exp.get("anomaly")
                if a_info:
                    st.write(f"**Anomaly Detection ({a_info['selected_detector']})**")
                    st.markdown(a_info["summary"])
                    for m in a_info.get("metrics_explanations", []):
                        st.markdown(
                            f"- **{m['metric']}** (`{m['formatted_value']}`) · "
                            f"*{m['label']}*: {m['interpretation']}"
                        )

                p_info = exp.get("pca")
                if p_info:
                    st.write("**Dimensionality Reduction (PCA)**")
                    st.markdown(p_info["summary"])
                    for m in p_info.get("metrics_explanations", []):
                        st.markdown(
                            f"- **{m['metric']}** (`{m['formatted_value']}`) · "
                            f"*{m['label']}*: {m['interpretation']}"
                        )

                takeaways = exp.get("key_takeaways") or []
                if takeaways:
                    st.write("**Key Takeaways**")
                    for t in takeaways:
                        st.markdown(f"- {t}")
            else:
                st.info("No explanation available for the current unsupervised results.")

        # ---- Why Did This Model Win? ----
        with st.expander("Why Did This Model Win?", expanded=False):
            active_cluster = selected_cluster if "selected_cluster" in locals() else None
            why_unsup = explain_unsupervised_selection(unsup_result, selected_model=active_cluster)
            if why_unsup.get("status") == "success":
                st.markdown(f"**Evaluation Context:** {why_unsup['headline']}")

                # Clustering comparison (if run)
                c_comp = why_unsup.get("clustering")
                if c_comp:
                    st.write("**Clustering Candidate Comparison**")
                    st.markdown(c_comp["summary"])
                    st.caption(c_comp["caveat"])
                    for cand in c_comp.get("candidates", []):
                        if cand.get("silhouette") is not None:
                            prefix = "🏆 " if cand["algorithm"] == c_comp["top_algorithm"] else "- "
                            st.markdown(
                                f"{prefix}**{cand['algorithm']}**: Silhouette score `{cand['silhouette']:.4f}` "
                                f"({cand['clusters']} clusters, {cand['noise']} noise points)"
                            )
                        else:
                            st.markdown(
                                f"- **{cand['algorithm']}**: Silhouette score not defined "
                                f"({cand['clusters']} clusters, {cand['noise']} noise points)"
                            )

                # Anomaly context (if run)
                a_comp = why_unsup.get("anomaly")
                if a_comp:
                    st.write("**Anomaly Detection Comparison (No Ground-Truth Winner)**")
                    st.markdown(a_comp["explanation"])

                # PCA context (if run)
                p_comp = why_unsup.get("pca")
                if p_comp:
                    st.write("**Dimensionality Reduction (PCA)**")
                    st.markdown(p_comp["explanation"])

                # Honest notes
                notes = why_unsup.get("honest_guidance") or []
                if notes:
                    st.write("**Honest Notes on Unsupervised Evaluation**")
                    for note in notes:
                        st.markdown(f"- {note}")
            else:
                st.info("No unsupervised comparison data available.")

        # ---- What Should I Do Next? ----
        with st.expander("What Should I Do Next?", expanded=False):
            active_cluster = selected_cluster if "selected_cluster" in locals() else None
            next_steps_unsup = recommend_next_steps_unsupervised(unsup_result, selected_model=active_cluster)
            if next_steps_unsup.get("status") == "success":
                st.markdown(f"**Recommended Action Plan:** {next_steps_unsup['headline']}")
                for rec in next_steps_unsup.get("recommendations", []):
                    st.markdown(f"**{rec['title']}**\n\n{rec['advice']}")
                st.caption(f"ℹ️ {next_steps_unsup['disclaimer']}")
            else:
                st.info("No recommendations available for the current unsupervised results.")

        # ---- Limitations and Trust? ----
        with st.expander("Limitations and Trust?", expanded=False):
            active_cluster = selected_cluster if "selected_cluster" in locals() else None
            limits_unsup = explain_limitations_unsupervised(unsup_result, selected_model=active_cluster)
            if limits_unsup.get("status") == "success":
                st.markdown(f"**Trust & Validation Notice:** {limits_unsup['headline']}")
                for item in limits_unsup.get("items", []):
                    st.markdown(f"**{item['title']}**\n\n{item['text']}")
                st.caption(f"🛡️ {limits_unsup['summary']}")
            else:
                st.info("No unsupervised results available to evaluate limitations.")

        # ---- Section 5 · Export ----
        ui.render_section(
            "Section 5", "Export",
            "Download all row-level assignments and serialised model bundles for integration or downstream analysis."
        )
        e1, e2 = st.columns(2)
        with e1:
            st.download_button(
                "Download results table (CSV)",
                unsup_result["results"].to_csv(index=False).encode("utf-8"),
                file_name="model_forge_unsupervised_results.csv",
                mime="text/csv",
                use_container_width=True,
            )
        with e2:
            model_buf = io.BytesIO()
            joblib.dump(unsup_result["fitted"], model_buf)
            st.download_button(
                "Download fitted models (.joblib)",
                model_buf.getvalue(),
                file_name="model_forge_unsupervised_models.joblib",
                mime="application/octet-stream",
                use_container_width=True,
            )
        st.caption(
            "The model bundle contains fitted estimators, preprocessing transformers, and feature names. "
            "Load via: `bundle = joblib.load('model_forge_unsupervised_models.joblib')`"
        )


# ==============================================================================
# SUPERVISED AUTOML WORKFLOW
# ==============================================================================
else:
    cols = list(df.columns)

    # ---- Section 2 · Configuration ----
    ui.render_section(
        "Section 2", "Configuration",
        "Designate the prediction target and select explanatory features."
    )
    target = st.selectbox("Target column (what to predict)", cols, index=len(cols) - 1)
    feature_choices = [c for c in cols if c != target]
    features = st.multiselect("Input features", feature_choices, default=feature_choices)

    auto_task = detect_task(df[target])
    task_label = st.radio(
        "Task type", ["Auto", "Classification", "Regression"],
        horizontal=True,
        help=f"Auto-detected task type: {auto_task}",
    )
    task_override = {"Auto": "auto", "Classification": "classification", "Regression": "regression"}[task_label]
    st.caption(f"Task type resolution: **{auto_task}** (auto-detected).")

    with st.expander("Advanced Optimization Settings", expanded=False):
        test_size = st.slider("Test set split size", 0.1, 0.4, 0.2, 0.05)
        cv_folds = st.slider("Cross-validation folds", 2, 10, 5)
        enable_tuning = st.checkbox("Hyperparameter tuning", value=True)
        search_iters = st.slider("Tuning search iterations", 5, 60, 20)

    # ---- Section 3 · Analysis ----
    ui.render_section("Section 3", "Analysis", "Run cross-validated tournament and tune winning pipeline.")
    run = st.button("Run Supervised Tournament", type="primary", disabled=not features)

    if run:
        prog = st.progress(0.0)
        status = st.empty()

        def cb(msg: str, frac: float):
            status.text(msg)
            prog.progress(min(max(frac, 0.0), 1.0))

        try:
            with st.spinner("Training and evaluating tournament models..."):
                st.session_state.result = run_automl(
                    df, target, features, task_override=task_override,
                    test_size=test_size, cv_folds=cv_folds,
                    enable_tuning=enable_tuning, search_iters=search_iters, progress=cb,
                )
            status.text("Tournament complete.")
        except Exception as exc:
            st.error(f"Could not complete supervised model training: {exc}")

    result = st.session_state.result

    # ---- Section 4 · Results ----
    if result is not None:
        ui.render_section(
            "Section 4", "Results",
            "Inspect held-out validation metrics, tournament ranking, and diagnostic plots."
        )
        st.success(
            f"Best model: **{result['best_model']}** · Task: **{result['task']}** · Optimized for `{result['scoring']}`"
        )

        n_classes = result.get("n_classes") or 0
        if result["task"] == "classification" and n_classes > 20:
            st.warning(
                f"Classification executed with {n_classes} distinct classes. "
                "If this column represents continuous numbers, set Task Type to Regression."
            )

        metric_items = list(result["metrics"].items())
        for (k, v), c in zip(metric_items, st.columns(max(1, len(metric_items)))):
            c.metric(k, f"{v:.4f}")

        tab_lb, tab_plots, tab_imp, tab_play, tab_dec = st.tabs(
            [
                "Model Comparison Arena",
                "Diagnostic Plots",
                "Feature Importance",
                "Prediction Playground",
                "Pipeline Decision Flow",
            ]
        )

        with tab_lb:
            arena = build_model_comparison_arena(result)
            if arena.get("status") == "success":
                st.markdown(
                    f"""
                    <div style="border: 1px solid #18181B; border-radius: 6px; padding: 1rem 1.25rem; background: #FAFAFA; margin-bottom: 1.25rem;">
                        <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #71717A; font-weight: 600; margin-bottom: 0.25rem;">Recommended Model</div>
                        <div style="font-size: 1.15rem; font-weight: 600; color: #18181B; margin-bottom: 0.4rem;">{arena['best_model']}</div>
                        <div style="font-size: 0.85rem; color: #27272A; line-height: 1.5;">{arena['recommendation_explanation']}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                st.dataframe(arena["table"], use_container_width=True, hide_index=True)
                st.caption(f"ℹ️ {arena['metric_guidance']}")
                if arena.get("failed_models"):
                    for fm in arena["failed_models"]:
                        st.caption(f"⚠️ **{fm['model']}** could not be evaluated: {fm.get('error', 'Execution error')}")
            else:
                st.dataframe(pd.DataFrame(result.get("leaderboard", [])), use_container_width=True)

            if result["tuned_params"]:
                st.write("**Tuned Hyperparameters**")
                st.json(result["tuned_params"])

        with tab_plots:
            cc1, cc2 = st.columns(2)
            with cc1:
                st.pyplot(plots.plot_target_distribution(result["y_test"], result["task"]))
            with cc2:
                if result["task"] == "classification":
                    if (result.get("n_classes") or 0) <= 20:
                        st.pyplot(plots.plot_confusion(result["confusion_matrix"], result["classes"]))
                    else:
                        st.info("Confusion matrix suppressed: target contains more than 20 classes.")
                else:
                    st.pyplot(plots.plot_pred_vs_actual(result["y_test"], result["y_pred"]))

        with tab_imp:
            feat_imp_df = result.get("feature_importance")
            imp_method = result.get("importance_method", "built-in")
            best_model_name = result.get("best_model", "Selected Model")

            exp_feat = explain_feature_importance(
                feat_imp_df,
                method=imp_method,
                model_name=best_model_name,
            )

            if exp_feat.get("has_features") and feat_imp_df is not None and not feat_imp_df.empty:
                # Top header bar: Method metadata and display slider
                col_chart_hdr, col_slider = st.columns([3, 2])
                with col_chart_hdr:
                    st.markdown("**Feature Importance Ranking**")
                    st.caption(f"Method: {exp_feat['method_title']} · {exp_feat['method_badge']}")

                n_available = len(feat_imp_df)
                default_top = min(15, n_available)
                with col_slider:
                    if n_available > 5:
                        display_k = st.slider(
                            "Display top features",
                            min_value=min(5, n_available),
                            max_value=min(30, n_available),
                            value=default_top,
                            step=1,
                            key="feat_imp_slider",
                        )
                    else:
                        display_k = n_available

                # Horizontal bar chart ranking input features by importance
                st.pyplot(plots.plot_importance(feat_imp_df, max_n=display_k))

                # Beginner-friendly explanation
                st.markdown("---")
                st.markdown("**Understanding This Chart**")
                st.write(exp_feat["beginner_explanation"])
                st.caption(exp_feat["method_detail"])

                # Prominent Causation vs. Correlation callout
                st.info(exp_feat["causation_warning"])

                # Formatted Detailed Breakdown Table
                st.markdown("**Feature Importance Breakdown**")
                table_df = exp_feat["table_df"]
                st.dataframe(table_df, use_container_width=True, hide_index=True)

                # Export CSV button
                csv_data = table_df.to_csv(index=False).encode("utf-8")
                target_col = result.get("target", "target")
                st.download_button(
                    label="Download Feature Importance (CSV)",
                    data=csv_data,
                    file_name=f"{target_col}_feature_importance.csv",
                    mime="text/csv",
                    key="btn_download_feat_imp",
                )
            else:
                st.warning(
                    f"**Feature importance unavailable for {best_model_name}.**\n\n"
                    "This model architecture does not provide intrinsic feature weights, "
                    "and permutation calculation was unavailable for this dataset configuration. "
                    "All model evaluations, metrics, and predictions remain completely valid."
                )
                st.info(
                    "**Note on Feature Importance:** Feature importance measures mathematical influence on predictions, "
                    "not proof of causation."
                )

        with tab_play:
            best_model_name = result.get("best_model", "Selected Model")
            task_type = result.get("task", "classification")
            target_name = result.get("target", "Target")

            st.markdown(f"**Interactive Prediction Playground · {best_model_name}**")
            st.caption(
                f"Test your trained model on custom feature inputs in real time. "
                f"The pipeline's fitted preprocessing (imputation, scaling, encoding) is reused automatically without retraining."
            )

            feature_cols = [f for f in result.get("features", []) if f != target_name]
            if not feature_cols:
                st.warning("No input feature columns available for prediction.")
            else:
                schema = get_feature_input_schema(df, feature_cols, target_col=target_name)

                # Quick fill helpers
                q_col1, q_col2, _ = st.columns([1.5, 1.5, 3])
                with q_col1:
                    if st.button("Load Median / Default Values", key="btn_fill_median"):
                        for item in schema:
                            st.session_state[f"play_input_{item['name']}"] = item["default"]
                        st.rerun()
                with q_col2:
                    if st.button("Load Random Row from Dataset", key="btn_fill_random"):
                        sample_row = df.sample(1, random_state=None).iloc[0]
                        for item in schema:
                            col_name = item["name"]
                            if col_name in sample_row and pd.notna(sample_row[col_name]):
                                val = sample_row[col_name]
                                if item["type"] == "numeric":
                                    st.session_state[f"play_input_{col_name}"] = float(val)
                                elif item["type"] == "boolean":
                                    st.session_state[f"play_input_{col_name}"] = bool(val)
                                else:
                                    st.session_state[f"play_input_{col_name}"] = str(val)
                        st.rerun()

                # Render dynamic input fields in a responsive 2-column or 3-column layout
                st.markdown("**Enter Feature Values:**")
                n_cols = 3 if len(schema) >= 6 else (2 if len(schema) >= 2 else 1)
                ui_cols = st.columns(n_cols)

                input_values = {}
                for idx, item in enumerate(schema):
                    col_target = ui_cols[idx % n_cols]
                    field_name = item["name"]
                    field_key = f"play_input_{field_name}"

                    with col_target:
                        if item["type"] == "numeric":
                            if field_key not in st.session_state:
                                st.session_state[field_key] = float(item["default"])
                            input_values[field_name] = st.number_input(
                                label=field_name,
                                step=float(item["step"]),
                                key=field_key,
                            )
                        elif item["type"] == "boolean":
                            if field_key not in st.session_state:
                                st.session_state[field_key] = item["default"]
                            input_values[field_name] = st.selectbox(
                                label=field_name,
                                options=item["options"],
                                key=field_key,
                            )
                        else:
                            # Categorical
                            if field_key not in st.session_state:
                                st.session_state[field_key] = item["default"]
                            input_values[field_name] = st.selectbox(
                                label=field_name,
                                options=item["options"],
                                key=field_key,
                            )

                # Generate Prediction Button
                st.markdown("")
                if st.button("Generate Prediction", type="primary", key="btn_run_prediction"):
                    pred_res = make_single_prediction(
                        pipeline=result["pipeline"],
                        task=task_type,
                        input_values=input_values,
                        feature_names=feature_cols,
                        target_name=target_name,
                        classes=result.get("classes"),
                    )

                    if pred_res.get("status") == "success":
                        st.markdown("---")
                        st.markdown("**Prediction Result**")

                        exp = explain_prediction(
                            task=task_type,
                            prediction=pred_res["prediction"],
                            probabilities=pred_res.get("probabilities"),
                            target_name=target_name,
                            best_model=best_model_name,
                        )

                        # Output Banner
                        if task_type == "classification":
                            st.markdown(
                                f"""
                                <div style="border: 2px solid #18181B; border-radius: 8px; padding: 1.25rem 1.5rem; background: #FAFAFA; margin-bottom: 1rem;">
                                    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #71717A; font-weight: 600; margin-bottom: 0.25rem;">Predicted Class for {target_name}</div>
                                    <div style="font-size: 1.6rem; font-weight: 700; color: #18181B;">{pred_res['prediction']}</div>
                                    <div style="font-size: 0.875rem; color: #27272A; margin-top: 0.4rem;">{exp['details']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                            # Class Probabilities Breakdown
                            if pred_res.get("has_probabilities") and pred_res.get("probabilities"):
                                st.write("**Predicted Class Probabilities**")
                                prob_df = pd.DataFrame([
                                    {
                                        "Class": p["class"],
                                        "Confidence Score": f"{p['probability']:.4f}",
                                        "Probability (%)": f"{p['percentage']:.1f}%",
                                    }
                                    for p in pred_res["probabilities"]
                                ])
                                st.dataframe(prob_df, use_container_width=True, hide_index=True)
                        else:
                            st.markdown(
                                f"""
                                <div style="border: 2px solid #18181B; border-radius: 8px; padding: 1.25rem 1.5rem; background: #FAFAFA; margin-bottom: 1rem;">
                                    <div style="font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; color: #71717A; font-weight: 600; margin-bottom: 0.25rem;">Estimated Value for {target_name}</div>
                                    <div style="font-size: 1.6rem; font-weight: 700; color: #18181B;">{pred_res['formatted_prediction']}</div>
                                    <div style="font-size: 0.875rem; color: #27272A; margin-top: 0.4rem;">{exp['details']}</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # Educational & Trust Callout (Requirement 8)
                        st.info(f"💡 **Understanding Predictions:** {exp['disclaimer']}")

                        # Input Record Summary
                        with st.expander("Inspect Raw Input Vector Sent to Pipeline", expanded=False):
                            st.dataframe(pd.DataFrame([input_values]), use_container_width=True)
                    else:
                        st.error(f"Prediction Error: {pred_res.get('message', 'Unknown failure')}")

        with tab_dec:
            decisions = build_decisions(result)
            ui.render_decision_graph("Data Preparation Pipeline", decisions["data"], "Raw Dataset")
            ui.render_decision_graph("Algorithm Selection & Tuning", decisions["model"], "Prepared Features")

        # ---- Explain My Results ----
        with st.expander("Explain My Results", expanded=False):
            exp = explain_supervised_results(result)
            if exp.get("status") == "success":
                st.markdown(f"**Overview:** {exp['headline']}")
                if exp.get("algorithm_rationale"):
                    st.caption(exp["algorithm_rationale"])

                metrics_exp = exp.get("metrics_explanations") or []
                if metrics_exp:
                    st.write("**Metric Breakdown**")
                    for m in metrics_exp:
                        st.markdown(
                            f"- **{m['metric']}** (`{m['formatted_value']}`) · "
                            f"*{m['label']}*: {m['interpretation']}"
                        )

                feat = exp.get("feature_insights") or {}
                if feat.get("has_features") and feat.get("summary"):
                    st.write("**Feature Impact**")
                    st.markdown(feat["summary"])

                takeaways = exp.get("key_takeaways") or []
                if takeaways:
                    st.write("**Key Takeaways**")
                    for t in takeaways:
                        st.markdown(f"- {t}")
            else:
                st.info("No explanation available for the current results.")

        # ---- Why Did This Model Win? ----
        with st.expander("Why Did This Model Win?", expanded=False):
            why = explain_why_model_won(result)
            if why.get("status") == "success":
                st.markdown(f"**Tournament Champion:** {why['headline']}")

                # Scoring objective
                metric_info = why.get("metric_info") or {}
                st.markdown(
                    f"**Optimization Metric:** `{why['scoring']}` ({metric_info.get('name', why['scoring'])}) · "
                    f"**{metric_info.get('direction_text', 'Higher is better')}**\n\n"
                    f"{metric_info.get('description', '')}"
                )

                # Margin & close contest analysis
                if why.get("margin_summary"):
                    st.markdown(f"**Tournament Contest:** {why['margin_summary']}")

                # Comparison Table / List
                comp_items = why.get("comparison_items") or []
                if comp_items:
                    st.write("**Leaderboard Comparison**")
                    for item in comp_items:
                        if item["role"] == "Winner":
                            st.markdown(
                                f"- 🏆 **{item['model']}** (Winner): `{item['score']:.4f}` ± {item['std']:.4f} "
                                f"({item['time_seconds']:.2f}s)"
                            )
                        elif item["role"] == "Runner-up":
                            st.markdown(
                                f"- 🥈 **{item['model']}** (Runner-up): `{item['score']:.4f}` ± {item['std']:.4f} "
                                f"(-{item['diff_from_best']:.4f}, {item['time_seconds']:.2f}s)"
                            )
                        elif item["role"] == "Contender":
                            st.markdown(
                                f"- **{item['model']}**: `{item['score']:.4f}` ± {item['std']:.4f} "
                                f"(-{item['diff_from_best']:.4f}, {item['time_seconds']:.2f}s)"
                            )
                        elif item["role"] == "Failed":
                            st.markdown(
                                f"- **{item['model']}**: *Evaluation failed* ({item.get('error', 'Error')})"
                            )

                # Algorithmic strengths
                if why.get("strengths_reason"):
                    st.write("**Architectural Advantage**")
                    st.markdown(why["strengths_reason"])

                # Tuning impact
                if why.get("tuning_impact"):
                    st.write("**Tuning Impact**")
                    st.markdown(why["tuning_impact"])

                # Honest limitations
                limits = why.get("honest_limitations") or []
                if limits:
                    st.write("**Honest Limitations & Context**")
                    for lim in limits:
                        st.markdown(f"- {lim}")
            else:
                st.info("No tournament comparison data available.")

        # ---- What Should I Do Next? ----
        with st.expander("What Should I Do Next?", expanded=False):
            next_steps = recommend_next_steps_supervised(result)
            if next_steps.get("status") == "success":
                st.markdown(f"**Recommended Action Plan:** {next_steps['headline']}")
                for rec in next_steps.get("recommendations", []):
                    st.markdown(f"**{rec['title']}**\n\n{rec['advice']}")
                st.caption(f"ℹ️ {next_steps['disclaimer']}")
            else:
                st.info("No recommendations available for the current results.")

        # ---- Limitations and Trust? ----
        with st.expander("Limitations and Trust?", expanded=False):
            limits_sup = explain_limitations_supervised(result)
            if limits_sup.get("status") == "success":
                st.markdown(f"**Trust & Validation Notice:** {limits_sup['headline']}")
                for item in limits_sup.get("items", []):
                    st.markdown(f"**{item['title']}**\n\n{item['text']}")
                st.caption(f"🛡️ {limits_sup['summary']}")
            else:
                st.info("No supervised results available to evaluate limitations.")

        # ---- Section 5 · Export ----
        ui.render_section("Section 5", "Export", "Download serialized pipeline artifact and documentation report.")
        ex1, ex2 = st.columns(2)
        with ex1:
            buf = io.BytesIO()
            joblib.dump(result["pipeline"], buf)
            st.download_button(
                "Download trained pipeline (.joblib)",
                buf.getvalue(),
                file_name="automl_model.joblib",
                mime="application/octet-stream",
                use_container_width=True,
            )
        with ex2:
            st.download_button(
                "Download summary report (.md)",
                build_report_md(result),
                file_name="automl_report.md",
                mime="text/markdown",
                use_container_width=True,
            )

        st.write("**Inference on new records:**")
        st.code(
            'import joblib, pandas as pd\n\n'
            'model = joblib.load("automl_model.joblib")\n'
            'new_data = pd.read_csv("new_records.csv")\n'
            'predictions = model.predict(new_data)',
            language="python",
        )

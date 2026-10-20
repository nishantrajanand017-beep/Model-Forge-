<div align="center">

# 🧠 Model Forge (AutoML Studio)

<a href="https://github.com/nishantrajanand017-beep/Model-Forge-">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&pause=1000&color=F7931E&center=true&vCenter=true&width=620&lines=Spreadsheet+in.+Trained+model+out.;Supervised+%26+Unsupervised+AutoML.;Transparent%2C+explainable+machine+learning.;Powered+entirely+by+scikit-learn." alt="Typing SVG" />
</a>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" alt="Python" />
  <img src="https://img.shields.io/badge/Streamlit-1.36+-FF4B4B?logo=streamlit&logoColor=white" alt="Streamlit" />
  <img src="https://img.shields.io/badge/scikit--learn-1.4+-F7931E?logo=scikit-learn&logoColor=white" alt="scikit-learn" />
  <img src="https://img.shields.io/badge/pandas-2.1+-150458?logo=pandas&logoColor=white" alt="pandas" />
  <img src="https://img.shields.io/badge/NumPy-1.26+-013243?logo=numpy&logoColor=white" alt="NumPy" />
  <img src="https://img.shields.io/badge/Matplotlib-3.8+-11557C?logo=python&logoColor=white" alt="Matplotlib" />
  <img src="https://img.shields.io/badge/License-MIT-3DA639?logo=opensourceinitiative&logoColor=white" alt="License" />
</p>

<em>A no-code, explainable machine-learning studio for tabular data — supporting supervised tournaments, unsupervised discovery, data quality auditing, and interactive inference.</em>

</div>

---

## 📌 Overview

**Model Forge (AutoML Studio)** turns CSV or Excel spreadsheets into trained, tuned, and evaluated machine learning pipelines. Whether predicting a target column or discovering latent groupings without labels, Model Forge guides you through the workflow: pre-flight data quality auditing, automated task detection, leakage-safe preprocessing, model tournament evaluation, hyperparameter optimization, and real-time inference testing.

### 🔒 Execution & Privacy Model
* **Local Execution (Default):** When run locally on your workstation, all computations (profiling, training, tuning, plotting, and inference) execute directly on your machine using Python and scikit-learn. No data leaves your machine, and no external API keys or cloud connections are required.
* **Hosted / Cloud Deployment (e.g., Streamlit Community Cloud):** When deployed to a hosted platform, computations run within the hosting provider's cloud container. Uploaded datasets and models reside in container memory during the session and are subject to the host environment's policies. Hosted execution does **not** run on your local device.

> **UI Preview:** Screenshots and visual walkthroughs will be placed in `docs/` in an upcoming release.

---

## 🚀 Key Features

### 1. 🔍 Pre-Flight Data Quality Inspector
Before model training, inspect dataset health with non-destructive diagnostics:
* **Duplicate & Missing Value Tracking:** Reports exact duplicate row counts, missing cell counts, and per-column completeness percentages.
* **Statistical Outlier Detection:** Employs the standard $1.5 \times \text{IQR}$ (Interquartile Range) method ($[Q_1 - 1.5\text{IQR}, Q_3 + 1.5\text{IQR}]$) to flag numerical extremes without mutating or deleting rows.
* **Dataset Health Rating:** Computes an overall rating badge (*Excellent*, *Good*, *Fair*, *Needs Attention*) alongside actionable data hygiene advice and an educational guide.
* **Non-Mutation Guarantee:** Diagnostic only. Raw datasets are never altered or filtered silently.

### 2. 🎯 Supervised AutoML & Model Tournament
* **Task Auto-Detection:** Intelligently infers classification versus regression based on target column characteristics and cardinality, with a manual override option.
* **Leakage-Safe Preprocessing:** Combines numerical median imputation and standard scaling, low-cardinality ($\le 20$) one-hot encoding, and high-cardinality ordinal encoding into a single scikit-learn `Pipeline`.
* **Model Tournament:** Concurrently evaluates and cross-validates a candidate portfolio:
  * **Classification:** Logistic Regression, Random Forest, HistGradientBoosting, K-Nearest Neighbors.
  * **Regression:** Ridge Regression, Random Forest, HistGradientBoosting, K-Nearest Neighbors.
* **Hyperparameter Optimization:** Performs `RandomizedSearchCV` on the winning pipeline architecture.
* **Model Comparison Arena:** Displays candidate cross-validation scores, standard deviations, run times, contest margins, and failure error logs.
* **Diagnostic Visualizations:** Generates target distributions, confusion matrices ($\le 20$ classes), and predicted-vs-actual scatter plots.
* **Feature Importance:** Computes tree/linear feature importances with automatic permutation importance fallback. Includes an interactive Top-K slider (5–30 features), correlation vs. causation guidance, and CSV export.

### 3. 🧪 Interactive Prediction Playground
* **Dynamic Form Schema:** Automatically generates type-safe input controls (numerical inputs with steps/bounds, categorical selectboxes, boolean dropdowns) based on training feature distributions.
* **Quick-Fill Helpers:** One-click population of input vectors using feature medians/modes or a random sample row directly from the dataset.
* **Real-Time Pipeline Inference:** Evaluates user inputs against the fitted preprocessing and estimator pipeline without retraining.
* **Confidence & Probabilities:** Displays predicted class probability percentages for classification models and formatted point estimates for regression models.

### 4. 🧩 Unsupervised AutoML
Designed for label-free exploratory data analysis, pattern recognition, and anomaly screening:
* **Clustering:** Fits K-Means, Agglomerative Clustering, DBSCAN, and Gaussian Mixture models. Compares algorithms using Silhouette score, Davies-Bouldin index, Calinski-Harabasz score, cluster counts, and noise point tallies.
* **2D Clustering Visualizer:** Projects clustered data into two dimensions (via PCA or leading features), projects K-Means cluster centroids, highlights DBSCAN noise points, displays cluster partition distributions, and provides CSV assignment downloads.
* **Anomaly Detection:** Fits Isolation Forest, Local Outlier Factor, and One-Class SVM. Provides a prior contamination slider and displays flagged anomaly records for inspection.
* **Dimensionality Reduction:** Computes PCA transformations with principal component variance and total explained variance metrics.

### 5. 💡 Explainability & Decision Audit
* **Pipeline Decision Flow:** Visualizes every data cleaning and model selection fork as a flowchart rendered directly in the interface.
* **Modular Explanation Panels:**
  * *"Why Did This Model Win?"* breaks down tournament scoring, algorithmic advantages, and tuning gains.
  * *"What Should I Do Next?"* outlines concrete recommendations for data collection and threshold calibration.
  * *"Limitations and Trust?"* details operational constraints, distribution shift warnings, and sample size considerations.

### 6. 💾 Export Capabilities
* **Supervised:** Download fitted scikit-learn pipelines (`.joblib`), Markdown audit reports (`.md`), and feature importance tables (`.csv`).
* **Unsupervised:** Download row-level cluster/anomaly assignments (`.csv`) and fitted model/transformer bundles (`.joblib`).

---

## 🛠️ Technology Stack

| Component | Library / Framework | Version Range | Purpose |
| :--- | :--- | :--- | :--- |
| **Interface** | Streamlit | `>=1.36` | Responsive web UI, interactive controls, metric displays |
| **ML Engine** | scikit-learn | `>=1.4` | Pipelines, estimators, cross-validation, search spaces, metrics |
| **Data Processing** | pandas | `>=2.1` | DataFrame manipulation, EDA profiling, table rendering |
| **Numerical Ops** | NumPy | `>=1.26` | Array manipulations, distribution calculations |
| **Scientific Utilities**| SciPy | `>=1.11` | Statistical distributions for hyperparameter sampling |
| **Visualizations** | Matplotlib | `>=3.8` | Monochrome diagnostic plots (Agg headless backend) |
| **Excel Ingestion** | openpyxl | `>=3.1` | Parsing `.xlsx` and `.xls` workbook uploads |
| **Serialization** | Joblib | `>=1.3` | Model and preprocessor persistence (`.joblib`) |

---

## ⚡ Quickstart

### Prerequisites
* Python 3.10, 3.11, or 3.12 installed on your system.
* Git installed.

### Windows (PowerShell)

```powershell
# 1. Clone repository and navigate to root
git clone https://github.com/nishantrajanand017-beep/Model-Forge-.git
cd Model-Forge-

# 2. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1
# Note: If PowerShell script execution is restricted, run:
# Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Rebuild bundled sample datasets
python scripts\make_sample_data.py

# 5. Launch the application
streamlit run app.py
```

### macOS / Linux (Bash or Zsh)

```bash
# 1. Clone repository and navigate to root
git clone https://github.com/nishantrajanand017-beep/Model-Forge-.git
cd Model-Forge-

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Rebuild bundled sample datasets
python scripts/make_sample_data.py

# 5. Launch the application
streamlit run app.py
```

After launching, open the local URL printed in the terminal (typically `http://localhost:8501`). You can upload your own tabular data or test using the bundled sample datasets (`sample_data/iris_classification.csv` or `sample_data/diabetes_regression.csv`).

---

## 🧪 Verification & Headless Testing

Model Forge includes standalone test scripts for continuous verification without opening the web browser:

### Headless Smoke Tests
```powershell
# Supervised workflow smoke test (Classification on Iris, Regression on Diabetes)
python scripts\smoke_test.py

# Unsupervised workflow smoke test (Clustering, Anomaly Detection, and PCA)
python scripts\smoke_test_unsupervised.py
```

### Focused Unit Test Scripts
```powershell
# Data Quality Inspector tests (missingness, duplicates, IQR outliers, non-mutation)
python scripts\test_data_quality_inspector.py

# Clustering Visualizer tests (2D projection, PCA coordinates, label alignment)
python scripts\test_clustering_visualizer.py

# Feature Importance tests (tree/linear extraction, permutation fallback)
python scripts\test_feature_importance.py

# Prediction Playground tests (schema inference, single prediction, probability output)
python scripts\test_prediction_playground.py
```

> **Testing Scope Note:** These smoke tests and focused scripts verify that underlying pipelines, algorithms, and transformations execute correctly across known scenarios. They do not constitute exhaustive unit test coverage or replace formal production test harnesses.

---

## 🗂️ Project Structure

```
Model-Forge-/
├── .gitignore                      # Git ignore patterns (virtualenvs, secrets, cache)
├── .streamlit/
│   └── config.toml                 # Streamlit theme and client configuration
├── LICENSE                         # MIT License
├── README.md                       # Project documentation
├── app.py                          # Streamlit application entry point
├── requirements.txt                # Pinned dependency requirements
├── sample_data/                    # Bundled benchmark datasets
│   ├── diabetes_regression.csv     # Continuous regression sample
│   └── iris_classification.csv     # Multiclass classification sample
├── automl/                         # Pure Python AutoML core library
│   ├── __init__.py                 # Library entry point and exports
│   ├── core.py                     # Supervised tournament orchestration & CV
│   ├── evaluation.py               # Metrics, confusion matrix, feature importance
│   ├── explain.py                  # Decision generator, victory rationale & advice
│   ├── models.py                   # Model definitions & hyperparameter distributions
│   ├── plots.py                    # Matplotlib charts (monochrome theme, Agg backend)
│   ├── prediction.py               # Prediction Playground schema & inference
│   ├── preprocessing.py            # Leakage-safe ColumnTransformer pipelines
│   ├── profiling.py                # Per-column EDA diagnostic profiling
│   ├── quality.py                  # Data Quality Inspector & IQR outlier detection
│   ├── report.py                   # Markdown audit report builder
│   ├── ui.py                       # CSS theme, cards, headers, DOT decision graphs
│   └── unsupervised.py             # Clustering, anomaly detection & PCA workflows
└── scripts/                        # Headless verification and test utilities
    ├── get_sports_data.py          # Data ingestion utility for external benchmark data
    ├── make_sample_data.py         # Utility to generate bundled sample datasets
    ├── smoke_test.py               # Supervised end-to-end headless smoke test
    ├── smoke_test_unsupervised.py  # Unsupervised end-to-end headless smoke test
    ├── test_clustering_visualizer.py     # Unit test for 2D clustering visualizer
    ├── test_data_quality_inspector.py    # Unit test for quality inspector
    ├── test_feature_importance.py        # Unit test for feature importance logic
    └── test_prediction_playground.py     # Unit test for interactive prediction logic
```

---

## 🤖 Supported Algorithm Portfolio

| Learning Paradigm | Task / Category | Algorithms Included | Default Optimization / Metric |
| :--- | :--- | :--- | :--- |
| **Supervised** | Classification | Logistic Regression, Random Forest, HistGradientBoosting, K-Nearest Neighbors | ROC-AUC (binary), Weighted F1 (multiclass) |
| **Supervised** | Regression | Ridge Regression, Random Forest, HistGradientBoosting, K-Nearest Neighbors | R² score (MAE & RMSE reported) |
| **Unsupervised** | Clustering | K-Means, Agglomerative Clustering, DBSCAN, Gaussian Mixture | Silhouette Score, Davies-Bouldin, Calinski-Harabasz |
| **Unsupervised** | Anomaly Detection | Isolation Forest, Local Outlier Factor, One-Class SVM | Configurable contamination fraction |
| **Unsupervised** | Dimensionality Reduction | Principal Component Analysis (PCA) | Explained variance ratio (PC1, PC2) |

---

## ⚠️ Notes, Assumptions & Limitations

* **Tabular Focus:** Model Forge is built specifically for structured, tabular data (numeric, categorical, boolean). Unstructured data (free text, images, audio) is not supported.
* **Dataset Scale:** The local unsupervised pipeline caps uploads at 100,000 rows to maintain responsive local performance. Silhouette scoring is subsampled at 5,000 rows.
* **Anomaly Contamination Prior:** In unsupervised anomaly detection without labels, the contamination slider represents an assumed prior fraction of anomalies, not ground-truth detection accuracy.
* **Serialization Caution:** Model files exported as `.joblib` contain serialized Python objects. Only load `.joblib` files created by trusted sources.
* **Experimental Status:** This studio is an ongoing developer project. It is not advertised as enterprise-ready or hardened against malicious inputs.

---

<div align="center">

<sub>Built by <a href="https://github.com/nishantrajanand017-beep">Nishant Raj Anand</a> · Licensed under the <a href="LICENSE">MIT License</a></sub>

</div>

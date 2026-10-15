<div align="center">

# 🧠 AutoML Studio

<a href="https://github.com/hirakparekh/AutoML">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=600&size=24&pause=1000&color=F7931E&center=true&vCenter=true&width=620&lines=Spreadsheet+in.+Trained+model+out.;No+code.+No+cloud.+No+data+leaves+your+device.;Powered+entirely+by+scikit-learn." alt="Typing SVG" />
</a>

<p>
  <img src="https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white" />
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white" />
  <img src="https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white" />
  <img src="https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white" />
  <img src="https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white" />
  <img src="https://img.shields.io/badge/License-MIT-3DA639?logo=opensourceinitiative&logoColor=white" />
</p>

<em>A local, no-code machine-learning studio that turns a spreadsheet into a trained, tuned model — entirely on your own machine.</em>

</div>

---

## ✨ What it does

Upload a CSV or Excel file, point at the column you want to predict, and AutoML Studio runs the
full workflow on its own: it profiles the data, decides between classification and regression,
cleans and encodes the features, trains and compares several models, tunes the best one, and
reports the results — then lets you download the trained model. **No code, no cloud, no APIs, and
no data ever leaves your device.** Everything is computed locally with scikit-learn.

<!-- 📸 Add a screenshot/GIF of the app here: docs/demo.png -->

## 🚀 Features

| | Feature | Detail |
|---|---|---|
| 📊 | **Ingest & profile** | Load CSV/Excel, preview the data, profile every column (type, missing %, cardinality) |
| 🎯 | **Auto-detect the task** | Classification vs regression from the target column, with a manual override |
| 🧹 | **Leakage-safe preprocessing** | Median imputation + standardization for numbers, one-hot / ordinal encoding for categories — all inside a single scikit-learn `Pipeline` |
| 🏆 | **Model tournament** | Cross-validates a portfolio (Logistic/Ridge, Random Forest, Gradient Boosting, KNN) and ranks on a task-appropriate metric |
| 🎛️ | **Hyperparameter tuning** | Randomized search on the winning model |
| 📈 | **Honest evaluation** | Accuracy / F1 / ROC-AUC or R² / RMSE / MAE on a held-out test set, plus a confusion matrix or predicted-vs-actual plot and feature importance (with a permutation fallback) |
| 🌳 | **Explains its reasoning** | A visual decision tree shows every fork the pipeline faced, glows the branch it chose, and justifies each |
| 💾 | **Export** | Download the fitted model as a `.joblib` file and a Markdown report |

## ⚡ Quickstart

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1          # if blocked: Set-ExecutionPolicy -Scope Process Bypass
pip install -r requirements.txt
python scripts\make_sample_data.py    # optional: writes sample_data\*.csv
streamlit run app.py
```

Then open the URL it prints (usually <http://localhost:8501>), upload a dataset
(try `sample_data/iris_classification.csv` or `sample_data/diabetes_regression.csv`),
choose the target column, and click **Run AutoML**.

> On macOS / Linux the only difference is activating the venv with `source .venv/bin/activate`.

## 🔍 How it works

```
Upload → Profile → Detect task → Preprocess (impute / scale / encode)
       → Cross-validated tournament → Tune best model
       → Evaluate on held-out test → Explain decisions → Export
```

The engine in `automl/` is pure Python with no UI dependency, so it's testable and reusable on its
own; `app.py` is a thin Streamlit front end over it.

## 🗂️ Project layout

```
app.py                 Streamlit UI
automl/
  core.py              orchestration: detect task → split → tournament → tune → evaluate
  preprocessing.py     ColumnTransformer (impute / scale / encode)
  models.py            model portfolios + hyperparameter search spaces
  evaluation.py        metrics, feature importance, permutation fallback
  explain.py           builds the "how the pipeline decided" decisions
  plots.py             matplotlib figures
  profiling.py         per-column data profile
  report.py            downloadable Markdown summary
  ui.py                theming, hero, flowchart, decision-tree rendering
scripts/
  make_sample_data.py  generate bundled sample datasets
  smoke_test.py        headless end-to-end check
```

## 🛠️ Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikit-learn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?logo=streamlit&logoColor=white)
![pandas](https://img.shields.io/badge/pandas-150458?logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?logo=python&logoColor=white)

---

<div align="center">
<sub>Built by <a href="https://github.com/nishantrajanand017-beep">Nishant Raj Anand</a> · Licensed under MIT</sub>
</div>


## Unsupervised AutoML

Model Forge includes a second **Unsupervised AutoML** mode. Unlike supervised learning, it does not require a target column. Upload a CSV or Excel file, select input columns and choose one or more analysis families:

- **Clustering:** K-Means, Agglomerative Clustering, DBSCAN and Gaussian Mixture. Where defined, the leaderboard compares silhouette score, plus cluster count, noise points and run time.
- **Anomaly detection:** Isolation Forest, Local Outlier Factor and One-Class SVM. View flagged rows and download results. The configured contamination value is an assumption about the expected anomaly fraction; anomaly rate is not accuracy.
- **Dimensionality reduction:** PCA produces a two-component view (or one component when the data only has one usable feature) and reports explained variance.

The results tab exports row-level CSV output and a Joblib bundle containing fitted preprocessing and estimators. The app remains local; data is not sent to an external service by this feature. Use only trusted `.joblib` files because loading serialized models can execute code.

### Notes and limitations

- The current unsupervised workflow is for tabular data and caps a run at 100,000 rows to avoid unexpectedly heavy local workloads.
- Numeric values are median-imputed and standardized; categorical values are most-frequent-imputed, one-hot encoded and standardized.
- Clustering quality metrics are only available when an algorithm produces a valid multi-cluster assignment. DBSCAN noise points are excluded from those metrics.
- There is no universal "best unsupervised model" score. Model Forge recommends a clustering candidate by silhouette score where available, while keeping anomaly detection and PCA comparisons specific to their task.

"""SteelSense Lab: read-only Streamlit inference with five saved classifiers.

Important: this app NEVER fits/retrains a model or changes fitted scalers.
It evaluates the held-out sample or an uploaded labelled test CSV.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st

from evaluation import METRIC_COLUMNS, check_upload, evaluate_model

ROOT = Path(__file__).resolve().parent
METADATA_FILE = ROOT / "model_metadata.json"
SAMPLE_FILE = ROOT / "test_data.csv"

st.set_page_config(
    page_title="SteelSense | Defect Classification Lab", page_icon="⚙️",
    layout="wide", initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container{padding-top:1.55rem;max-width:1320px}
    .lab-header{background:linear-gradient(110deg,#123447,#166D72);color:#fff;
      padding:24px 28px;border-radius:15px;margin-bottom:20px}
    .lab-kicker{color:#ADE6D9;font-size:0.78rem;font-weight:700;letter-spacing:0.16em}
    .lab-header h1{color:#fff;font-size:2.1rem;margin:7px 0 7px 0}
    .lab-header p{color:#D9ECEC;font-size:0.96rem;margin:0}
    .mini-note{background:#EEF5F3;padding:12px 16px;border-left:4px solid #0F766E;
      border-radius:8px;margin-top:12px}
    div[data-testid='stMetric']{background:#fff;border:1px solid #DBE9E7;
      border-radius:11px;padding:12px 16px}
    </style>
    <div class="lab-header">
      <div class="lab-kicker">INDUSTRIAL QUALITY INTELLIGENCE / ML LAB</div>
      <h1>SteelSense: Seven-Fault Classifier</h1>
      <p>Compare five trained classifiers on labelled steel-plate test records,
      with class-level diagnostics and transparent evaluation.</p>
    </div>
    """, unsafe_allow_html=True,
)


@st.cache_resource(show_spinner=False)
def load_assets():
    if not METADATA_FILE.exists():
        raise FileNotFoundError("Training metadata is missing. Run 'python train_models.py' first.")
    metadata = json.loads(METADATA_FILE.read_text(encoding="utf-8"))
    models = {}
    for label, relative_path in metadata["model_files"].items():
        p = ROOT / relative_path
        if not p.is_file():
            raise FileNotFoundError(f"Missing saved model: {relative_path}. Run training first.")
        models[label] = joblib.load(p)  # Only load your own trusted .joblib files.
    return metadata, models


try:
    info, models = load_assets()
except (OSError, ValueError, KeyError) as err:
    st.error(f"Setup incomplete: {err}")
    st.info("Train models on the original UCI dataset, commit model/*.joblib and metadata, then deploy.")
    st.stop()

with st.sidebar:
    st.header("Experiment setup")
    st.caption("Trained classifiers are loaded from disk, never fitted on uploads.")
    active = st.selectbox("Choose a classification model", list(models), index=0)
    st.divider()
    st.markdown("**Test data input**")
    uploaded = st.file_uploader(
        "Upload a labelled test CSV", type=["csv"],
        help="Include all 27 feature columns and the Fault_Type target column.",
    )
    if SAMPLE_FILE.exists():
        st.download_button(
            "Download held-out sample test CSV", data=SAMPLE_FILE.read_bytes(),
            file_name="test_data.csv", mime="text/csv", use_container_width=True,
        )
    st.caption("Recommended: upload only test observations to respect Cloud free-tier limits.")
    st.divider()
    st.caption(f"Source: UCI dataset #198  ·  {info['total_rows']:,} rows  ·  "
               f"{len(info['feature_names'])} features  ·  {len(info['class_names'])} classes")

if uploaded is not None:
    if getattr(uploaded, "size", 0) > 5 * 1024 * 1024:
        st.error("Test CSV exceeds the 5 MB limit.")
        st.stop()
    try:
        raw = pd.read_csv(uploaded)
    except Exception as err:
        st.error(f"The uploaded CSV could not be parsed: {err}")
        st.stop()
    dataset_label = "Uploaded labelled CSV"
else:
    if not SAMPLE_FILE.exists():
        st.error("Neither a test CSV nor bundled test_data.csv is available.")
        st.stop()
    raw = pd.read_csv(SAMPLE_FILE)
    dataset_label = "Original held-out test CSV (bundled)"

try:
    test = check_upload(raw, info["feature_names"], info["target"], info["class_names"])
except ValueError as err:
    st.error(f"Invalid test data: {err}")
    st.stop()

missing_classes = sorted(set(info["class_names"]) - set(test[info["target"]].unique()))
if missing_classes:
    st.warning("Your uploaded CSV is missing these target categories: "
               + ", ".join(missing_classes)
               + ". AUC will be N/A; macro scores use all seven categories "
               "and assign zero to missing-class scores.")
X_test = test[info["feature_names"]]
y_test = test[info["target"]]

# Every displayed score is freshly recomputed on the same uploaded test CSV.
# This loop ONLY invokes predict / predict_proba on persisted models.
all_scores, details = {}, {}
for name, fitted_model in models.items():
    metrics, cm, report, predictions = evaluate_model(
        fitted_model, X_test, y_test, info["class_names"],
    )
    all_scores[name] = metrics
    details[name] = (cm, report, predictions)

st.subheader("Dataset inspection")
a, b, c, d = st.columns(4)
a.metric("Evaluation records", f"{len(test):,}")
b.metric("Input features", len(info["feature_names"]))
c.metric("Fault categories", len(info["class_names"]))
d.metric("Selected model", active)
st.caption(f"Currently evaluating: {dataset_label}")

with st.expander("Show test data schema and first 10 rows"):
    st.dataframe(test.head(10), use_container_width=True, hide_index=True)
    st.write("Fault type counts (this test CSV)")
    st.dataframe(
        test[info["target"]].value_counts().rename_axis("Fault type").reset_index(name="Records"),
        hide_index=True, use_container_width=True,
    )

st.divider()
st.subheader("Five-model comparison")
results = pd.DataFrame.from_dict(all_scores, orient="index")[METRIC_COLUMNS]
results.index.name = "Classifier"
st.dataframe(results.style.format({col: "{:.4f}" for col in METRIC_COLUMNS}, na_rep="N/A"),
             use_container_width=True)
st.caption("Macro-averaged Precision, Recall and F1; multiclass macro OvR AUC; multiclass MCC. "
           "AUC is unavailable if the uploaded CSV omits one or more training classes.")

st.divider()
st.subheader(f"Detailed evaluation: {active}")
selected_scores = all_scores[active]
for col, metric in zip(st.columns(6), METRIC_COLUMNS):
    score = selected_scores[metric]
    col.metric(metric, f"{score:.4f}" if score is not None else "N/A")

matrix, classification, predicted = details[active]
left, right = st.columns([1.12, 0.88], gap="large")
with left:
    st.markdown("**Confusion matrix**")
    fig, ax = plt.subplots(figsize=(8.1, 6.3))
    sns.heatmap(
        matrix, annot=True, fmt="d", cmap="GnBu", cbar=False,
        xticklabels=info["class_names"], yticklabels=info["class_names"],
        linewidths=0.35, linecolor="#d5e2e0", ax=ax,
    )
    ax.set_xlabel("Predicted category")
    ax.set_ylabel("Actual category")
    plt.setp(ax.get_xticklabels(), rotation=38, ha="right")
    fig.tight_layout()
    st.pyplot(fig, clear_figure=True)
with right:
    st.markdown("**Classification report by fault**")
    rows = [
        {"Fault type": label, "Precision": classification[label]["precision"],
         "Recall": classification[label]["recall"], "F1": classification[label]["f1-score"],
         "Support": int(classification[label]["support"])}
        for label in info["class_names"]
    ]
    report_frame = pd.DataFrame(rows).set_index("Fault type")
    st.dataframe(
        report_frame.style.format({"Precision": "{:.3f}", "Recall": "{:.3f}",
                                   "F1": "{:.3f}"}), use_container_width=True,
    )
    st.markdown(
        '<div class="mini-note">Reading the matrix: values on the diagonal '
        'are correct classifications; off-diagonal values are mistakes.</div>',
        unsafe_allow_html=True,
    )

st.subheader("Inspect individual predictions")
comparison = pd.DataFrame({
    "Actual fault": y_test.to_numpy(),
    "Predicted fault": predicted,
    "Correct?": (y_test.to_numpy() == predicted),
})
st.dataframe(comparison.head(30), use_container_width=True, hide_index=True)
st.download_button(
    "Download prediction review CSV", data=comparison.to_csv(index=False).encode(),
    file_name="steel_fault_predictions.csv", mime="text/csv",
)
st.caption("Educational demonstration only: outputs are not manufacturing safety decisions.")

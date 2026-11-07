"""Generate metric-grounded README sections after the real UCI training run.

The produced observations are a FIRST-DRAFT analysis. Read, validate and
rewrite them in your own words before academic submission.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def fmt(value):
    return "N/A" if value is None else f"{value:.4f}"


def observation(name, stats, cls_report):
    """Refer to observed scores rather than guessing model behaviour."""
    low_class = min(cls_report, key=lambda label: cls_report[label]["recall"])
    low_recall = cls_report[low_class]["recall"]
    prefix = {
        "Logistic Regression": "This linear classifier uses standardized numeric measurements.",
        "Decision Tree": "This model learns threshold rules and needs no feature scaling.",
        "KNN": "This distance-based model compares standardized feature vectors.",
        "Naive Bayes": "Gaussian Naive Bayes assumes conditional feature independence within each class.",
        "Support Vector Machine": "The RBF SVM models nonlinear boundaries after standardization.",
    }[name]
    return (
        f"{prefix} On the held-out set its accuracy was {fmt(stats['Accuracy'])} "
        f"and its macro-F1 was {fmt(stats['F1'])}; the MCC was {fmt(stats['MCC'])}. "
        f"Its weakest class-level recall was {low_class} ({low_recall:.3f}), "
        "so overall accuracy alone would not describe every fault equally well."
    )


def create_readme(meta: dict, repo_url: str = "[ADD GITHUB REPOSITORY URL]", app_url: str = "[ADD LIVE STREAMLIT URL]") -> str:
    classes = meta["class_names"]
    rows = ["| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    for model, scores in meta["metrics"].items():
        rows.append("| " + model + " | " + " | ".join(fmt(scores[m]) for m in
                    ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]) + " |")
    observed = ["| ML Model Name | Observation about model performance |",
                "|---|---|"]
    for model, scores in meta["metrics"].items():
        observed.append(f"| {model} | {observation(model, scores, meta['per_class_report'][model])} |")
    winner = meta["winner"]
    win = meta["metrics"][winner]
    observed.append(
        f"| **Overall winner** | **{winner}** achieved the highest held-out macro-F1 "
        f"({fmt(win['F1'])}) using the stated selection rule; its MCC was {fmt(win['MCC'])}. |"
    )
    distribution = "\n".join(f"| {cls} | {meta['class_distribution'][cls]} |" for cls in classes)
    return f"""# SteelSense: Comparative Classification of Steel Plate Surface Faults

## a. Problem statement

A steel plate can have several types of surface faults. Given measurements of a known fault region, the task is to classify the fault into one of seven categories. The project compares five standard machine-learning classifiers with a single consistent train/test split. The deployed application evaluates saved classifiers on **labelled test CSV files**; it does not learn from uploaded data. The aim is an educational model comparison, not an industrial safety system.

## b. Dataset description

- **Dataset:** Steel Plates Faults, UCI Machine Learning Repository (dataset ID 198).
- **Official source:** {meta['official_dataset_page']}
- **Dataset DOI:** {meta['doi']}
- **Creators / attribution:** M. Buscema, S. Terzi and W. Tastle (2010).
- **License:** {meta['source_license']}.
- **Instances:** {meta['total_rows']:,}; **predictor features:** {len(meta['feature_names'])}; **target classes:** {len(classes)}.
- **Missing values:** None in the original dataset, verified in the preprocessing script.
- **Target:** `{meta['target']}`. The original UCI file has seven binary target columns. Exactly one is active per row; the script validates this and converts the seven flags to one class label. None of the seven label flags is used as an input feature.
- **Feature types:** 27 numeric measurements describing regions, brightness, perimeter, shape and steel type.
- **Data division:** {meta['training_rows']:,} training rows and {meta['test_rows']:,} held-out test rows; stratified 75/25 split with `random_state={meta['split_seed']}`.

### Class counts in the source dataset

| Fault type | Number of records |
|---|---:|
{distribution}

The class sizes are unequal. Therefore, macro-averaged precision, recall and F1 are emphasized, with multiclass MCC also used to judge model consistency across categories.

## c. GitHub Repository Link

**GitHub repository:** {repo_url}

The repository contains `app.py`, `train_models.py`, `data_source.py`, `evaluation.py`, `requirements.txt`, `README.md`, `test_data.csv`, trained `.joblib` files in `model/`, `model_metadata.json`, `model_comparison.csv`, source data attribution and validation code. The trained preprocessing steps are stored inside each model pipeline.

**Live Streamlit application:** {app_url}

## d. Models used and evaluation

The five classifiers were fitted using the **same training rows** and evaluated on the **same untouched test rows**:

1. Logistic Regression: median imputation, StandardScaler, class-balanced logistic classifier.
2. Decision Tree Classifier: median imputation and depth-controlled class-balanced tree.
3. K-Nearest Neighbor Classifier: median imputation, StandardScaler and distance-weighted KNN.
4. Naive Bayes Classifier: GaussianNB after median imputation and StandardScaler.
5. Support Vector Machine: median imputation, StandardScaler and RBF-kernel probabilistic SVC.

### Model comparison (held-out test CSV)

{chr(10).join(rows)}

**Metric definitions:** Accuracy is the fraction of correct predictions. AUC is multiclass **one-vs-rest (OvR) ROC AUC, macro averaged** using probability outputs. Precision, Recall and F1 are **macro averaged**, giving each fault category equal weight. MCC is the single multiclass Matthews correlation coefficient. An AUC of N/A means the uploaded sample is missing a target class (the original held-out test set should contain all seven classes).

### Observations about model performance

{chr(10).join(observed)}

### Overall winner for this dataset

**{winner}** is the winner according to the pre-stated rule of highest **macro-F1**, using **MCC** and then **macro OvR AUC** as tie-breakers. This choice is based on the held-out test split, not a guarantee that the same algorithm will win on future industrial data. Differences across fault categories should also be examined with the confusion matrix and class-wise report.

## e. Streamlit application and deployment

The Streamlit app supports CSV upload (only test data), a dropdown for all five models, a six-metric comparison for all models, a detailed view of the selected model, a confusion matrix, a per-class classification report, and a downloadable prediction review. It loads the saved fitted pipelines; **uploaded data is never used to retrain**.

### Reproduce in BITS Virtual Lab

1. Clone or open the project in the **BITS Virtual Lab** (required by the assignment).
2. Install Python dependencies: `python -m pip install -r requirements.txt`.
3. Run training: `python train_models.py`. If the UCI download is blocked, download the official ZIP from UCI with a browser and run `python train_models.py --data-path '/path/to/steel+plates+faults.zip'`.
4. Check the generated `test_data.csv`, `model_comparison.csv`, five `model/*.joblib` files and `model_metadata.json`.
5. Regenerate README with the real results: `python build_readme.py --repo-url YOUR_GITHUB_LINK --app-url YOUR_STREAMLIT_LINK`.
6. Run locally with `streamlit run app.py` and upload **test_data.csv** to confirm metrics match the README comparison table.
7. Capture **one genuine screenshot showing the work running on BITS Virtual Lab** for inclusion in the PDF.
8. Commit all required files and deploy `app.py` via Streamlit Community Cloud, connected to the chosen GitHub repository.
9. Create the **single final submission PDF**, in the required order, using `python build_submission.py --github-url ... --app-url ... --screenshot ...`.

**Reproducibility:** The script records the train/test seed, class distribution, feature names, dependency version and SHA-256 of the held-out test CSV in `model_metadata.json`. The CSV can be uploaded to repeat the evaluation without fitting any model again.

**Academic integrity note:** Use the code and this auto-generated interpretation as learning material. Verify the reported metrics on BITS Virtual Lab, make original experiment and UI decisions, explain the reasoning yourself, and write the final observations in your own words before submission; the assignment prohibits direct AI copy-paste submissions.
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-url", default="[ADD GITHUB REPOSITORY URL]")
    parser.add_argument("--app-url", default="[ADD LIVE STREAMLIT URL]")
    opts = parser.parse_args()
    path = ROOT / "model_metadata.json"
    if not path.exists():
        raise SystemExit("model_metadata.json is missing. First run python train_models.py on UCI data.")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    result = create_readme(metadata, opts.repo_url, opts.app_url)
    (ROOT / "README.md").write_text(result, encoding="utf-8")
    print(f"README.md updated with {len(metadata['metrics'])} real model comparison rows.")


if __name__ == "__main__":
    main()

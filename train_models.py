"""Train exactly five classifiers on ONE official UCI dataset.

Run this script in BITS Virtual Lab and capture a genuine screenshot of
execution for the assignment's Virtual Lab mark.

Example:
    python train_models.py --data-path 'steel+plates+faults.zip'
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from data_source import (
    CLASS_NAMES, DATASET_DOI, FEATURE_NAMES, TARGET, UCI_DATASET_PAGE,
    load_uci_data,
)
from evaluation import METRIC_COLUMNS, evaluate_model

PROJECT_ROOT = Path(__file__).resolve().parent
SPLIT_SEED = 91
TEST_SHARE = 0.25


# Change one parameter at a time during your own lab experiments.
# We deliberately keep all five families and the same stratified split.
def candidate_models():
    return {
        "Logistic Regression": Pipeline([
            ("missing", SimpleImputer(strategy="median")),
            ("normalize", StandardScaler()),
            ("classifier", LogisticRegression(
                C=1.2, max_iter=4000, class_weight="balanced", random_state=SPLIT_SEED,
            )),
        ]),
        "Decision Tree": Pipeline([
            ("missing", SimpleImputer(strategy="median")),
            ("classifier", DecisionTreeClassifier(
                max_depth=10, min_samples_leaf=4, class_weight="balanced", random_state=SPLIT_SEED,
            )),
        ]),
        "KNN": Pipeline([
            ("missing", SimpleImputer(strategy="median")),
            ("normalize", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=9, weights="distance")),
        ]),
        "Naive Bayes": Pipeline([
            ("missing", SimpleImputer(strategy="median")),
            ("normalize", StandardScaler()),
            ("classifier", GaussianNB(var_smoothing=1e-8)),
        ]),
        "Support Vector Machine": Pipeline([
            ("missing", SimpleImputer(strategy="median")),
            ("normalize", StandardScaler()),
            ("classifier", SVC(
                C=3.0, kernel="rbf", gamma="scale", probability=True,
                class_weight="balanced", random_state=SPLIT_SEED,
            )),
        ]),
    }


def model_filename(name: str) -> str:
    return name.lower().replace(" ", "_") + ".joblib"


def run_training(data: pd.DataFrame, output: Path = PROJECT_ROOT) -> dict:
    """Fit on train only, then evaluate once on the untouched held-out set."""
    (output / "model").mkdir(parents=True, exist_ok=True)
    (output / "data").mkdir(parents=True, exist_ok=True)
    data.to_csv(output / "data" / "steel_plates_prepared.csv", index=False)

    train, test = train_test_split(
        data, test_size=TEST_SHARE, stratify=data[TARGET], random_state=SPLIT_SEED,
    )
    test = test.reset_index(drop=True)
    test.to_csv(output / "test_data.csv", index=False)

    results = {}
    per_class = {}
    confusion = {}
    paths = {}
    for name, estimator in candidate_models().items():
        estimator.fit(train[FEATURE_NAMES], train[TARGET])
        scores, matrix, report, _ = evaluate_model(
            estimator, test[FEATURE_NAMES], test[TARGET], CLASS_NAMES,
        )
        file = output / "model" / model_filename(name)
        joblib.dump(estimator, file, compress=3)
        results[name] = scores
        per_class[name] = {
            label: {"precision": report[label]["precision"],
                    "recall": report[label]["recall"],
                    "f1-score": report[label]["f1-score"],
                    "support": report[label]["support"]}
            for label in CLASS_NAMES
        }
        confusion[name] = matrix.tolist()
        paths[name] = "model/" + file.name
        print(f"{name:<25s} Accuracy={scores['Accuracy']:.4f}, "
              f"F1(macro)={scores['F1']:.4f}, MCC={scores['MCC']:.4f}")

    comparison = pd.DataFrame.from_dict(results, orient="index")[METRIC_COLUMNS]
    comparison.index.name = "ML Model Name"
    comparison.to_csv(output / "model_comparison.csv", float_format="%.6f")
    winner = sorted(results, key=lambda m: (
        results[m]["F1"], results[m]["MCC"], results[m]["AUC"] or -1.0,
    ), reverse=True)[0]

    metadata = {
        "dataset": "Steel Plates Faults (UCI #198)",
        "official_dataset_page": UCI_DATASET_PAGE,
        "doi": DATASET_DOI,
        "source_license": "Creative Commons Attribution 4.0 International (CC BY 4.0)",
        "total_rows": len(data),
        "training_rows": len(train),
        "test_rows": len(test),
        "test_fraction": TEST_SHARE,
        "split_seed": SPLIT_SEED,
        "split_type": "stratified, fixed random_state",
        "feature_names": FEATURE_NAMES,
        "target": TARGET,
        "class_names": CLASS_NAMES,
        "class_distribution": {k: int(v) for k, v in data[TARGET].value_counts().items()},
        "test_class_distribution": {k: int(v) for k, v in test[TARGET].value_counts().items()},
        "test_csv_sha256": hashlib.sha256((output / "test_data.csv").read_bytes()).hexdigest(),
        "model_files": paths,
        "metrics": results,
        "per_class_report": per_class,
        "confusion_matrices": confusion,
        "winner": winner,
        "winner_rule": "Highest macro-F1; MCC then macro OvR AUC break ties",
        "evaluation_definitions": {
            "Accuracy": "Correct classifications divided by total test records",
            "AUC": "Macro-average of seven one-vs-rest ROC AUC values (predict_proba)",
            "Precision": "Unweighted (macro) mean of seven class precision values",
            "Recall": "Unweighted (macro) mean of seven class recall values",
            "F1": "Unweighted (macro) mean of seven per-class F1 values",
            "MCC": "Single multiclass Matthews correlation coefficient",
        },
        "scikit_learn_version": sklearn.__version__,
    }
    (output / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"\nUCI records: {len(data)} | Features: {len(FEATURE_NAMES)}")
    print(f"Split: {len(train)} training / {len(test)} test (no overlap by construction)")
    print("\n", comparison.round(4).to_string())
    print(f"\nSelected winner (macro-F1 -> MCC -> AUC): {winner}")
    return metadata


def main():
    parser = argparse.ArgumentParser(description="Five models on UCI Steel Plates Faults")
    parser.add_argument("--data-path", default=None, help="Local UCI ZIP or prepared full CSV")
    args = parser.parse_args()
    data = load_uci_data(args.data_path)
    run_training(data)


if __name__ == "__main__":
    main()

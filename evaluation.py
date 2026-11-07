"""Uniform, multiclass-aware metrics used in training AND the Streamlit app."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score,
)

METRIC_COLUMNS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def evaluate_model(model, features: pd.DataFrame, actual: pd.Series, class_names: list[str]):
    """Return metrics, confusion matrix, per-class report and predictions.

    Macro scores treat all fault categories equally. Multiclass AUC is
    macro one-vs-rest computed with `predict_proba`; it is undefined if the
    test CSV omits one of the trained classes, in which case return None.
    """
    classes = list(model.classes_)
    if set(classes) != set(class_names):
        raise ValueError("Model class labels do not match the training metadata.")
    predicted = model.predict(features)
    probabilities = model.predict_proba(features)
    metrics = {
        "Accuracy": float(accuracy_score(actual, predicted)),
        "Precision": float(precision_score(actual, predicted, labels=class_names, average="macro", zero_division=0)),
        "Recall": float(recall_score(actual, predicted, labels=class_names, average="macro", zero_division=0)),
        "F1": float(f1_score(actual, predicted, labels=class_names, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(actual, predicted)),
    }
    # sklearn's AUC column ordering must match model.classes_, not arbitrary UI order.
    if set(actual.unique()) == set(classes):
        try:
            auc = roc_auc_score(
                actual, probabilities, multi_class="ovr", average="macro", labels=classes,
            )
            metrics["AUC"] = float(auc)
        except ValueError:
            metrics["AUC"] = None
    else:
        metrics["AUC"] = None
    result = {key: metrics.get(key) for key in METRIC_COLUMNS}
    matrix = confusion_matrix(actual, predicted, labels=class_names)
    report = classification_report(
        actual, predicted, labels=class_names, output_dict=True, zero_division=0,
    )
    return result, matrix, report, predicted


def check_upload(raw: pd.DataFrame, feature_names: list[str], target: str, classes: list[str]):
    """Validate labelled test CSV - never use labels as predictor features."""
    if raw.empty:
        raise ValueError("CSV is empty. Upload labelled test rows, not only headers.")
    if len(raw) > 2500:
        raise ValueError("Upload test data only (maximum 2,500 rows), not the whole training set.")
    if not raw.columns.is_unique:
        raise ValueError("CSV contains duplicate column headers.")
    missing = [c for c in feature_names + [target] if c not in raw.columns]
    if missing:
        raise ValueError(f"CSV is missing required feature/target columns: {', '.join(missing)}")
    parsed = raw[feature_names + [target]].copy()
    for column in feature_names:
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
    bad = parsed[feature_names].isna() | ~np.isfinite(parsed[feature_names].to_numpy(dtype=float))
    if bad.to_numpy().any():
        raise ValueError("Predictor columns must contain finite numbers only; check blank/text values.")
    unknown = sorted(set(parsed[target].dropna().unique()) - set(classes))
    if unknown or parsed[target].isna().any():
        raise ValueError(f"Fault_Type contains missing or unknown categories: {unknown}")
    return parsed

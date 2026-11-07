"""Preflight that real training outputs are present and internally consistent."""
from pathlib import Path
import json

import joblib
import pandas as pd

from data_source import CLASS_NAMES, FEATURE_NAMES, TARGET
from evaluation import evaluate_model

ROOT = Path(__file__).resolve().parent


def verify():
    required = [ROOT / "app.py", ROOT / "train_models.py", ROOT / "README.md",
                ROOT / "requirements.txt", ROOT / "test_data.csv", ROOT / "model_metadata.json",
                ROOT / "model_comparison.csv", ROOT / "data" / "steel_plates_prepared.csv"]
    missing = [str(p.relative_to(ROOT)) for p in required if not p.is_file()]
    if missing:
        print("INCOMPLETE: Missing generated project files:")
        print("\n".join(" - " + p for p in missing))
        print("Run python train_models.py on the ORIGINAL UCI data in BITS Lab.")
        return False
    meta = json.loads((ROOT / "model_metadata.json").read_text(encoding="utf-8"))
    raw = pd.read_csv(ROOT / "data" / "steel_plates_prepared.csv")
    if len(raw) != 1941 or raw.columns.tolist() != FEATURE_NAMES + [TARGET]:
        raise ValueError("Source data CSV does not match expected official 1,941 x 28 schema.")
    test = pd.read_csv(ROOT / "test_data.csv")
    if len(test) != meta["test_rows"]:
        raise ValueError("test_data.csv size does not match metadata.")
    if set(test[TARGET].unique()) != set(CLASS_NAMES):
        raise ValueError("Test data must contain all seven fault classes.")
    if len(meta["model_files"]) != 5:
        raise ValueError("Expected exactly five model artifacts.")
    for label, relative in meta["model_files"].items():
        file = ROOT / relative
        if not file.exists():
            raise ValueError(f"Missing fitted model: {relative}")
        loaded = joblib.load(file)
        current, *_ = evaluate_model(loaded, test[FEATURE_NAMES], test[TARGET], CLASS_NAMES)
        for metric, expected in meta["metrics"][label].items():
            if expected is None or current[metric] is None:
                if expected != current[metric]:
                    raise ValueError(f"Unexpected {label} {metric}")
            elif abs(expected-current[metric]) > 1e-8:
                raise ValueError(f"Stored and recomputed {label} {metric} differ.")
    content = (ROOT / "README.md").read_text(encoding="utf-8")
    if "[ADD" in content or "Not yet measured" in content:
        print("INCOMPLETE: README still has missing links or results.")
        return False
    print("PASS: Original dataset size/schema, 5 model files, all 6 computed metrics, and README results.")
    print("EXTERNAL CHECK STILL REQUIRED: public GitHub + live Streamlit URLs and one authentic BITS Lab screenshot.")
    return True


if __name__ == "__main__":
    raise SystemExit(0 if verify() else 1)

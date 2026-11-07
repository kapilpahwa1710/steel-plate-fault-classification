"""Download and validate the UCI Steel Plates Faults dataset (ID 198).

The UCI release stores labels as seven mutually exclusive indicator columns.
This module retains all 27 explanatory variables and creates one multiclass
label. It never uses the target indicator columns as predictor inputs.
"""
from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zipfile import ZipFile

import numpy as np
import pandas as pd

UCI_DATASET_PAGE = "https://archive.ics.uci.edu/dataset/198/steel+plates+faults"
UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/198/steel+plates+faults.zip"
DATASET_DOI = "https://doi.org/10.24432/C5J88N"

FEATURE_NAMES = [
    "X_Minimum", "X_Maximum", "Y_Minimum", "Y_Maximum", "Pixels_Areas",
    "X_Perimeter", "Y_Perimeter", "Sum_of_Luminosity",
    "Minimum_of_Luminosity", "Maximum_of_Luminosity", "Length_of_Conveyer",
    "TypeOfSteel_A300", "TypeOfSteel_A400", "Steel_Plate_Thickness",
    "Edges_Index", "Empty_Index", "Square_Index", "Outside_X_Index",
    "Edges_X_Index", "Edges_Y_Index", "Outside_Global_Index", "LogOfAreas",
    "Log_X_Index", "Log_Y_Index", "Orientation_Index", "Luminosity_Index",
    "SigmoidOfAreas",
]
CLASS_NAMES = [
    "Pastry", "Z_Scratch", "K_Scatch", "Stains", "Dirtiness", "Bumps",
    "Other_Faults",
]
TARGET = "Fault_Type"
EXPECTED_ROWS = 1941


def _from_uci_zip(data: bytes) -> pd.DataFrame:
    """Parse official `Faults.NNA` space-separated 34-column matrix."""
    with ZipFile(BytesIO(data)) as archive:
        names = [name for name in archive.namelist() if name.lower().endswith("faults.nna")]
        if len(names) != 1:
            raise ValueError(f"Expected one Faults.NNA inside UCI ZIP; found: {names}")
        raw = archive.read(names[0])

    frame = pd.read_csv(BytesIO(raw), sep=r"\s+", header=None)
    if frame.shape[1] != len(FEATURE_NAMES) + len(CLASS_NAMES):
        raise ValueError(f"UCI file must have 34 columns; got {frame.shape[1]}")
    frame.columns = FEATURE_NAMES + CLASS_NAMES
    flags = frame[CLASS_NAMES].to_numpy(dtype=float)
    if not np.isin(flags, [0, 1]).all() or not (flags.sum(axis=1) == 1).all():
        raise ValueError("Each row must have exactly one active fault indicator (0/1).")
    frame[TARGET] = np.array(CLASS_NAMES)[flags.argmax(axis=1)]
    return frame[FEATURE_NAMES + [TARGET]]


def _from_prepared_csv(csv_path: Path) -> pd.DataFrame:
    """Read an explicitly provided CSV with 27 predictors and one target label."""
    frame = pd.read_csv(csv_path)
    if TARGET not in frame.columns and "Fault" in frame.columns:
        frame = frame.rename(columns={"Fault": TARGET})
    missing = [name for name in FEATURE_NAMES + [TARGET] if name not in frame.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")
    # This is an explicit spelling harmonization for a known mirror variant.
    frame[TARGET] = frame[TARGET].replace({"K_Scratch": "K_Scatch"})
    return frame[FEATURE_NAMES + [TARGET]]


def verify_dataset(frame: pd.DataFrame, require_full: bool = True) -> pd.DataFrame:
    """Fail early for truncated data, missing values, bad labels or changed schema."""
    frame = frame.copy()
    if frame.columns.tolist() != FEATURE_NAMES + [TARGET]:
        raise ValueError("Dataset columns are not the expected 27 features + Fault_Type.")
    if require_full and len(frame) != EXPECTED_ROWS:
        raise ValueError(f"Expected {EXPECTED_ROWS} observations from UCI; got {len(frame)}")
    if len(frame) < 500 or len(FEATURE_NAMES) < 12:
        raise ValueError("Assignment dataset-size requirement not met.")
    for col in FEATURE_NAMES:
        frame[col] = pd.to_numeric(frame[col], errors="raise")
    if frame[FEATURE_NAMES].isna().any().any() or not np.isfinite(frame[FEATURE_NAMES].to_numpy(dtype=float)).all():
        raise ValueError("All 27 predictor columns must contain finite numeric values.")
    if frame[TARGET].isna().any() or set(frame[TARGET].unique()) != set(CLASS_NAMES):
        raise ValueError("Expected each of the seven known fault categories in the target.")
    if (frame[TARGET].value_counts() < 2).any():
        raise ValueError("Every fault type needs at least two rows for stratified splitting.")
    return frame


def load_uci_data(local_file: str | None = None) -> pd.DataFrame:
    """Download the official archive, or use a manually downloaded ZIP/CSV.

    For restricted BITS Lab networking, download the ZIP from the UCI page
    using the browser and pass --data-path /path/to/steel+plates+faults.zip.
    """
    if local_file:
        path = Path(local_file).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"Dataset file not found: {path}")
        data = _from_uci_zip(path.read_bytes()) if path.suffix.lower() == ".zip" else _from_prepared_csv(path)
    else:
        try:
            req = Request(UCI_ZIP_URL, headers={"User-Agent": "ML-Assignment-Study/1.0"})
            with urlopen(req, timeout=40) as connection:
                data = _from_uci_zip(connection.read())
        except Exception as error:
            raise RuntimeError(
                "Cannot fetch the UCI dataset. Download 'Steel Plates Faults' ZIP from "
                f"{UCI_DATASET_PAGE}, then run: python train_models.py --data-path "
                "'/path/to/steel+plates+faults.zip'."
            ) from error
    return verify_dataset(data)

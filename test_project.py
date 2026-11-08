"""Offline QA using synthetic test fixtures. NOT real assignment model scores."""
import json
import sys
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_source import (CLASS_NAMES, FEATURE_NAMES, TARGET, _from_uci_zip,
                         verify_dataset)
from evaluation import check_upload, evaluate_model
from train_models import run_training


def fake_uci_frame():
    values, label_ids = make_classification(
        n_samples=1941, n_features=27, n_informative=18, n_redundant=4,
        n_repeated=0, n_classes=7, n_clusters_per_class=1,
        weights=[.08, .10, .16, .04, .04, .20, .38],
        flip_y=0, random_state=31,
    )
    return pd.DataFrame(values, columns=FEATURE_NAMES).assign(
        **{TARGET: np.asarray(CLASS_NAMES)[label_ids]}
    )


class DatasetParsing(unittest.TestCase):
    def test_original_seven_flags_are_excluded_from_features(self):
        sample = np.array([[1.5] * 27 + [1, 0, 0, 0, 0, 0, 0],
                           [2.0] * 27 + [0, 0, 0, 0, 1, 0, 0]])
        lines = '\n'.join(' '.join(str(x) for x in row) for row in sample)
        mem = BytesIO()
        with ZipFile(mem, 'w') as z:
            z.writestr('Faults.NNA', lines)
        parsed = _from_uci_zip(mem.getvalue())
        self.assertEqual(parsed.shape, (2, 28))
        self.assertEqual(parsed[TARGET].tolist(), [CLASS_NAMES[0], CLASS_NAMES[4]])

    def test_dataset_requirements(self):
        sample = verify_dataset(fake_uci_frame())
        self.assertEqual(sample.shape, (1941, 28))

    def test_upload_rejects_missing_label_and_bad_types(self):
        frame = fake_uci_frame().head(10)
        with self.assertRaisesRegex(ValueError, 'missing'):
            check_upload(frame.drop(columns=[TARGET]), FEATURE_NAMES, TARGET, CLASS_NAMES)
        frame = frame.copy()
        frame[FEATURE_NAMES[0]] = frame[FEATURE_NAMES[0]].astype(object)
        frame.loc[frame.index[0], FEATURE_NAMES[0]] = 'bad'
        with self.assertRaisesRegex(ValueError, 'finite'):
            check_upload(frame, FEATURE_NAMES, TARGET, CLASS_NAMES)


class ModelPipeline(unittest.TestCase):
    def test_five_classifiers_saved_evaluated_without_refitting(self):
        frame = verify_dataset(fake_uci_frame())
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            meta = run_training(frame, output=root)
            self.assertEqual(meta['total_rows'], 1941)
            self.assertEqual(len(meta['model_files']), 5)
            self.assertEqual(len(meta['class_names']), 7)
            test = pd.read_csv(root / 'test_data.csv')
            self.assertEqual(test.shape[1], 28)
            self.assertGreaterEqual(len(test), 400)
            csv_metrics = pd.read_csv(root / 'model_comparison.csv')
            self.assertEqual(csv_metrics.shape, (5, 7))
            for title, location in meta['model_files'].items():
                model = joblib.load(root / location)
                scores, mat, report, predictions = evaluate_model(
                    model, test[FEATURE_NAMES], test[TARGET], CLASS_NAMES,
                )
                self.assertEqual(mat.sum(), len(test))
                self.assertEqual(len(predictions), len(test))
                self.assertAlmostEqual(scores['F1'], meta['metrics'][title]['F1'], places=10)
                self.assertIsNotNone(scores['AUC'])
                self.assertTrue((0 <= scores['F1'] <= 1))
                self.assertTrue((-1 <= scores['MCC'] <= 1))
            self.assertTrue((root / 'model_metadata.json').is_file())


if __name__ == '__main__':
    unittest.main(verbosity=2)

# SteelSense: Comparative Classification of Steel Plate Surface Faults

## a. Problem statement

A steel plate can have several types of surface faults. Given measurements of a known fault region, the task is to classify the fault into one of seven categories. The project compares five standard machine-learning classifiers with a single consistent train/test split. The deployed application evaluates saved classifiers on **labelled test CSV files**; it does not learn from uploaded data. The aim is an educational model comparison, not an industrial safety system.

## b. Dataset description

- **Dataset:** Steel Plates Faults, UCI Machine Learning Repository (dataset ID 198).
- **Official source:** https://archive.ics.uci.edu/dataset/198/steel+plates+faults
- **Dataset DOI:** https://doi.org/10.24432/C5J88N
- **Creators / attribution:** M. Buscema, S. Terzi and W. Tastle (2010).
- **License:** Creative Commons Attribution 4.0 International (CC BY 4.0).
- **Instances:** 1,941; **predictor features:** 27; **target classes:** 7.
- **Missing values:** None in the original dataset, verified in the preprocessing script.
- **Target:** `Fault_Type`. The original UCI file has seven binary target columns. Exactly one is active per row; the script validates this and converts the seven flags to one class label. None of the seven label flags is used as an input feature.
- **Feature types:** 27 numeric measurements describing regions, brightness, perimeter, shape and steel type.
- **Data division:** 1,455 training rows and 486 held-out test rows; stratified 75/25 split with `random_state=91`.

### Class counts in the source dataset

| Fault type | Number of records |
|---|---:|
| Pastry | 158 |
| Z_Scratch | 190 |
| K_Scatch | 391 |
| Stains | 72 |
| Dirtiness | 55 |
| Bumps | 402 |
| Other_Faults | 673 |

The class sizes are unequal. Therefore, macro-averaged precision, recall and F1 are emphasized, with multiclass MCC also used to judge model consistency across categories.

## c. GitHub Repository Link

**GitHub repository:** https://github.com/kapilpahwa1710/steel-plate-fault-classification

The repository contains `app.py`, `train_models.py`, `data_source.py`, `evaluation.py`, `requirements.txt`, `README.md`, `test_data.csv`, trained `.joblib` files in `model/`, `model_metadata.json`, `model_comparison.csv`, source data attribution and validation code. The trained preprocessing steps are stored inside each model pipeline.

**Live Streamlit application:** [ADD LIVE STREAMLIT URL]

## d. Models used and evaluation

The five classifiers were fitted using the **same training rows** and evaluated on the **same untouched test rows**:

1. Logistic Regression: median imputation, StandardScaler, class-balanced logistic classifier.
2. Decision Tree Classifier: median imputation and depth-controlled class-balanced tree.
3. K-Nearest Neighbor Classifier: median imputation, StandardScaler and distance-weighted KNN.
4. Naive Bayes Classifier: GaussianNB after median imputation and StandardScaler.
5. Support Vector Machine: median imputation, StandardScaler and RBF-kernel probabilistic SVC.

### Model comparison (held-out test CSV)

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.6420 | 0.9237 | 0.6069 | 0.7370 | 0.6428 | 0.5758 |
| Decision Tree | 0.6543 | 0.8641 | 0.6641 | 0.7239 | 0.6806 | 0.5797 |
| KNN | 0.7119 | 0.9332 | 0.7547 | 0.7256 | 0.7316 | 0.6324 |
| Naive Bayes | 0.5905 | 0.9105 | 0.6171 | 0.7039 | 0.6242 | 0.5263 |
| Support Vector Machine | 0.7181 | 0.9445 | 0.7286 | 0.7883 | 0.7507 | 0.6521 |

**Metric definitions:** Accuracy is the fraction of correct predictions. AUC is multiclass **one-vs-rest (OvR) ROC AUC, macro averaged** using probability outputs. Precision, Recall and F1 are **macro averaged**, giving each fault category equal weight. MCC is the single multiclass Matthews correlation coefficient. An AUC of N/A means the uploaded sample is missing a target class (the original held-out test set should contain all seven classes).

### Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | This linear classifier uses standardized numeric measurements. On the held-out set its accuracy was 0.6420 and its macro-F1 was 0.6428; the MCC was 0.5758. Its weakest class-level recall was Other_Faults (0.375), so overall accuracy alone would not describe every fault equally well. |
| Decision Tree | This model learns threshold rules and needs no feature scaling. On the held-out set its accuracy was 0.6543 and its macro-F1 was 0.6806; the MCC was 0.5797. Its weakest class-level recall was Other_Faults (0.429), so overall accuracy alone would not describe every fault equally well. |
| KNN | This distance-based model compares standardized feature vectors. On the held-out set its accuracy was 0.7119 and its macro-F1 was 0.7316; the MCC was 0.6324. Its weakest class-level recall was Pastry (0.359), so overall accuracy alone would not describe every fault equally well. |
| Naive Bayes | Gaussian Naive Bayes assumes conditional feature independence within each class. On the held-out set its accuracy was 0.5905 and its macro-F1 was 0.6242; the MCC was 0.5263. Its weakest class-level recall was Other_Faults (0.238), so overall accuracy alone would not describe every fault equally well. |
| Support Vector Machine | The RBF SVM models nonlinear boundaries after standardization. On the held-out set its accuracy was 0.7181 and its macro-F1 was 0.7507; the MCC was 0.6521. Its weakest class-level recall was Other_Faults (0.536), so overall accuracy alone would not describe every fault equally well. |
| **Overall winner** | **Support Vector Machine** achieved the highest held-out macro-F1 (0.7507) using the stated selection rule; its MCC was 0.6521. |

### Overall winner for this dataset

**Support Vector Machine** is the winner according to the pre-stated rule of highest **macro-F1**, using **MCC** and then **macro OvR AUC** as tie-breakers. This choice is based on the held-out test split, not a guarantee that the same algorithm will win on future industrial data. Differences across fault categories should also be examined with the confusion matrix and class-wise report.

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

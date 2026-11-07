# SteelSense: Comparative Classification of Steel Plate Surface Faults

> **WORKING REFERENCE - NOT READY TO SUBMIT.** This README is a study scaffold.
> Run the project on the official UCI dataset in BITS Virtual Lab, then run
> `python build_readme.py --repo-url ... --app-url ...` to replace the missing
> experimental results with **real** measurements. Add your own explanations.

## a. Problem statement

Compare five multiclass machine-learning models for classifying observed steel-plate surface faults into seven known categories. An interactive Streamlit app evaluates saved classifiers on a labelled test CSV and shows the required metrics, confusion matrix and classification report, without fitting on the test set.

## b. Dataset description

- **Name:** Steel Plates Faults, UCI Machine Learning Repository, dataset ID 198.
- **Official dataset:** https://archive.ics.uci.edu/dataset/198/steel+plates+faults
- **DOI:** https://doi.org/10.24432/C5J88N
- **Attribution:** M. Buscema, S. Terzi and W. Tastle (2010); CC BY 4.0.
- **Instances:** 1,941; **features:** 27; **classes:** 7.
- **Class labels:** Pastry, Z_Scratch, K_Scatch, Stains, Dirtiness, Bumps, Other_Faults.
- **Features:** numerical attributes describing region boundaries, perimeter, areas, luminosity, plate properties and shape indices.
- **Target column in this project:** `Fault_Type`, created from the seven mutually exclusive fault-label columns in the UCI original. The original indicator columns are never included as predictor features.
- **Missing values:** None according to UCI. The script independently checks completeness.
- **Experiment:** Stratified train/test split, 75%/25%, random seed 91. Results must be generated on BITS Virtual Lab.

## c. GitHub Repository Link

**GitHub repository:** [ADD GITHUB REPOSITORY URL]

**Live Streamlit app:** [ADD LIVE STREAMLIT URL]

The GitHub repository must contain source code, requirements, README, held-out test CSV, model metadata, and all five saved `.joblib` pipelines.

## d. Models used: comparison table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured |
| Decision Tree | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured |
| KNN | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured |
| Naive Bayes | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured |
| Support Vector Machine | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured | Not yet measured |

**Definitions:** Precision, Recall and F1 are macro-averaged for the seven fault classes. AUC is multiclass macro one-vs-rest based on probabilities. MCC is the multiclass Matthews correlation coefficient. All scores must come from the same real UCI held-out test records.

### Observations about model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | [ADD METRIC-BASED OBSERVATION AFTER UCI RUN] |
| Decision Tree | [ADD METRIC-BASED OBSERVATION AFTER UCI RUN] |
| KNN | [ADD METRIC-BASED OBSERVATION AFTER UCI RUN] |
| Naive Bayes | [ADD METRIC-BASED OBSERVATION AFTER UCI RUN] |
| Support Vector Machine | [ADD METRIC-BASED OBSERVATION AFTER UCI RUN] |
| **Overall winner for your dataset?** | [DECIDE FROM REAL TEST RESULTS: HIGHEST MACRO-F1, MCC AS TIE-BREAKER] |

## e. Streamlit app, deployment and reproducibility

- `python -m pip install -r requirements.txt`
- `python train_models.py` or `python train_models.py --data-path '/path/to/steel+plates+faults.zip'` if BITS Lab blocks the UCI download.
- `python -m unittest discover -s tests -v` for offline technical checks; these use **synthetic fixtures** and do not produce submission results.
- `python build_readme.py --repo-url 'https://github.com/ACCOUNT/REPOSITORY' --app-url 'https://YOUR-APP.streamlit.app'`
- `streamlit run app.py` to check all four required Streamlit features.
- Capture **one genuine screenshot** in BITS Virtual Lab after executing the model training; check the terminal output or notebook results are readable.
- Push the source code, README, actual test_data.csv, models and metadata to GitHub.
- Deploy using Streamlit Community Cloud. Set repository, branch, and entry-point `app.py`.
- Install PDF-only dependencies before building the submission: `python -m pip install -r requirements-report.txt`.
- `python build_submission.py --github-url 'https://github.com/ACCOUNT/REPOSITORY' --app-url 'https://YOUR-APP.streamlit.app' --screenshot '/path/to/BITS-lab-proof.png'` builds the ONE final ordered PDF.

**Academic integrity:** The handout explicitly says AI is permitted only for learning support, not direct copy-paste submissions. Verify the metrics on the actual UCI data yourself, personalize code and UI meaningfully, and write the final interpretation in your own words.

# AtyAN

Stice et al., Predicting Risk Factors of Atypical Anorexia

## Notebook summary

The original monolithic notebooks have been decomposed into focused workflows that make it easier to audit each analytical step and to run them outside of Colab. Every notebook loads `BP1234-ONSET.csv` directly from the repository and shares a single preprocessing module (`analysis_utils.py`).

### `AtyAN_ANOVA.ipynb`
- Recreates the Aim 1 baseline contrasts (AAN vs. No AAN, weight-loss controls, and obese comparators).
- Includes the disorder-level comparisons (AAN vs. AN/BN/BED/PU; full and partial) together with BH-FDR q-values.

### `AtyAN_UnivariateModels.ipynb`
- Merges the original univariate onset-prediction workflow with the persistence/remission cohort so all single-feature analyses live in one place.
- Recreates the onset prediction task (dropping baseline AN/AAN diagnoses and labeling any mBMI-defined onset across waves 1–6) and the cleaned persistence vs. remission labeling before running the logistic regressions.

### `AtyAN_MultivariateModels.ipynb`
- Reuses the persistence cohort and evaluates the multivariate models (logistic regression, calibrated linear SVM, gradient boosting, random forest, and the BRF-inspired balanced forest).
- Surfaces cross-validated metrics and the tree-based feature importances while skipping the deprecated classification-tree heuristics.

## Shared utilities

`analysis_utils.py` houses the cleaned data-loading logic, baseline feature construction, diagnosis masks, ANOVA helpers, the persistence and onset labeling routines, and the modeling utilities. Importing from this module keeps the notebooks synchronized and removes the need for Drive mounts or duplicated code.

## Environment setup

Install a modern Python (3.9+) interpreter and then create an isolated environment:

```bash
conda create --name atyAN python=3.10
conda activate atyAN
pip install -r requirements.txt
```

For Google Colab, run the following once you have uploaded/cloned the repository so all notebooks share the same dependencies:

```python
!pip install -q -r requirements.txt
```

## Command-line execution

`main.py` provides a lightweight driver when you want to run the workflows without opening the notebooks. By default, all segments are executed and their outputs are written under `results/` as CSVs.

```bash
python main.py                           # run ANOVA + univariate + multivariate
python main.py --run-anova               # ANOVA only
python main.py --run-univariate          # univariate logistics only
python main.py --run-multivariate        # multivariate persistence models only
python main.py --output-dir outputs_dir  # customize the export folder
```

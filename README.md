# AtyAN

Stice et al., Predicting Risk Factors of Atypical Anorexia

## Notebook summary

The original monolithic notebooks have been decomposed into focused workflows that make it easier to audit each analytical step and to run them outside of Colab. Every notebook loads `BP1234-ONSET.csv` directly from the repository and shares a single preprocessing module (`analysis_utils.py`).

### `AtyAN_ANOVA.ipynb`
- Recreates the Aim 1 baseline contrasts (AAN vs. No AAN, weight-loss controls, and obese comparators).
- Includes the disorder-level comparisons (AAN vs. AN/BN/BED/PU; full and partial) together with BH-FDR q-values.

- Merges the original univariate onset-prediction workflow with the persistence/remission cohort so all single-feature analyses live in one place.
- Recreates the onset prediction task (dropping baseline AN/AAN diagnoses and labeling any mBMI-defined onset across waves 1–6) across the eight requested predictors (`FEAR_w1`, `WSO_w1`, `FAT_w1`, `CB_w1`, `w1dres`, `w1dep`, `w1tii`, `w1socf`).
- Restricts the persistence/remission cohort to waves 1, 4, 5, and 6, computes the new `aan_persistence`/`aan_remission` labels plus the `aan_course_flag`, and surfaces the 12 `[feature]-persistence` columns that follow the wave where each participant first meets the persistence definition.
- Adds the same model zoo used by the multivariate notebook (balanced RF in shallow/medium/deep configurations, logistic regression, iBRF, TabPFN RF, AutoTabPFN) with stratified repeated holdout so the onset and persistence cohorts can be compared quickly in Colab.

- Reuses the persistence cohort and evaluates the expanded model zoo (shallow/medium/deep balanced RF variants, logistic regression, iBRF, TabPFN RF, AutoTabPFN) with stratified repeated holdout to mirror the univariate notebook.
- Surfaces split-level metrics, overfitting flags, and the feature-importance tables where the estimators expose them.

All notebooks now start with a minimal `pip install -r requirements.txt` cell so Colab runtimes can hydrate the dependencies without manual Drive mounts. The TabPFN components require an up-to-date `torch` with CUDA support on GPU runtimes; the provided `requirements.txt` installs `torch>=2.1`, `tabpfn`, and `tabpfn-extensions[all]` in the correct order.

## Shared utilities

`analysis_utils.py` houses the cleaned data-loading logic, baseline feature construction, diagnosis masks, ANOVA helpers, the persistence and onset labeling routines, and the modeling utilities. Importing from this module keeps the notebooks synchronized and removes the need for Drive mounts or duplicated code. The helpers now derive the persistence/remission labels exclusively from waves 1/4/5/6, expose the `aan_course_flag` convenience column, and populate the wave-aware `[feature]-persistence` predictors so every participant is compared at the correct baseline.

## Environment setup

Install a modern Python (3.9+) interpreter and then create an isolated environment:

```bash
conda create --name atyAN python=3.10
conda activate atyAN
pip install -r requirements.txt
```

For Google Colab, run the following once you have uploaded/cloned the repository so all notebooks share the same dependencies and to ensure TabPFN/iBRF stay synced with PyTorch:

```python
!pip install -q -r requirements.txt
```

If the GPU runtime ships with an older CUDA build, upgrade PyTorch first (e.g., `pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`) and then re-run the requirements cell so `tabpfn-extensions` links against the refreshed stack.

## Model zoo and repeated holdout

`analysis_utils.py` exposes both the per-feature logistic regressions and a reusable model zoo. Each dataset (univariate onset, univariate persistence, multivariate persistence) can run the seven estimators we maintain:

1. Shallow balanced random forest (max depth 3, balanced subsampling tuned for ~3% positives).
2. Medium balanced random forest (max depth 6, balanced subsampling tuned for ~3% positives).
3. Deep balanced random forest (max depth 12, balanced subsampling tuned for ~3% positives).
4. Logistic regression with class weights and strong regularization.
5. Improved Balanced Random Forest (`iBRF` with `balance_split=0.65`, `n_estimators=200`).
6. TabPFN Random Forest (`RandomForestTabPFNClassifier` with shallow trees on top of a base TabPFN transformer).
7. AutoTabPFN (post-hoc ensemble with the "medium_quality" preset for tractable runtime).

All of them share the same stratified repeated holdout routine (configurable repeats/test-size) and surface train/test ROC-AUC deltas so overfitting can be spotted without rerunning the full suite.

## Command-line execution

`main.py` provides a lightweight driver when you want to run the workflows without opening the notebooks. By default, all segments are executed and their outputs are written under `results/` as CSVs. Use the `--model-names` flag when you only need to rerun a subset of the zoo (e.g., to recheck TabPFN without touching the forests), and adjust the repeated holdout parameters to probe stability.

```bash
python main.py                           # run ANOVA + univariate + multivariate
python main.py --run-anova               # ANOVA only
python main.py --run-univariate          # univariate logistics only
python main.py --run-multivariate        # multivariate persistence models only
python main.py --output-dir outputs_dir  # customize the export folder
python main.py --model-names logistic_regression,tabpfn_random_forest
python main.py --holdout-repeats 10 --holdout-test-size 0.25
```

import json
import math
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix,
    roc_auc_score,
    average_precision_score,
    brier_score_loss,
)
from sklearn.linear_model import LogisticRegression
from imblearn.ensemble import BalancedRandomForestClassifier

from analysis_utils import engineer_baseline_features, prepare_univariate_prediction_dataset

FEATURES_14 = [
    "w1tii",
    "w1bs",
    "w1dres",
    "w1socf",
    "w1dep",
    "w1intbmi",
    "BE_w1",
    "CB_w1",
    "WSO_w1",
    "FEAR_w1",
    "FAT_w1",
    "LEB_w1",
    "cond_bp",
    "cond_hw",
]

COND_MAP = {
    "peer delivered": "BP",
    "ebody": "BP",
    "clincian delivered": "BP",
    "clinician delivered": "BP",
    "diss. (bp)": "BP",
    "exp writing": "BP",
    "control/video control": "Control",
    "healthy weight": "Healthy Weight",
}

COND_CANONICAL = {
    "BP": "BP",
    "Control": "Control",
    "Healthy Weight": "Healthy Weight",
}


def clean_and_encode_condition(df: pd.DataFrame) -> pd.DataFrame:
    if "study_cond" not in df.columns:
        raise KeyError("Missing 'study_cond' column; use dataset with condition labels.")

    cond_raw = df["study_cond"]
    if cond_raw.isna().any() or cond_raw.astype(str).str.strip().eq("").any():
        raise ValueError("Found missing/blank entries in 'study_cond'; expected none.")

    cond_norm = cond_raw.astype(str).str.strip().str.lower()
    cond_clean = cond_norm.map(COND_MAP)

    if cond_clean.isna().any():
        bad_vals = sorted(cond_raw.loc[cond_clean.isna()].unique())
        raise ValueError(f"Unmapped 'study_cond' values: {bad_vals}")

    df = df.copy()
    df["cond_clean"] = cond_clean.map(COND_CANONICAL).astype("category")
    df["cond_bp"] = (df["cond_clean"] == "BP").astype(int)
    df["cond_hw"] = (df["cond_clean"] == "Healthy Weight").astype(int)
    return df


MI_M = 3
SEED = 42


def _iter_imputer(seed: int, posterior: bool):
    return IterativeImputer(
        random_state=int(seed),
        sample_posterior=bool(posterior),
        max_iter=10,
        initial_strategy="median",
        skip_complete=True,
    )


def _safe_logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def fit_logistic_recalibrator(y_train: np.ndarray, p_train: np.ndarray):
    import statsmodels.api as sm

    x = _safe_logit(p_train)
    X_ = sm.add_constant(x, has_constant="add")
    res = sm.GLM(y_train, X_, family=sm.families.Binomial()).fit()
    a = float(res.params[0])
    b = float(res.params[1])
    return a, b


def apply_logistic_recalibrator(p: np.ndarray, a: float, b: float) -> np.ndarray:
    x = _safe_logit(p)
    z = a + b * x
    return 1.0 / (1.0 + np.exp(-z))


def calibration_slope_intercept(y_true: np.ndarray, probs: np.ndarray):
    import statsmodels.api as sm

    x = _safe_logit(probs)
    X_ = sm.add_constant(x, has_constant="add")
    res = sm.GLM(y_true, X_, family=sm.families.Binomial()).fit()
    return float(res.params[0]), float(res.params[1])


def ece_quantile(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=int)
    probs = np.asarray(probs, dtype=float)
    probs = np.clip(probs, 0.0, 1.0)
    qs = np.linspace(0, 1, int(n_bins) + 1)
    edges = np.quantile(probs, qs)
    edges[0] = 0.0
    edges[-1] = 1.0
    bin_ids = np.digitize(probs, edges[1:-1], right=True)
    ece = 0.0
    for b in range(int(n_bins)):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        w = float(np.mean(mask))
        obs = float(np.mean(y_true[mask]))
        pred = float(np.mean(probs[mask]))
        ece += w * abs(obs - pred)
    return float(ece)


def summarize_probs(probs, y):
    y_arr = np.asarray(y, dtype=int)
    auc = float(roc_auc_score(y_arr, probs))
    auprc = float(average_precision_score(y_arr, probs))
    brier = float(brier_score_loss(y_arr, probs))
    cint, cslope = calibration_slope_intercept(y_arr, probs)
    ece = float(ece_quantile(y_arr, probs, n_bins=10))
    return {
        "auc": auc,
        "auprc": auprc,
        "brier": brier,
        "cal_int": cint,
        "cal_slope": cslope,
        "ece": ece,
    }


def make_brf_det_pipeline(det_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", _iter_imputer(seed=det_seed, posterior=False)),
            (
                "model",
                BalancedRandomForestClassifier(
                    random_state=det_seed,
                    n_jobs=1,
                ),
            ),
        ]
    )


def make_brf_mi_pipeline(*, imputer_seed: int, params: dict) -> Pipeline:
    est = Pipeline(
        [
            ("imputer", _iter_imputer(seed=imputer_seed, posterior=True)),
            (
                "model",
                BalancedRandomForestClassifier(
                    random_state=SEED,
                    n_jobs=1,
                ),
            ),
        ]
    )
    est.set_params(**params)
    return est


def make_log_det_pipeline(det_seed: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", _iter_imputer(seed=det_seed, posterior=False)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def make_log_mi_pipeline(*, imputer_seed: int, params: dict) -> Pipeline:
    est = Pipeline(
        [
            ("imputer", _iter_imputer(seed=imputer_seed, posterior=True)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    est.set_params(**params)
    return est


BRF_GRID = {
    "model__n_estimators": [400],
    "model__max_depth": [3, 4, 5],
    "model__min_samples_leaf": [4],
    "model__max_features": ["sqrt"],
}

LOG_GRID = {
    "model__C": [0.01],
    "model__penalty": ["l2"],
    "model__solver": ["lbfgs"],
}


def nested_cv_oof(make_det_pipeline, make_mi_pipeline, grid, X, y, outer_cv):
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    y_arr = np.asarray(y, dtype=int)

    oof_uncal = np.zeros(len(y_arr), dtype=float)
    oof_cal = np.zeros(len(y_arr), dtype=float)
    best_params_per_fold = []

    import time
    t0_all = time.time()
    for fold_i, (tr, te) in enumerate(outer_cv.split(X, y), start=1):
        t0_fold = time.time()
        print(f"  Fold {fold_i}/5: inner tuning start", flush=True)
        inner_cv = StratifiedKFold(
            n_splits=5,
            shuffle=True,
            random_state=SEED + 1000 * fold_i,
        )

        search = GridSearchCV(
            estimator=make_det_pipeline(det_seed=SEED + 1000 * fold_i),
            param_grid=grid,
            scoring="roc_auc",
            n_jobs=1,
            cv=inner_cv,
            error_score="raise",
        )
        search.fit(X.iloc[tr], y.iloc[tr])
        best_params = dict(search.best_params_)
        best_params_per_fold.append(best_params)
        print(
            f"  Fold {fold_i}/5: best inner AUROC={search.best_score_:.3f}, params={best_params}",
            flush=True,
        )

        p_tr = np.zeros(len(tr), dtype=float)
        p_te = np.zeros(len(te), dtype=float)

        for j in range(int(MI_M)):
            imp_seed = SEED + 10000 * fold_i + j
            est = make_mi_pipeline(imputer_seed=imp_seed, params=best_params)
            est.fit(X.iloc[tr], y.iloc[tr])
            p_tr += est.predict_proba(X.iloc[tr])[:, 1]
            p_te += est.predict_proba(X.iloc[te])[:, 1]

        p_tr /= float(MI_M)
        p_te /= float(MI_M)

        oof_uncal[te] = p_te

        a, b = fit_logistic_recalibrator(y_arr[tr], p_tr)
        oof_cal[te] = apply_logistic_recalibrator(p_te, a=a, b=b)
        print(
            f"  Fold {fold_i}/5: done in {time.time() - t0_fold:.1f}s",
            flush=True,
        )

    print(f"  Model total time: {time.time() - t0_all:.1f}s", flush=True)
    return oof_uncal, oof_cal, best_params_per_fold


THRESHOLDS = np.unique(
    np.concatenate(
        [
            np.linspace(0.01, 0.10, 10),
            np.linspace(0.12, 0.50, 20),
            np.linspace(0.55, 0.95, 9),
        ]
    )
)
MIN_SENS = 0.60
MIN_ACC = 0.60


def eval_thresholds(probs: np.ndarray, y: pd.Series):
    y_arr = np.asarray(y, dtype=int)
    rows = []
    auc = float(roc_auc_score(y_arr, probs))
    for thr in THRESHOLDS:
        preds = (probs >= thr).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_arr, preds).ravel()
        sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        acc = (tp + tn) / (tp + tn + fp + fn)
        bal = (sens + spec) / 2 if np.isfinite(sens) and np.isfinite(spec) else np.nan
        rows.append(
            {
                "threshold": float(thr),
                "auc": auc,
                "accuracy": float(acc),
                "sensitivity": float(sens),
                "specificity": float(spec),
                "balanced_accuracy": float(bal),
            }
        )
    df = pd.DataFrame(rows)
    feasible = df[(df["sensitivity"] >= MIN_SENS) & (df["accuracy"] >= MIN_ACC)].copy()
    if feasible.empty:
        feasible = df.copy()
    best = feasible.sort_values(
        ["balanced_accuracy", "sensitivity", "specificity"],
        ascending=[False, False, False],
    ).iloc[0]
    return {k: float(best[k]) for k in ["threshold", "auc", "accuracy", "sensitivity", "specificity", "balanced_accuracy"]}


def main():
    pref = Path("AAN-ONSET-MERGED.csv")
    fallback = Path("BP1234-ONSET.csv")
    if pref.exists():
        raw_df = pd.read_csv(pref, low_memory=False)
        ds = pref.name
    elif fallback.exists():
        raw_df = pd.read_csv(fallback, low_memory=False)
        ds = fallback.name
    else:
        raise FileNotFoundError("AAN-ONSET-MERGED.csv or BP1234-ONSET.csv is required.")

    feature_df, feature_sets = engineer_baseline_features(raw_df)
    feature_sets["all_features"] = [c for c in feature_sets.get("all_features", []) if not str(c).endswith("-persistence")]

    feature_df = clean_and_encode_condition(feature_df)
    condition_covariates = ["cond_bp", "cond_hw"]
    feature_sets["condition_covariates"] = condition_covariates
    feature_sets["all_features"] = list(dict.fromkeys(feature_sets["all_features"] + condition_covariates))

    design_df = prepare_univariate_prediction_dataset(feature_df, feature_sets["all_features"])
    y = design_df["aan_onset_anywave"].astype(int)
    X = design_df[FEATURES_14].copy()

    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)

    print(f"Using dataset: {ds}")
    print(f"X shape: {X.shape}; prevalence={y.mean():.4f}")

    print("Running BRF nested CV...")
    brf_oof, brf_oof_cal, brf_best = nested_cv_oof(
        make_brf_det_pipeline,
        make_brf_mi_pipeline,
        BRF_GRID,
        X,
        y,
        outer_cv,
    )

    print("Running Logistic nested CV...")
    log_oof, log_oof_cal, log_best = nested_cv_oof(
        make_log_det_pipeline,
        make_log_mi_pipeline,
        LOG_GRID,
        X,
        y,
        outer_cv,
    )

    out = {
        "uncalibrated": {
            "brf": summarize_probs(brf_oof, y),
            "logistic": summarize_probs(log_oof, y),
        },
        "calibrated": {
            "brf": summarize_probs(brf_oof_cal, y),
            "logistic": summarize_probs(log_oof_cal, y),
        },
        "thresholds": {
            "uncalibrated": {
                "brf": eval_thresholds(brf_oof, y),
                "logistic": eval_thresholds(log_oof, y),
            },
            "calibrated": {
                "brf": eval_thresholds(brf_oof_cal, y),
                "logistic": eval_thresholds(log_oof_cal, y),
            },
        },
        "best_params": {
            "brf": brf_best,
            "logistic": log_best,
        },
    }

    print("\\n=== SUMMARY ===")
    print(json.dumps(out, indent=2))
    Path("multivariate_methods_only_metrics.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("Saved multivariate_methods_only_metrics.json")


if __name__ == "__main__":
    main()

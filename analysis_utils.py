"""Shared helpers for the reorganized AtyAN notebooks."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

DATA_PATH = Path(__file__).resolve().parent / "BP1234-ONSET.csv"
WAVES: List[int] = [1, 2, 3, 4, 5, 6]
BASELINE_RISK_COLS: List[str] = [
    "w1tii",
    "w1bs",
    "w1dres",
    "w1socf",
    "w1dep",
    "w1intbmi",
    "w1age",
]
CB_COLUMNS = ["w1ed8a", "w1ed9a", "w1ed10a", "w1ed11a"]
TRUTHY_STRINGS = {"TRUE", "T", "YES", "Y", "1", "PRESENT"}
FALSY_STRINGS = {"FALSE", "F", "NO", "N", "0", "ABSENT"}


class SimpleBalancedRandomForestClassifier(RandomForestClassifier):
    """A lightweight stand-in for imblearn's BalancedRandomForestClassifier."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("class_weight", "balanced_subsample")
        super().__init__(*args, **kwargs)


def load_base_dataset(columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read the uploaded CSV directly from disk."""
    if columns:
        return pd.read_csv(DATA_PATH, usecols=list(columns))
    return pd.read_csv(DATA_PATH)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_boolean(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(np.nan)
    as_str = series.astype(str).str.strip().str.upper()
    out = pd.Series(np.nan, index=series.index)
    out[as_str.isin(TRUTHY_STRINGS)] = True
    out[as_str.isin(FALSY_STRINGS)] = False
    numeric = pd.to_numeric(series, errors="coerce")
    out[numeric.notna()] = numeric[numeric.notna()] != 0
    return out


def engineer_baseline_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, List[str]]]:
    """Create the baseline risk and prodromal composites used across notebooks."""
    work = df.copy()

    risk_cols: List[str] = []
    for col in BASELINE_RISK_COLS:
        if col in work.columns:
            work[col] = _coerce_numeric(work[col])
            risk_cols.append(col)

    work["BE_w1"] = _coerce_numeric(work.get("w1ede1a", pd.Series(np.nan, index=work.index)))

    cb_present = [c for c in CB_COLUMNS if c in work.columns]
    if cb_present:
        cb_block = work[cb_present].apply(_coerce_numeric)
        work["CB_w1"] = cb_block.max(axis=1, skipna=True)
        work.loc[cb_block.notna().sum(axis=1) == 0, "CB_w1"] = np.nan
    else:
        work["CB_w1"] = np.nan

    work["WSO_w1"] = _coerce_numeric(work.get("w1ed15a", pd.Series(np.nan, index=work.index)))
    work["FEAR_w1"] = _coerce_numeric(work.get("w1ed17a", pd.Series(np.nan, index=work.index)))
    work["FAT_w1"] = _coerce_numeric(work.get("w1ed19a", pd.Series(np.nan, index=work.index)))

    mbmi_pct = _coerce_numeric(work.get("w1mbmi_pct", pd.Series(np.nan, index=work.index)))
    work["LEB_w1"] = np.clip(90.0 - mbmi_pct, 0, None) / 90.0

    prodromal_cols = [c for c in ["BE_w1", "CB_w1", "WSO_w1", "FEAR_w1", "FAT_w1", "LEB_w1"] if c in work.columns]
    all_features = list(dict.fromkeys(risk_cols + prodromal_cols))

    feature_sets = {
        "risk": risk_cols,
        "prodromal": prodromal_cols,
        "all_features": all_features,
        "outcomes": all_features,
    }
    return work, feature_sets


def _match_prefix_columns(df: pd.DataFrame, prefixes: Sequence[str]) -> List[str]:
    cols: List[str] = []
    for prefix in prefixes:
        pattern = re.compile(rf"^{re.escape(prefix)}[._-]?", flags=re.IGNORECASE)
        cols.extend([c for c in df.columns if pattern.match(c)])
    return cols


def has_cols(df: pd.DataFrame, prefixes: Sequence[str]) -> pd.Series:
    matched = _match_prefix_columns(df, prefixes)
    if not matched:
        return pd.Series(False, index=df.index)
    sub = df[matched]
    truthy = sub.apply(_coerce_boolean)
    numeric = sub.apply(_coerce_numeric)
    numeric_present = numeric.notna() & numeric.ne(0)
    mask = truthy.fillna(False) | numeric_present.fillna(False)
    return mask.any(axis=1)


def build_diagnosis_masks(df: pd.DataFrame) -> Dict[str, pd.Series]:
    mask_AN = has_cols(df, ["fan", "pan"])
    mask_BN = has_cols(df, ["fbn", "pbn"])
    mask_BED = has_cols(df, ["fbe", "pbe"])
    mask_PU = has_cols(df, ["fpu", "ppu"])

    onset = _coerce_boolean(df.get("w1ONSET-FULL", pd.Series(np.nan, index=df.index)))
    mask_atyAN = onset.fillna(False)

    mask_AAN = mask_atyAN & ~mask_AN & ~mask_BN & ~mask_BED & ~mask_PU
    mask_NoAAN = ~mask_AAN & ~mask_AN & ~mask_BN & ~mask_BED & ~mask_PU
    mask_WL_noCog = _coerce_boolean(df.get("w1HEALTHY-WL", pd.Series(False, index=df.index))).fillna(False) & ~mask_AAN
    mask_OBESE_noAAN = _coerce_numeric(df.get("w1intbmi", pd.Series(np.nan, index=df.index))).gt(30).fillna(False) & ~mask_AAN

    return {
        "AN": mask_AN,
        "BN": mask_BN,
        "BED": mask_BED,
        "PU": mask_PU,
        "atyAN": mask_atyAN,
        "AAN": mask_AAN,
        "NoAAN": mask_NoAAN,
        "WL_noCog": mask_WL_noCog,
        "Obese_noAAN": mask_OBESE_noAAN,
    }


def apply_bh_fdr(p_values: Sequence[float]) -> np.ndarray:
    p = np.array(pd.to_numeric(pd.Series(p_values), errors="coerce"), dtype=float)
    mask = np.isfinite(p)
    q = np.full_like(p, np.nan)
    n = mask.sum()
    if n == 0:
        return q
    values = p[mask]
    order = np.argsort(values)
    ranks = np.empty_like(order)
    ranks[order] = np.arange(1, n + 1)
    adjusted = values * n / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0, 1)
    q[mask] = adjusted
    return q


def anova_one_contrast(
    df: pd.DataFrame,
    mask_g1: pd.Series,
    mask_g2: pd.Series,
    outcomes: Sequence[str],
    name_g1: str,
    name_g2: str,
) -> pd.DataFrame:
    mask_g1 = mask_g1.reindex(df.index).fillna(False).astype(bool)
    mask_g2 = mask_g2.reindex(df.index).fillna(False).astype(bool) & ~mask_g1
    rows = []
    for outcome in outcomes:
        if outcome not in df.columns:
            continue
        data = pd.to_numeric(df[outcome], errors="coerce")
        g1 = data[mask_g1].dropna()
        g2 = data[mask_g2].dropna()
        if g1.empty or g2.empty:
            continue
        group_labels = pd.Series(
            np.where(mask_g1, name_g1, np.where(mask_g2, name_g2, np.nan)),
            index=df.index,
        )
        dfx = pd.DataFrame({"y": data, "group": group_labels})
        dfx = dfx.dropna(subset=["y", "group"])
        if dfx["group"].nunique() < 2:
            continue
        import statsmodels.formula.api as smf  # local import to keep notebooks lightweight
        import statsmodels.api as sm

        model = smf.ols("y ~ C(group)", data=dfx).fit()
        aov = sm.stats.anova_lm(model, typ=2)
        F = float(aov.loc["C(group)", "F"])
        p_val = float(aov.loc["C(group)", "PR(>F)"])
        rows.append(
            {
                "outcome": outcome,
                f"mean_{name_g1}": g1.mean(),
                f"sd_{name_g1}": g1.std(ddof=1),
                f"n_{name_g1}": int(g1.shape[0]),
                f"mean_{name_g2}": g2.mean(),
                f"sd_{name_g2}": g2.std(ddof=1),
                f"n_{name_g2}": int(g2.shape[0]),
                "F": F,
                "p": p_val,
            }
        )
    result = pd.DataFrame(rows)
    if not result.empty:
        result["q"] = apply_bh_fdr(result["p"].to_numpy())
    return result.sort_values("p")


def run_baseline_anova_contrasts(
    df: pd.DataFrame,
    masks: Dict[str, pd.Series],
    outcomes: Sequence[str],
) -> Dict[str, pd.DataFrame]:
    comparisons = {
        "AAN_vs_NoAAN": ("AAN_w1", masks["AAN"], "NoAAN_w1", masks["NoAAN"]),
        "AAN_vs_WL10_noCog": ("AAN_w1", masks["AAN"], "WL10_noCog_w1", masks["WL_noCog"]),
        "AAN_vs_Obese": ("AAN_w1", masks["AAN"], "Obese_noAAN_w1", masks["Obese_noAAN"]),
    }
    tables: Dict[str, pd.DataFrame] = {}
    for key, (name_g1, mask_g1, name_g2, mask_g2) in comparisons.items():
        tables[key] = anova_one_contrast(df, mask_g1, mask_g2, outcomes, name_g1, name_g2)
    return tables


def run_disorder_level_anova(df: pd.DataFrame, outcomes: Sequence[str]) -> pd.DataFrame:
    masks = build_diagnosis_masks(df)
    mask_atyAN = masks["atyAN"]
    families = {
        ("AN", "full"): has_cols(df, ["fan"]),
        ("AN", "partial"): has_cols(df, ["pan"]),
        ("BN", "full"): has_cols(df, ["fbn"]),
        ("BN", "partial"): has_cols(df, ["pbn"]),
        ("BED", "full"): has_cols(df, ["fbe"]),
        ("BED", "partial"): has_cols(df, ["pbe"]),
        ("PU", "full"): has_cols(df, ["fpu"]),
        ("PU", "partial"): has_cols(df, ["ppu"]),
    }
    tables = []
    for (disorder, level), mask in families.items():
        pure_mask = mask & ~mask_atyAN
        if pure_mask.sum() < 5:
            continue
        tbl = anova_one_contrast(df, mask_atyAN, pure_mask, outcomes, "AAN_w1", f"{disorder}_{level}")
        if tbl.empty:
            continue
        tbl["disorder"] = disorder
        tbl["level"] = level
        tables.append(tbl)
    if not tables:
        return pd.DataFrame(columns=["outcome", "disorder", "level", "p", "q"])
    combined = pd.concat(tables, ignore_index=True)
    combined["q"] = apply_bh_fdr(combined["p"].to_numpy())
    return combined.sort_values(["outcome", "disorder", "level"])


def _detect_onset_columns(df: pd.DataFrame) -> List[str]:
    cols: List[str] = []
    for wave in WAVES:
        primary = f"w{wave}ONSET-FULL"
        if primary in df.columns:
            cols.append(primary)
        mbmi = f"w{wave}ONSET-FULL-mBMI"
        if mbmi in df.columns:
            cols.append(mbmi)
    return cols


def _persistence_label(row: pd.Series) -> float:
    arr = row.dropna().astype(bool).to_numpy()
    if not arr.size or not arr.any():
        return math.nan
    first = np.argmax(arr)
    last = len(arr) - 1 - np.argmax(arr[::-1])
    segment = arr[first : last + 1]
    has_gap = (~segment).any()
    return 1.0 if not has_gap else 0.0


def prepare_persistence_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    onset_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    cols = list(onset_cols) if onset_cols else _detect_onset_columns(df)
    present_cols = [c for c in cols if c in df.columns]
    if not present_cols:
        raise ValueError("No onset columns available in the dataset.")
    onset_block = df[present_cols].apply(_coerce_boolean)
    eligible = onset_block.any(axis=1)
    complete = onset_block.notna().all(axis=1)
    subset = df.loc[eligible & complete].copy()
    subset["aan_persistence"] = onset_block.loc[subset.index].apply(_persistence_label, axis=1)
    subset = subset.dropna(subset=["aan_persistence"])
    usable_features = [c for c in feature_cols if c in subset.columns]
    return subset[usable_features + ["aan_persistence"]]


def prepare_univariate_prediction_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    onset_weight_label: str = "mBMI",
) -> pd.DataFrame:
    """Dataset for the univariate onset prediction models."""

    mask_fan = has_cols(df, ["fan"])
    mask_pan = has_cols(df, ["pan"])
    mask_w1_onset = _coerce_boolean(df.get("w1ONSET-FULL", pd.Series(np.nan, index=df.index))).fillna(False)
    subset = df.loc[~(mask_fan | mask_pan | mask_w1_onset)].copy()

    onset_pattern = re.compile(rf"^w\d+ONSET-FULL-{re.escape(onset_weight_label)}$", re.IGNORECASE)
    onset_cols = [c for c in subset.columns if onset_pattern.match(c)]
    if not onset_cols:
        raise ValueError(f"No ONSET-FULL-{onset_weight_label} columns found in the dataset.")

    onset_block = subset[onset_cols].apply(_coerce_numeric).fillna(0)
    subset["aan_onset_anywave"] = onset_block.gt(0).any(axis=1).astype(int)

    usable_features = [c for c in feature_cols if c in subset.columns]
    return subset[usable_features + ["aan_onset_anywave"]]


def _binary_predictions(probas: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probas >= threshold).astype(int)


def run_univariate_logistic_regressions(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "aan_persistence",
) -> pd.DataFrame:
    y = df[target_col].astype(int)
    results = []
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        X = df[[feature]]
        pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=1000, penalty="l2", solver="lbfgs", class_weight="balanced"),
                ),
            ]
        )
        probas = cross_val_predict(pipeline, X, y, cv=cv, method="predict_proba")[:, 1]
        preds = _binary_predictions(probas)
        metrics = {
            "feature": feature,
            "roc_auc": roc_auc_score(y, probas),
            "average_precision": average_precision_score(y, probas),
            "balanced_accuracy": balanced_accuracy_score(y, preds),
            "f1": f1_score(y, preds),
            "accuracy": accuracy_score(y, preds),
        }
        pipeline.fit(X, y)
        metrics["coef"] = float(pipeline.named_steps["model"].coef_[0][0])
        metrics["intercept"] = float(pipeline.named_steps["model"].intercept_[0])
        results.append(metrics)
    return pd.DataFrame(results).sort_values("roc_auc", ascending=False)


def evaluate_multivariate_models(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "aan_persistence",
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
    usable_features = [c for c in feature_cols if c in df.columns]
    X = df[usable_features]
    y = df[target_col].astype(int)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "LogisticRegression": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(max_iter=2000, penalty="l2", solver="lbfgs", class_weight="balanced"),
                ),
            ]
        ),
        "LinearSVM": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    CalibratedClassifierCV(LinearSVC(class_weight="balanced", max_iter=5000), method="sigmoid", cv=3),
                ),
            ]
        ),
        "GradientBoosting": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("model", GradientBoostingClassifier(random_state=42)),
            ]
        ),
        "RandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=5,
                        class_weight="balanced_subsample",
                        random_state=42,
                    ),
                ),
            ]
        ),
        "BalancedRandomForest": Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    SimpleBalancedRandomForestClassifier(
                        n_estimators=500,
                        max_depth=None,
                        min_samples_leaf=5,
                        random_state=42,
                    ),
                ),
            ]
        ),
    }
    metrics_rows = []
    feature_tables: Dict[str, pd.DataFrame] = {}
    for name, pipeline in models.items():
        method = "predict_proba"
        probas = cross_val_predict(pipeline, X, y, cv=cv, method=method)[:, 1]
        preds = _binary_predictions(probas)
        metrics_rows.append(
            {
                "model": name,
                "roc_auc": roc_auc_score(y, probas),
                "average_precision": average_precision_score(y, probas),
                "balanced_accuracy": balanced_accuracy_score(y, preds),
                "f1": f1_score(y, preds),
                "accuracy": accuracy_score(y, preds),
            }
        )
        pipeline.fit(X, y)
        model = pipeline.named_steps["model"]
        if hasattr(model, "feature_importances_"):
            fi = pd.DataFrame(
                {
                    "feature": usable_features,
                    "importance": model.feature_importances_,
                }
            ).sort_values("importance", ascending=False)
            feature_tables[name] = fi
    metrics_df = pd.DataFrame(metrics_rows).sort_values("roc_auc", ascending=False)
    return metrics_df, feature_tables

"""Shared helpers for the reorganized AtyAN notebooks."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
MODEL_FEATURE_COLUMNS: List[str] = [
    "w1tii",
    "w1bs",
    "w1dres",
    "w1socf",
    "w1dep",
    "WSO_w1",
    "FEAR_w1",
    "FAT_w1",
]
CB_COLUMNS = ["w1ed8a", "w1ed9a", "w1ed10a", "w1ed11a"]
TRUTHY_STRINGS = {"TRUE", "T", "YES", "Y", "1", "PRESENT"}
FALSY_STRINGS = {"FALSE", "F", "NO", "N", "0", "ABSENT"}

PERSISTENCE_ONSET_COLUMNS: Mapping[int, str] = {
    1: "w1ONSET-FULL",
    2: "w2ONSET-FULL-mBMI",
    3: "w3ONSET-FULL-mBMI",
    4: "w4ONSET-FULL-mBMI",
    5: "w5ONSET-FULL-mBMI",
}
PERSISTENCE_SYMPTOM_THRESHOLD: float = 4.0


class SimpleBalancedRandomForestClassifier(RandomForestClassifier):
    """A lightweight stand-in for imblearn's BalancedRandomForestClassifier."""

    def __init__(self, *args, **kwargs):
        kwargs.setdefault("class_weight", "balanced_subsample")
        super().__init__(*args, **kwargs)


try:  # optional dependency
    from imblearn.ensemble import BalancedRandomForestClassifier as ImblearnBRF
except Exception:  # pragma: no cover - imblearn is optional
    ImblearnBRF = None

try:
    from ibrf import iBRF as IBRFClassifier
except Exception:  # pragma: no cover - iBRF is optional
    IBRFClassifier = None


OVERFIT_DELTA = 0.08


def load_base_dataset(columns: Optional[Sequence[str]] = None) -> pd.DataFrame:
    """Read the uploaded CSV directly from disk."""
    return pd.read_csv(DATA_PATH, low_memory=False)


def _coerce_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _coerce_boolean(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(pd.NA, dtype="boolean")
    as_str = series.astype(str).str.strip().str.upper()
    out = pd.Series(pd.NA, index=series.index, dtype="boolean")
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

    model_features = [col for col in MODEL_FEATURE_COLUMNS if col in work.columns]

    feature_sets = {
        "risk": risk_cols,
        "prodromal": prodromal_cols,
        "all_features": all_features,
        "outcomes": all_features,
        "model_features": model_features,
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


def _build_wave_block(
    df: pd.DataFrame,
    mapping: Mapping[int, str],
    coercer: Callable[[pd.Series], pd.Series],
) -> pd.DataFrame:
    pairs = [(wave, column) for wave, column in mapping.items() if column in df.columns]
    if not pairs:
        return pd.DataFrame(index=df.index)
    pairs.sort(key=lambda item: item[0])
    columns = [column for _, column in pairs]
    block = df[columns].apply(coercer)
    block.columns = [wave for wave, _ in pairs]
    return block


def _persistence_next_wave_labels(df: pd.DataFrame) -> pd.Series:
    onset_block = _build_wave_block(df, PERSISTENCE_ONSET_COLUMNS, _coerce_boolean)
    if onset_block.empty:
        raise ValueError("No onset columns available in the dataset.")

    ed14_mapping = {wave: f"w{wave}ed14" for wave in range(2, 7)}
    ed16_mapping = {wave: f"w{wave}ed16" for wave in range(2, 7)}
    ed14_block = _build_wave_block(df, ed14_mapping, _coerce_numeric)
    ed16_block = _build_wave_block(df, ed16_mapping, _coerce_numeric)

    if ed14_block.empty or ed16_block.empty:
        raise ValueError("Symptom severity columns (ed14/ed16) are missing for the follow-up waves.")

    labels = pd.Series(np.nan, index=df.index, dtype=float)

    for wave in sorted(onset_block.columns):
        next_wave = wave + 1
        if next_wave not in ed14_block.columns or next_wave not in ed16_block.columns:
            continue
        onset_flags = onset_block[wave].fillna(False).astype(bool)
        follow_14 = ed14_block[next_wave]
        follow_16 = ed16_block[next_wave]
        mask = onset_flags & follow_14.notna() & follow_16.notna() & labels.isna()
        if not mask.any():
            continue
        persistent = (follow_14 > PERSISTENCE_SYMPTOM_THRESHOLD) & (
            follow_16 > PERSISTENCE_SYMPTOM_THRESHOLD
        )
        labels.loc[mask] = persistent.loc[mask].astype(float)

    return labels


def prepare_persistence_dataset(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    onset_cols: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    if onset_cols:
        raise ValueError(
            "Custom onset column overrides are no longer supported by the persistence definition."
        )
    labels = _persistence_next_wave_labels(df)
    subset = df.loc[labels.notna()].copy()
    if subset.empty:
        return pd.DataFrame(columns=[*feature_cols, "aan_persistence"])
    subset["aan_persistence"] = labels.loc[subset.index]
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

    onset_pattern = re.compile(
        rf"^w([1-6])ONSET-FULL-{re.escape(onset_weight_label)}$",
        re.IGNORECASE,
    )
    onset_cols = [c for c in subset.columns if onset_pattern.match(c)]
    if not onset_cols:
        raise ValueError(f"No ONSET-FULL-{onset_weight_label} columns found in the dataset.")

    onset_block = subset[onset_cols].apply(_coerce_numeric).fillna(0)
    subset["aan_onset_anywave"] = onset_block.gt(0).any(axis=1).astype(int)

    usable_features = [c for c in feature_cols if c in subset.columns]
    return subset[usable_features + ["aan_onset_anywave"]]


def _stratified_holdout_indices(
    y: pd.Series,
    repeats: int = 5,
    test_size: float = 0.3,
    random_state: int = 42,
):
    splitter = StratifiedShuffleSplit(
        n_splits=repeats,
        test_size=test_size,
        random_state=random_state,
    )
    y_array = np.asarray(y)
    dummy = np.zeros((len(y_array), 1))
    for train_idx, test_idx in splitter.split(dummy, y_array):
        yield train_idx, test_idx


def _binary_predictions(probas: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    return (probas >= threshold).astype(int)


def _safe_metric(func, *args) -> float:
    try:
        return float(func(*args))
    except ValueError:
        return float("nan")


def _safe_rate(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def _score_probabilities(y_true: Sequence[int], probas: np.ndarray) -> Dict[str, float]:
    preds = _binary_predictions(probas)
    y_true_arr = np.asarray(y_true)
    preds_arr = np.asarray(preds)
    tp = float(((y_true_arr == 1) & (preds_arr == 1)).sum())
    tn = float(((y_true_arr == 0) & (preds_arr == 0)).sum())
    fp = float(((y_true_arr == 0) & (preds_arr == 1)).sum())
    fn = float(((y_true_arr == 1) & (preds_arr == 0)).sum())
    sensitivity = _safe_rate(tp, tp + fn)
    specificity = _safe_rate(tn, tn + fp)
    if math.isnan(sensitivity) or math.isnan(specificity):
        g_mean = float("nan")
    else:
        g_mean = math.sqrt(sensitivity * specificity)
    return {
        "roc_auc": _safe_metric(roc_auc_score, y_true, probas),
        "average_precision": _safe_metric(average_precision_score, y_true, probas),
        "balanced_accuracy": _safe_metric(balanced_accuracy_score, y_true, preds),
        "f_score": _safe_metric(f1_score, y_true, preds),
        "accuracy": _safe_metric(accuracy_score, y_true, preds),
        "sensitivity": sensitivity,
        "specificity": specificity,
        "g_mean": g_mean,
    }


def _aggregate_metric_blocks(blocks: List[Dict[str, float]], prefix: str) -> Dict[str, float]:
    if not blocks:
        return {}
    summary: Dict[str, float] = {}
    for key in blocks[0]:
        values = np.array([row[key] for row in blocks], dtype=float)
        summary[f"{prefix}_{key}_mean"] = float(np.nanmean(values))
        summary[f"{prefix}_{key}_std"] = float(np.nanstd(values))
    return summary


def _tree_pipeline(model) -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def _logistic_pipeline(random_state: int = 42) -> Pipeline:
    model = LogisticRegression(
        max_iter=4000,
        solver="lbfgs",
        class_weight={0: 0.35, 1: 0.65},
        C=0.8,
        random_state=random_state,
    )
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("model", model),
    ])


def _balanced_rf_pipeline(random_state: int = 42) -> Pipeline:
    if ImblearnBRF is not None:
        model = ImblearnBRF(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=35,
            max_features="sqrt",
            max_samples=0.8,
            sampling_strategy="auto",
            replacement=False,
            random_state=random_state,
        )
    else:
        model = SimpleBalancedRandomForestClassifier(
            n_estimators=400,
            max_depth=4,
            min_samples_leaf=35,
            max_features="sqrt",
            max_samples=0.8,
            random_state=random_state,
        )
    return _tree_pipeline(model)


def _ibrf_pipeline(random_state: int = 42) -> Pipeline:
    if IBRFClassifier is None:
        raise ImportError("iBRF is not installed. Run `pip install ibrf`." )
    model = IBRFClassifier(
        balance_split=0.65,
        n_estimators=200,
        max_depth=4,
        min_samples_leaf=30,
        max_features="sqrt",
        max_samples=0.8,
        random_state=random_state,
    )
    return _tree_pipeline(model)


def _tabpfn_pipeline(random_state: int = 42) -> Pipeline:
    try:
        from tabpfn import TabPFNClassifier
    except Exception as exc:  # pragma: no cover - optional heavy deps
        raise ImportError("TabPFN requires the `tabpfn` package. Run `pip install tabpfn`.") from exc
    model = TabPFNClassifier(random_state=random_state)
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", model),
    ])


def _model_builders(random_state: int = 42) -> Dict[str, Callable[[], Pipeline]]:
    return {
        "balanced_random_forest": lambda: _balanced_rf_pipeline(random_state),
        "logistic_regression": lambda: _logistic_pipeline(random_state),
        "ibrf": lambda: _ibrf_pipeline(random_state),
        "tabpfn": lambda: _tabpfn_pipeline(random_state),
    }


def available_model_names() -> List[str]:
    return list(_model_builders().keys())


def _normalize_model_name(name: str) -> str:
    return name.strip().lower().replace("-", "_").replace(" ", "_")


def run_univariate_logistic_regressions(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "aan_persistence",
    repeats: int = 5,
    test_size: float = 0.3,
    random_state: int = 42,
) -> pd.DataFrame:
    y = df[target_col].astype(int)
    splits = list(_stratified_holdout_indices(y, repeats, test_size, random_state))
    if not splits:
        return pd.DataFrame()
    base_pipeline = _logistic_pipeline(random_state)
    results = []
    for feature in feature_cols:
        if feature not in df.columns:
            continue
        X = df[[feature]]
        train_blocks: List[Dict[str, float]] = []
        test_blocks: List[Dict[str, float]] = []
        for train_idx, test_idx in splits:
            pipeline = clone(base_pipeline)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_train, y_train)
            train_blocks.append(_score_probabilities(y_train, pipeline.predict_proba(X_train)[:, 1]))
            test_blocks.append(_score_probabilities(y_test, pipeline.predict_proba(X_test)[:, 1]))
        summary = {"feature": feature}
        summary.update(_aggregate_metric_blocks(train_blocks, "train"))
        summary.update(_aggregate_metric_blocks(test_blocks, "test"))
        train_auc = summary.get("train_roc_auc_mean", float("nan"))
        test_auc = summary.get("test_roc_auc_mean", float("nan"))
        if math.isnan(train_auc) or math.isnan(test_auc):
            summary["overfit_flag"] = False
        else:
            summary["overfit_flag"] = (train_auc - test_auc) >= OVERFIT_DELTA
        final_model = clone(base_pipeline).fit(X, y)
        summary["coef"] = float(final_model.named_steps["model"].coef_[0][0])
        summary["intercept"] = float(final_model.named_steps["model"].intercept_[0])
        results.append(summary)
    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results).sort_values("test_roc_auc_mean", ascending=False)


def evaluate_model_zoo(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
    target_col: str = "aan_persistence",
    model_names: Optional[Sequence[str]] = None,
    repeats: int = 5,
    test_size: float = 0.3,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame], Dict[str, pd.DataFrame], Dict[str, str]]:
    usable_features = [c for c in feature_cols if c in df.columns]
    if not usable_features or target_col not in df.columns:
        return pd.DataFrame(), {}, {}, {}
    X = df[usable_features]
    y = df[target_col].astype(int)
    splits = list(_stratified_holdout_indices(y, repeats, test_size, random_state))
    if not splits:
        return pd.DataFrame(), {}, {}, {}
    builders = _model_builders(random_state)
    name_map = {_normalize_model_name(k): k for k in builders}
    if model_names:
        selected: List[str] = []
        for requested in model_names:
            key = _normalize_model_name(requested)
            if key not in name_map:
                raise ValueError(
                    f"Unknown model '{requested}'. Available: {', '.join(builders.keys())}"
                )
            selected.append(name_map[key])
    else:
        selected = list(builders.keys())

    metrics_rows = []
    feature_tables: Dict[str, pd.DataFrame] = {}
    split_tables: Dict[str, pd.DataFrame] = {}
    errors: Dict[str, str] = {}

    for name in selected:
        try:
            base_pipeline = builders[name]()
        except ImportError as exc:
            errors[name] = str(exc)
            continue

        train_blocks: List[Dict[str, float]] = []
        test_blocks: List[Dict[str, float]] = []
        split_records: List[Dict[str, float]] = []
        for split_id, (train_idx, test_idx) in enumerate(splits, start=1):
            pipeline = clone(base_pipeline)
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            pipeline.fit(X_train, y_train)
            train_scores = _score_probabilities(y_train, pipeline.predict_proba(X_train)[:, 1])
            test_scores = _score_probabilities(y_test, pipeline.predict_proba(X_test)[:, 1])
            train_blocks.append(train_scores)
            test_blocks.append(test_scores)
            split_records.append({"split": split_id, "stage": "train", **train_scores})
            split_records.append({"split": split_id, "stage": "test", **test_scores})
        summary = {"model": name}
        summary.update(_aggregate_metric_blocks(train_blocks, "train"))
        summary.update(_aggregate_metric_blocks(test_blocks, "test"))
        train_auc = summary.get("train_roc_auc_mean", float("nan"))
        test_auc = summary.get("test_roc_auc_mean", float("nan"))
        if math.isnan(train_auc) or math.isnan(test_auc):
            summary["overfit_flag"] = False
        else:
            summary["overfit_flag"] = (train_auc - test_auc) >= OVERFIT_DELTA
        try:
            final_pipeline = builders[name]()
            final_pipeline.fit(X, y)
            estimator = final_pipeline.named_steps["model"]
            if hasattr(estimator, "feature_importances_"):
                table = pd.DataFrame(
                    {
                        "feature": usable_features,
                        "importance": estimator.feature_importances_,
                    }
                ).sort_values("importance", ascending=False)
                feature_tables[name] = table
            elif hasattr(estimator, "coef_"):
                coefs = estimator.coef_[0]
                table = pd.DataFrame(
                    {
                        "feature": usable_features,
                        "coefficient": coefs,
                        "abs_coefficient": np.abs(coefs),
                    }
                ).sort_values("abs_coefficient", ascending=False)
                feature_tables[name] = table
        except Exception:
            pass
        metrics_rows.append(summary)
        split_tables[name] = pd.DataFrame(split_records)

    if not metrics_rows:
        return pd.DataFrame(), split_tables, feature_tables, errors
    metrics_df = pd.DataFrame(metrics_rows).sort_values("test_roc_auc_mean", ascending=False)
    return metrics_df, split_tables, feature_tables, errors

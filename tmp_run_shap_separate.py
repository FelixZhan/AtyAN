from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

FEATURE_LABELS = {
    "w1bs": "Body dissatisfaction",
    "w1dep": "Negative affect",
    "w1dres": "Dieting",
    "w1intbmi": "BMI",
    "w1socf": "Psychosocial functioning",
    "w1tii": "Thin-ideal internalization",
    "BE_w1": "Binge eating",
    "CB_w1": "Compensatory behaviors",
    "FAT_w1": "Feeling fat",
    "FEAR_w1": "Fear of weight gain",
    "LEB_w1": "Lower-than-expected BMI",
    "WSO_w1": "Weight/shape overvaluation",
    "cond_bp": "Intervention condition: BP (vs Control/HW)",
    "cond_hw": "Intervention condition: Healthy Weight (vs Control/BP)",
}

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

COND_CANONICAL = {"BP": "BP", "Control": "Control", "Healthy Weight": "Healthy Weight"}


def english_name(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature)


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


def _iter_imputer(seed: int = 42, posterior: bool = False) -> IterativeImputer:
    return IterativeImputer(
        random_state=int(seed),
        sample_posterior=bool(posterior),
        max_iter=10,
        initial_strategy="median",
        skip_complete=True,
    )


def _plot_label(feature: str) -> str:
    label = english_name(feature)
    return label.replace("Intervention condition:", "").strip()


def _mean_abs_importance(expl) -> np.ndarray:
    vals = getattr(expl, "values", expl)
    arr = np.asarray(vals)
    if arr.ndim == 3 and arr.shape[-1] >= 2:
        arr = arr[:, :, 1]
    return np.abs(arr).mean(axis=0)


def _save_top10_bar(df: pd.DataFrame, model_name: str, out_png: Path) -> None:
    top = df.sort_values("mean_abs_shap", ascending=False).head(10).copy()
    top = top.sort_values("mean_abs_shap", ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(top["feature_english"], top["mean_abs_shap"], height=0.6)
    plt.xlabel("Mean |SHAP value|")
    plt.title(f"{model_name} SHAP Importance (Top 10)")
    plt.tight_layout()
    plt.savefig(out_png, dpi=220)
    plt.close()


def main() -> None:
    data_pref = Path("AAN-ONSET-MERGED.csv")
    data_fallback = Path("BP1234-ONSET.csv")
    if data_pref.exists():
        raw_df = pd.read_csv(data_pref, low_memory=False)
        data_name = data_pref.name
    elif data_fallback.exists():
        raw_df = pd.read_csv(data_fallback, low_memory=False)
        data_name = data_fallback.name
    else:
        raise FileNotFoundError("AAN-ONSET-MERGED.csv or BP1234-ONSET.csv is required.")

    feature_df, feature_sets = engineer_baseline_features(raw_df)
    feature_sets["all_features"] = [
        c for c in feature_sets.get("all_features", []) if not str(c).endswith("-persistence")
    ]

    feature_df = clean_and_encode_condition(feature_df)
    cond_covars = ["cond_bp", "cond_hw"]
    feature_sets["condition_covariates"] = cond_covars
    feature_sets["all_features"] = list(dict.fromkeys(feature_sets["all_features"] + cond_covars))

    design_df = prepare_univariate_prediction_dataset(feature_df, feature_sets["all_features"])
    X = design_df[FEATURES_14].copy()
    y = design_df["aan_onset_anywave"].astype(int)

    print(f"Using dataset: {data_name}")
    print(f"Design shape: {X.shape}; prevalence={y.mean():.4f}")

    brf = Pipeline(
        [
            ("imputer", _iter_imputer(seed=42, posterior=False)),
            (
                "model",
                BalancedRandomForestClassifier(
                    random_state=42,
                    n_jobs=1,
                    n_estimators=400,
                    max_depth=3,
                    min_samples_leaf=4,
                    max_features="sqrt",
                ),
            ),
        ]
    )

    log = Pipeline(
        [
            ("imputer", _iter_imputer(seed=42, posterior=False)),
            ("scaler", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    random_state=42,
                    max_iter=3000,
                    class_weight="balanced",
                    C=0.01,
                    solver="lbfgs",
                    penalty="l2",
                ),
            ),
        ]
    )

    brf.fit(X, y)
    log.fit(X, y)

    feat_names = [_plot_label(c) for c in X.columns]

    brf_pre = brf[:-1]
    brf_model = brf.named_steps["model"]
    X_brf_t = pd.DataFrame(brf_pre.transform(X), columns=feat_names, index=X.index)

    log_pre = log[:-1]
    log_model = log.named_steps["model"]
    X_log_t = pd.DataFrame(log_pre.transform(X), columns=feat_names, index=X.index)

    brf_explainer = shap.TreeExplainer(brf_model)
    brf_shap_full = brf_explainer(X_brf_t)
    brf_shap = shap.Explanation(
        values=(
            brf_shap_full.values[..., 1]
            if np.asarray(brf_shap_full.values).ndim == 3
            else brf_shap_full.values
        ),
        base_values=(
            brf_shap_full.base_values[..., 1]
            if np.asarray(brf_shap_full.base_values).ndim == 2
            else brf_shap_full.base_values
        ),
        data=brf_shap_full.data,
        feature_names=feat_names,
    )

    log_explainer = shap.LinearExplainer(log_model, X_log_t)
    log_shap = log_explainer(X_log_t)

    brf_imp = pd.DataFrame(
        {
            "feature_english": feat_names,
            "mean_abs_shap": _mean_abs_importance(brf_shap),
        }
    ).sort_values("mean_abs_shap", ascending=False)
    log_imp = pd.DataFrame(
        {
            "feature_english": feat_names,
            "mean_abs_shap": _mean_abs_importance(log_shap),
        }
    ).sort_values("mean_abs_shap", ascending=False)

    out_dir = Path("shap_outputs")
    out_dir.mkdir(exist_ok=True)

    brf_csv = out_dir / "brf_shap_importance.csv"
    log_csv = out_dir / "logistic_shap_importance.csv"
    brf_imp.to_csv(brf_csv, index=False)
    log_imp.to_csv(log_csv, index=False)

    _save_top10_bar(brf_imp, "Balanced Random Forest", out_dir / "brf_shap_top10.png")
    _save_top10_bar(log_imp, "Logistic Regression", out_dir / "logistic_shap_top10.png")

    plt.figure(figsize=(12, 6))
    shap.plots.beeswarm(brf_shap, max_display=14, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "brf_shap_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(12, 6))
    shap.plots.beeswarm(log_shap, max_display=14, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "logistic_shap_beeswarm.png", dpi=220, bbox_inches="tight")
    plt.close()

    print("\nTop 10 BRF features by mean |SHAP|:")
    print(brf_imp.head(10).to_string(index=False))
    print("\nTop 10 Logistic features by mean |SHAP|:")
    print(log_imp.head(10).to_string(index=False))
    print(f"\nSaved SHAP outputs to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()

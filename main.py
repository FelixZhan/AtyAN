"""Command-line entry point for the AtyAN analyses."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Sequence

import pandas as pd

from analysis_utils import (
    build_diagnosis_masks,
    engineer_baseline_features,
    evaluate_multivariate_models,
    load_base_dataset,
    prepare_persistence_dataset,
    prepare_univariate_prediction_dataset,
    run_baseline_anova_contrasts,
    run_disorder_level_anova,
    run_univariate_logistic_regressions,
)


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_tables(tables: Iterable[tuple[str, pd.DataFrame]], directory: Path) -> None:
    for name, table in tables:
        if table is None or table.empty:
            continue
        table.to_csv(directory / f"{name}.csv", index=False)


def run_anova_segment(df: pd.DataFrame, feature_cols: Sequence[str], output_dir: Path) -> None:
    print("Running baseline ANOVA contrasts…")
    anova_dir = _ensure_dir(output_dir / "anova")
    masks = build_diagnosis_masks(df)
    baseline_tables = run_baseline_anova_contrasts(df, masks, feature_cols)
    _save_tables(baseline_tables.items(), anova_dir)

    print("Running disorder-level ANOVA (AAN vs. other diagnoses)…")
    disorder_table = run_disorder_level_anova(df, feature_cols)
    if disorder_table.empty:
        print("Disorder-level ANOVA returned no rows (insufficient data).")
    else:
        disorder_table.to_csv(anova_dir / "AAN_vs_other_diagnoses.csv", index=False)


def run_univariate_segment(df: pd.DataFrame, feature_cols: Sequence[str], output_dir: Path) -> None:
    print("Running univariate onset-prediction logistics…")
    uni_dir = _ensure_dir(output_dir / "univariate")
    onset_df = prepare_univariate_prediction_dataset(df, feature_cols)
    onset_results = run_univariate_logistic_regressions(
        onset_df, feature_cols, target_col="aan_onset_anywave"
    )
    if onset_results.empty:
        print("Univariate onset regression returned no rows (features missing).")
    else:
        onset_results.to_csv(uni_dir / "onset_logistic_metrics.csv", index=False)

    print("Running univariate persistence vs. remission logistics…")
    persistence_df = prepare_persistence_dataset(df, feature_cols)
    persistence_results = run_univariate_logistic_regressions(
        persistence_df, feature_cols, target_col="aan_persistence"
    )
    if persistence_results.empty:
        print("Univariate persistence regression returned no rows (features missing).")
    else:
        persistence_results.to_csv(uni_dir / "persistence_logistic_metrics.csv", index=False)


def run_multivariate_segment(df: pd.DataFrame, feature_cols: Sequence[str], output_dir: Path) -> None:
    print("Running multivariate models for persistence vs. remission…")
    multi_dir = _ensure_dir(output_dir / "multivariate")
    persistence_df = prepare_persistence_dataset(df, feature_cols)
    metrics, feature_tables = evaluate_multivariate_models(
        persistence_df, feature_cols, target_col="aan_persistence"
    )
    if metrics.empty:
        print("Multivariate evaluation returned no rows (insufficient data).")
    else:
        metrics.to_csv(multi_dir / "model_comparison.csv", index=False)
    for name, table in feature_tables.items():
        if table.empty:
            continue
        table.to_csv(multi_dir / f"feature_importances_{name}.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the reorganized AtyAN analyses without opening the notebooks. "
            "If no flags are provided, every segment is executed."
        )
    )
    parser.add_argument(
        "--run-anova",
        action="store_true",
        help="Execute the baseline and disorder-level ANOVA contrasts.",
    )
    parser.add_argument(
        "--run-univariate",
        action="store_true",
        help="Execute the univariate logistic regressions (onset and persistence).",
    )
    parser.add_argument(
        "--run-multivariate",
        action="store_true",
        help="Execute the multivariate persistence vs. remission models.",
    )
    parser.add_argument(
        "--output-dir",
        default="results",
        type=Path,
        help="Directory where CSV outputs will be written (default: %(default)s).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_all = not (args.run_anova or args.run_univariate or args.run_multivariate)

    df_raw = load_base_dataset()
    df_engineered, feature_sets = engineer_baseline_features(df_raw)
    feature_cols = feature_sets["all_features"]

    output_dir = _ensure_dir(Path(args.output_dir))

    if args.run_anova or run_all:
        run_anova_segment(df_engineered, feature_cols, output_dir)
    if args.run_univariate or run_all:
        run_univariate_segment(df_engineered, feature_cols, output_dir)
    if args.run_multivariate or run_all:
        run_multivariate_segment(df_engineered, feature_cols, output_dir)


if __name__ == "__main__":
    main()

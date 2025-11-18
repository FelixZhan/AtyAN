"""Command-line entry point for the AtyAN analyses."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

import pandas as pd

from analysis_utils import (
    available_model_names,
    build_diagnosis_masks,
    engineer_baseline_features,
    evaluate_model_zoo,
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


def _save_split_tables(split_tables: Dict[str, pd.DataFrame], directory: Path, prefix: str) -> None:
    for model_name, table in split_tables.items():
        if table is None or table.empty:
            continue
        table.to_csv(directory / f"{prefix}_split_metrics_{model_name}.csv", index=False)


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


def run_univariate_segment(
    df: pd.DataFrame,
    onset_feature_cols: Sequence[str],
    persistence_feature_cols: Sequence[str],
    output_dir: Path,
    model_names: Optional[Sequence[str]] = None,
    repeats: int = 5,
    test_size: float = 0.3,
) -> None:
    print("Running univariate onset-prediction logistics…")
    uni_dir = _ensure_dir(output_dir / "univariate")
    onset_df = prepare_univariate_prediction_dataset(df, onset_feature_cols)
    onset_results = run_univariate_logistic_regressions(
        onset_df,
        onset_feature_cols,
        target_col="aan_onset_anywave",
        repeats=repeats,
        test_size=test_size,
    )
    if onset_results.empty:
        print("Univariate onset regression returned no rows (features missing).")
    else:
        onset_results.to_csv(uni_dir / "onset_logistic_metrics.csv", index=False)

    print("Running univariate persistence vs. remission logistics…")
    persistence_df = prepare_persistence_dataset(df, persistence_feature_cols)
    persistence_results = run_univariate_logistic_regressions(
        persistence_df,
        persistence_feature_cols,
        target_col="aan_persistence",
        repeats=repeats,
        test_size=test_size,
    )
    if persistence_results.empty:
        print("Univariate persistence regression returned no rows (features missing).")
    else:
        persistence_results.to_csv(uni_dir / "persistence_logistic_metrics.csv", index=False)

    print("Running onset model zoo…")
    onset_metrics, onset_splits, _, onset_errors = evaluate_model_zoo(
        onset_df,
        onset_feature_cols,
        target_col="aan_onset_anywave",
        model_names=model_names,
        repeats=repeats,
        test_size=test_size,
    )
    if onset_metrics.empty:
        print("Model zoo could not run on the onset dataset (check errors above).")
    else:
        onset_metrics.to_csv(uni_dir / "onset_model_zoo_summary.csv", index=False)
        _save_split_tables(onset_splits, uni_dir, "onset")
    for model_name, message in onset_errors.items():
        print(f"[onset] Skipped {model_name}: {message}")

    print("Running persistence model zoo…")
    persistence_metrics, persistence_splits, feature_tables, persistence_errors = evaluate_model_zoo(
        persistence_df,
        persistence_feature_cols,
        target_col="aan_persistence",
        model_names=model_names,
        repeats=repeats,
        test_size=test_size,
    )
    if persistence_metrics.empty:
        print("Model zoo could not run on the persistence dataset (check errors above).")
    else:
        persistence_metrics.to_csv(uni_dir / "persistence_model_zoo_summary.csv", index=False)
        _save_split_tables(persistence_splits, uni_dir, "persistence")
        _save_tables(
            (
                (f"persistence_feature_table_{name}", table)
                for name, table in feature_tables.items()
            ),
            uni_dir,
        )
    for model_name, message in persistence_errors.items():
        print(f"[persistence] Skipped {model_name}: {message}")


def run_multivariate_segment(
    df: pd.DataFrame,
    persistence_feature_cols: Sequence[str],
    output_dir: Path,
    model_names: Optional[Sequence[str]] = None,
    repeats: int = 5,
    test_size: float = 0.3,
) -> None:
    print("Running multivariate models for persistence vs. remission…")
    multi_dir = _ensure_dir(output_dir / "multivariate")
    persistence_df = prepare_persistence_dataset(df, persistence_feature_cols)
    metrics, split_tables, feature_tables, errors = evaluate_model_zoo(
        persistence_df,
        persistence_feature_cols,
        target_col="aan_persistence",
        model_names=model_names,
        repeats=repeats,
        test_size=test_size,
    )
    if metrics.empty:
        print("Multivariate evaluation returned no rows (insufficient data).")
    else:
        metrics.to_csv(multi_dir / "model_comparison.csv", index=False)
        _save_split_tables(split_tables, multi_dir, "multivariate")
    for name, table in feature_tables.items():
        if table.empty:
            continue
        table.to_csv(multi_dir / f"feature_importances_{name}.csv", index=False)
    for model_name, message in errors.items():
        print(f"[multivariate] Skipped {model_name}: {message}")


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
        "--model-names",
        type=str,
        default=None,
        help=(
            "Comma-separated subset of models to run. Options: "
            + ", ".join(available_model_names())
        ),
    )
    parser.add_argument(
        "--holdout-repeats",
        type=int,
        default=5,
        help="Number of stratified repeated holdout splits (default: %(default)s).",
    )
    parser.add_argument(
        "--holdout-test-size",
        type=float,
        default=0.3,
        help="Test size for the repeated holdout splits (default: %(default)s).",
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

    model_names = None
    if args.model_names:
        model_names = [name.strip() for name in args.model_names.split(",") if name.strip()]

    df_raw = load_base_dataset()
    df_engineered, feature_sets = engineer_baseline_features(df_raw)
    anova_features = feature_sets.get("anova_features", [])
    onset_features = feature_sets.get("onset_features", [])
    persistence_features = feature_sets.get("persistence_features", [])

    output_dir = _ensure_dir(Path(args.output_dir))

    if args.run_anova or run_all:
        run_anova_segment(df_engineered, anova_features, output_dir)
    if args.run_univariate or run_all:
        run_univariate_segment(
            df_engineered,
            onset_features,
            persistence_features,
            output_dir,
            model_names=model_names,
            repeats=args.holdout_repeats,
            test_size=args.holdout_test_size,
        )
    if args.run_multivariate or run_all:
        run_multivariate_segment(
            df_engineered,
            persistence_features,
            output_dir,
            model_names=model_names,
            repeats=args.holdout_repeats,
            test_size=args.holdout_test_size,
        )


if __name__ == "__main__":
    main()

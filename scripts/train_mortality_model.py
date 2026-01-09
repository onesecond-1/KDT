"""Train early mortality prediction models for sepsis patients."""

from __future__ import annotations

import argparse

import pandas as pd

from kdt.modeling import train_mortality_models
from kdt.sepsis import label_sepsis
from kdt.sofa import compute_sofa_scores


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compute SOFA scores from early ICU data, label sepsis, "
            "and train mortality prediction models."
        )
    )
    parser.add_argument("--input", required=True, help="Path to input CSV file")
    parser.add_argument(
        "--target",
        default="mortality_28d",
        help="Target mortality label column (default: mortality_28d)",
    )
    parser.add_argument(
        "--infection-col",
        default="suspected_infection",
        help="Column indicating suspected infection",
    )
    parser.add_argument(
        "--group-col",
        default="subject_id",
        help="Optional patient group column",
    )
    parser.add_argument(
        "--test-size", type=float, default=0.2, help="Test split fraction"
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = compute_sofa_scores(df)
    df = label_sepsis(df, infection_col=args.infection_col)

    sepsis_df = df[df["sepsis_label"]].copy()
    if sepsis_df.empty:
        raise ValueError("No sepsis patients found after labeling.")

    feature_cols = [
        "sofa_total",
        "sofa_respiration",
        "sofa_coagulation",
        "sofa_liver",
        "sofa_cardiovascular",
        "sofa_cns",
        "sofa_renal",
    ]

    results, _ = train_mortality_models(
        sepsis_df,
        feature_cols=feature_cols,
        target_col=args.target,
        group_col=args.group_col,
        test_size=args.test_size,
    )

    for result in results:
        print(
            f"{result.model_name}: ROC-AUC={result.roc_auc:.3f}, "
            f"PR-AUC={result.pr_auc:.3f}"
        )


if __name__ == "__main__":
    main()

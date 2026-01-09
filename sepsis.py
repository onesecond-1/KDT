"""Sepsis labeling utilities."""

from __future__ import annotations

import pandas as pd


def label_sepsis(
    df: pd.DataFrame,
    infection_col: str = "suspected_infection",
    sofa_total_col: str = "sofa_total",
    output_col: str = "sepsis_label",
) -> pd.DataFrame:
    """Label sepsis using suspected infection + SOFA >= 2 (Sepsis-3).

    Args:
        df: Input dataframe with infection and SOFA columns.
        infection_col: Column indicating suspected infection (boolean or 0/1).
        sofa_total_col: Column with total SOFA score.
        output_col: Name of the output label column.
    """

    if infection_col not in df.columns:
        raise ValueError(f"Missing required infection column: {infection_col}")
    if sofa_total_col not in df.columns:
        raise ValueError(f"Missing required SOFA column: {sofa_total_col}")

    labeled = df.copy()
    labeled[output_col] = (labeled[infection_col].astype(bool)) & (
        labeled[sofa_total_col] >= 2
    )
    return labeled

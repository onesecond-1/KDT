"""SOFA score computation for early ICU data."""

from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = {
    "pao2",
    "fio2",
    "platelets",
    "bilirubin",
    "map",
    "gcs",
    "creatinine",
}

OPTIONAL_COLUMNS = {
    "ventilated",
    "dopamine",
    "dobutamine",
    "epinephrine",
    "norepinephrine",
    "urine_output",
}


def _require_columns(df: pd.DataFrame) -> None:
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for SOFA scoring: {sorted(missing)}")


def _score_respiration(df: pd.DataFrame) -> pd.Series:
    pf_ratio = df["pao2"] / df["fio2"]
    ventilated = df.get("ventilated", pd.Series(False, index=df.index)).astype(bool)

    score = pd.Series(0, index=df.index, dtype=int)
    score = score.mask(pf_ratio < 400, 1)
    score = score.mask(pf_ratio < 300, 2)
    score = score.mask((pf_ratio < 200) & ventilated, 3)
    score = score.mask((pf_ratio < 100) & ventilated, 4)
    return score


def _score_coagulation(df: pd.DataFrame) -> pd.Series:
    platelets = df["platelets"]
    score = pd.Series(0, index=df.index, dtype=int)
    score = score.mask(platelets < 150, 1)
    score = score.mask(platelets < 100, 2)
    score = score.mask(platelets < 50, 3)
    score = score.mask(platelets < 20, 4)
    return score


def _score_liver(df: pd.DataFrame) -> pd.Series:
    bili = df["bilirubin"]
    score = pd.Series(0, index=df.index, dtype=int)
    score = score.mask(bili >= 1.2, 1)
    score = score.mask(bili >= 2.0, 2)
    score = score.mask(bili >= 6.0, 3)
    score = score.mask(bili >= 12.0, 4)
    return score


def _score_cardiovascular(df: pd.DataFrame) -> pd.Series:
    map_val = df["map"]
    dopamine = df.get("dopamine", pd.Series(0.0, index=df.index))
    dobutamine = df.get("dobutamine", pd.Series(0.0, index=df.index))
    epinephrine = df.get("epinephrine", pd.Series(0.0, index=df.index))
    norepinephrine = df.get("norepinephrine", pd.Series(0.0, index=df.index))

    score = pd.Series(0, index=df.index, dtype=int)
    score = score.mask(map_val < 70, 1)

    score = score.mask((dopamine > 0) & (dopamine <= 5), 2)
    score = score.mask(dobutamine > 0, 2)

    score = score.mask((dopamine > 5) & (dopamine <= 15), 3)
    score = score.mask((epinephrine > 0) & (epinephrine <= 0.1), 3)
    score = score.mask((norepinephrine > 0) & (norepinephrine <= 0.1), 3)

    score = score.mask(dopamine > 15, 4)
    score = score.mask(epinephrine > 0.1, 4)
    score = score.mask(norepinephrine > 0.1, 4)

    return score


def _score_cns(df: pd.DataFrame) -> pd.Series:
    gcs = df["gcs"]
    score = pd.Series(0, index=df.index, dtype=int)
    score = score.mask(gcs < 15, 1)
    score = score.mask(gcs < 13, 2)
    score = score.mask(gcs < 10, 3)
    score = score.mask(gcs < 6, 4)
    return score


def _score_renal(df: pd.DataFrame) -> pd.Series:
    creatinine = df["creatinine"]
    urine_output = df.get("urine_output", pd.Series(np.nan, index=df.index))

    score_creat = pd.Series(0, index=df.index, dtype=int)
    score_creat = score_creat.mask(creatinine >= 1.2, 1)
    score_creat = score_creat.mask(creatinine >= 2.0, 2)
    score_creat = score_creat.mask(creatinine >= 3.5, 3)
    score_creat = score_creat.mask(creatinine >= 5.0, 4)

    score_urine = pd.Series(0, index=df.index, dtype=int)
    score_urine = score_urine.mask(urine_output < 500, 3)
    score_urine = score_urine.mask(urine_output < 200, 4)

    return pd.concat([score_creat, score_urine], axis=1).max(axis=1)


def compute_sofa_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Compute SOFA sub-scores and total score for each row.

    Expected columns:
      - pao2, fio2, platelets, bilirubin, map, gcs, creatinine
    Optional columns:
      - ventilated, dopamine, dobutamine, epinephrine, norepinephrine, urine_output
    """

    _require_columns(df)

    scored = df.copy()
    scored["sofa_respiration"] = _score_respiration(scored)
    scored["sofa_coagulation"] = _score_coagulation(scored)
    scored["sofa_liver"] = _score_liver(scored)
    scored["sofa_cardiovascular"] = _score_cardiovascular(scored)
    scored["sofa_cns"] = _score_cns(scored)
    scored["sofa_renal"] = _score_renal(scored)
    scored["sofa_total"] = scored[
        [
            "sofa_respiration",
            "sofa_coagulation",
            "sofa_liver",
            "sofa_cardiovascular",
            "sofa_cns",
            "sofa_renal",
        ]
    ].sum(axis=1)

    return scored

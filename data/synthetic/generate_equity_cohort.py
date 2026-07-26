"""Generate a synthetic Canadian chronic-disease cohort calibrated to
publicly documented health-equity gaps.

This produces **entirely synthetic** patient-level data — no real person's
records, and in particular no real Indigenous patient data, are used or
required. See ``docs/data-governance/ocap-and-privacy.md`` for why that
matters and ``data/synthetic/calibration_notes.md`` for the full rationale
behind every modeling choice below, including which gaps are precisely
calibrated to a published statistic and which are only directionally
modeled.

Two gaps are calibrated to specific, cited public figures:

- ``INDIGENOUS_POPULATION_SHARE`` (~5.0%) — Statistics Canada, 2021 Census
  of Population.
- ``REMOTE_TO_ACCESSIBLE_HOSPITALIZATION_RATE_RATIO`` (~1.294) — CIHI,
  national rural vs. urban hospitalization rate, 2024-2025
  ("Rural health care in Canada" indicator: 8.74 vs. 6.75 per 100
  residents).

A third, related gap — the ~9-year First Nations vs. non-Indigenous
life-expectancy difference reported by Statistics Canada
(catalogue 82-003-X, "Mortality among First Nations people, 2006 to
2016") — motivates the *direction and rough order of magnitude* of the
survival hazard ratio used here, but is not reproduced exactly: life
expectancy at birth and post-diagnosis survival time are different
constructs, and transplanting one onto the other would overstate this
generator's precision. See calibration_notes.md.

Run with: ``python data/synthetic/generate_equity_cohort.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent
SEED = 20260726

# --- Calibrated targets (see module docstring and calibration_notes.md) ---
INDIGENOUS_POPULATION_SHARE = 0.050
REMOTE_TO_ACCESSIBLE_HOSPITALIZATION_RATE_RATIO = 8.74 / 6.75
ACCESSIBLE_HOSPITALIZATION_RATE_PER_100 = 6.75

# --- Illustrative (directionally, not point, calibrated) parameters ---
INDIGENOUS_SURVIVAL_HAZARD_RATIO = 1.45
REMOTE_SURVIVAL_HAZARD_RATIO = 1.20
LATE_STAGE_BASE_PROBABILITY = 0.35
LATE_STAGE_REMOTENESS_LOGIT_SHIFT = 0.9
LATE_STAGE_INDIGENOUS_LOGIT_SHIFT = 0.5

REMOTENESS_LEVELS = ["easily_accessible", "accessible", "less_accessible", "remote", "very_remote"]
REMOTENESS_PROBABILITIES = [0.68, 0.16, 0.08, 0.05, 0.03]
REMOTE_LEVELS = {"remote", "very_remote"}

PROVINCES = ["ON", "QC", "BC", "AB", "MB", "SK", "NS", "NB", "NL", "PE", "YT", "NT", "NU"]
PROVINCE_PROBABILITIES = [
    0.385,
    0.225,
    0.135,
    0.115,
    0.036,
    0.029,
    0.026,
    0.021,
    0.014,
    0.004,
    0.004,
    0.003,
    0.003,
]

CONDITIONS = ["colorectal_cancer", "breast_cancer", "lung_cancer", "cardiovascular_event"]

WEIBULL_SHAPE = 1.1
WEIBULL_BASELINE_SCALE_DAYS = 365.25 * 6
ADMIN_CENSORING_DAYS = 365.25 * 8


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _calibrate_indigenous_intercept(
    remoteness_score: np.ndarray, slope: float, target_mean: float
) -> float:
    """Bisect for the logistic intercept that hits ``target_mean`` prevalence.

    Indigenous-proxy status is modeled as mildly more likely in remote
    areas (consistent with Statistics Canada geographic distributions),
    while the *overall* prevalence is pinned to the calibrated national
    figure.
    """
    lo, hi = -10.0, 10.0
    for _ in range(40):
        mid = (lo + hi) / 2
        mean_p = _logistic(mid + slope * remoteness_score).mean()
        if mean_p > target_mean:
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def generate_equity_cohort(n_patients: int = 4000, seed: int = SEED) -> pd.DataFrame:
    """Generate the synthetic equity-calibrated chronic-disease cohort.

    Parameters
    ----------
    n_patients : int, default 4000
        Number of synthetic patients to generate.
    seed : int, default SEED
        Seed for the random generator; the default reproduces the fixture
        committed to this repository.

    Returns
    -------
    pandas.DataFrame
        One row per synthetic patient. Columns include demographic and
        geographic fields (``age_at_diagnosis``, ``sex``, ``province``,
        ``remoteness_level``, ``remoteness_score``,
        ``indigenous_identity_proxy``, ``income_quintile``), clinical
        fields (``condition``, ``diagnosis_date``, ``stage_at_diagnosis``,
        ``hospitalizations_per_year``), and survival fields
        (``time_to_event_days``, ``event_observed``).

    Examples
    --------
    >>> df = generate_equity_cohort(n_patients=200, seed=1)
    >>> len(df)
    200
    >>> bool((df["remoteness_level"].isin(REMOTENESS_LEVELS)).all())
    True
    """
    rng = np.random.default_rng(seed)
    n = n_patients

    age_at_diagnosis = rng.normal(loc=64, scale=12, size=n).clip(18, 100)
    sex = rng.choice(["F", "M"], size=n)
    province = rng.choice(PROVINCES, size=n, p=PROVINCE_PROBABILITIES)

    remoteness_level = rng.choice(REMOTENESS_LEVELS, size=n, p=REMOTENESS_PROBABILITIES)
    level_rank = pd.Categorical(remoteness_level, categories=REMOTENESS_LEVELS, ordered=True).codes
    remoteness_score = (level_rank + rng.uniform(0, 1, size=n)) / len(REMOTENESS_LEVELS)
    is_remote = np.isin(remoteness_level, list(REMOTE_LEVELS))

    intercept = _calibrate_indigenous_intercept(
        remoteness_score, slope=1.5, target_mean=INDIGENOUS_POPULATION_SHARE
    )
    indigenous_p = _logistic(intercept + 1.5 * remoteness_score)
    indigenous_identity_proxy = rng.binomial(1, indigenous_p).astype(bool)

    income_logit = (
        1.0
        - 1.8 * remoteness_score
        - 0.6 * indigenous_identity_proxy.astype(float)
        + rng.normal(0, 0.8, size=n)
    )
    income_quintile = pd.qcut(income_logit, 5, labels=[1, 2, 3, 4, 5]).astype(int)

    hospitalization_rate = (
        np.where(
            is_remote,
            ACCESSIBLE_HOSPITALIZATION_RATE_PER_100
            * REMOTE_TO_ACCESSIBLE_HOSPITALIZATION_RATE_RATIO,
            ACCESSIBLE_HOSPITALIZATION_RATE_PER_100,
        )
        / 100.0
    )
    hospitalizations_per_year = rng.poisson(hospitalization_rate)

    condition = rng.choice(CONDITIONS, size=n)
    diagnosis_date = pd.Timestamp("2015-01-01") + pd.to_timedelta(
        rng.integers(0, 365 * 8, size=n), unit="D"
    )

    late_stage_logit = (
        np.log(LATE_STAGE_BASE_PROBABILITY / (1 - LATE_STAGE_BASE_PROBABILITY))
        + LATE_STAGE_REMOTENESS_LOGIT_SHIFT * remoteness_score
        + LATE_STAGE_INDIGENOUS_LOGIT_SHIFT * indigenous_identity_proxy.astype(float)
    )
    late_stage_p = _logistic(late_stage_logit)
    is_late_stage = rng.binomial(1, late_stage_p).astype(bool)
    stage_at_diagnosis = np.where(
        is_late_stage,
        rng.choice(["III", "IV"], size=n),
        rng.choice(["I", "II"], size=n),
    )

    log_hazard_ratio = (
        np.log(INDIGENOUS_SURVIVAL_HAZARD_RATIO) * indigenous_identity_proxy.astype(float)
        + np.log(REMOTE_SURVIVAL_HAZARD_RATIO) * remoteness_score
        + 0.35 * is_late_stage.astype(float)
        + 0.02 * (age_at_diagnosis - 64) / 10
    )
    scale = WEIBULL_BASELINE_SCALE_DAYS * np.exp(-log_hazard_ratio / WEIBULL_SHAPE)
    u = rng.uniform(1e-9, 1 - 1e-9, size=n)
    time_to_death_days = scale * (-np.log(u)) ** (1 / WEIBULL_SHAPE)

    time_to_event_days = np.minimum(time_to_death_days, ADMIN_CENSORING_DAYS)
    event_observed = time_to_death_days <= ADMIN_CENSORING_DAYS

    return pd.DataFrame(
        {
            "patient_id": np.arange(1, n + 1),
            "age_at_diagnosis": age_at_diagnosis.round(1),
            "sex": sex,
            "province": province,
            "remoteness_level": remoteness_level,
            "remoteness_score": remoteness_score.round(4),
            "indigenous_identity_proxy": indigenous_identity_proxy,
            "income_quintile": income_quintile,
            "condition": condition,
            "diagnosis_date": diagnosis_date,
            "stage_at_diagnosis": stage_at_diagnosis,
            "hospitalizations_per_year": hospitalizations_per_year,
            "time_to_event_days": time_to_event_days.round(1),
            "event_observed": event_observed,
        }
    )


def calibration_report(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """Compare the generated cohort's key statistics against their calibration targets.

    Returns
    -------
    dict[str, dict[str, float]]
        For each calibrated quantity: ``observed``, ``target``, and
        ``relative_error``.

    Examples
    --------
    >>> df = generate_equity_cohort(n_patients=4000, seed=1)
    >>> report = calibration_report(df)
    >>> report["indigenous_population_share"]["relative_error"] < 0.15
    True
    """
    observed_indigenous_share = float(df["indigenous_identity_proxy"].mean())

    accessible_rate = df.loc[
        df["remoteness_level"] == "easily_accessible", "hospitalizations_per_year"
    ].mean()
    remote_rate = df.loc[
        df["remoteness_level"].isin(REMOTE_LEVELS), "hospitalizations_per_year"
    ].mean()
    observed_ratio = float(remote_rate / accessible_rate) if accessible_rate else float("nan")

    def _relative_error(observed: float, target: float) -> float:
        return abs(observed - target) / target

    return {
        "indigenous_population_share": {
            "observed": observed_indigenous_share,
            "target": INDIGENOUS_POPULATION_SHARE,
            "relative_error": _relative_error(
                observed_indigenous_share, INDIGENOUS_POPULATION_SHARE
            ),
        },
        "remote_to_accessible_hospitalization_rate_ratio": {
            "observed": observed_ratio,
            "target": REMOTE_TO_ACCESSIBLE_HOSPITALIZATION_RATE_RATIO,
            "relative_error": _relative_error(
                observed_ratio, REMOTE_TO_ACCESSIBLE_HOSPITALIZATION_RATE_RATIO
            ),
        },
    }


if __name__ == "__main__":
    cohort = generate_equity_cohort(n_patients=4000, seed=SEED)
    cohort.to_parquet(OUT_DIR / "equity_cohort.parquet", index=False)
    cohort.head(50).to_csv(OUT_DIR / "equity_cohort_preview.csv", index=False)

    report = calibration_report(cohort)
    lines = [
        "# Calibration report (generated)",
        "",
        "Produced by `generate_equity_cohort.py`; do not hand-edit — regenerate instead.",
        "",
        "| Quantity | Observed | Target | Relative error |",
        "|---|---|---|---|",
    ]
    for name, stats in report.items():
        lines.append(
            f"| {name} | {stats['observed']:.4f} | {stats['target']:.4f} "
            f"| {stats['relative_error']:.2%} |"
        )
    (OUT_DIR / "calibration_report.md").write_text("\n".join(lines) + "\n")

    print(f"Wrote {len(cohort)}-row synthetic equity cohort to {OUT_DIR}")
    print("\n".join(lines))

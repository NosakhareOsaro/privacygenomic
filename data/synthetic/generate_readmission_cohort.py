"""Synthetic 30-day readmission cohort, with an injected demographic disparity.

Unlike ``generate_equity_cohort.py`` (calibrated where possible to real,
cited public statistics — see ``calibration_notes.md``), this generator
exists to test *fairness auditing tooling itself*: the outcome label
below has a **known, injected** disparity across `remoteness_level` and
`indigenous_identity_proxy`, so `equitable_rwe.fairness.audit_fairness`
and the readmission model in `equitable_rwe.ml` can be checked against a
ground truth ("does the audit correctly detect the disparity we put in?"),
the same benchmarking logic used in `benchmark/`.

The clinical/utilization features (`n_prior_admissions_12mo`,
`length_of_stay_days`, `n_comorbidities`, `discharge_disposition`,
`polypharmacy_count`) drive the readmission model's predictions.
`indigenous_identity_proxy` and `remoteness_level` are **deliberately
excluded** from the model's feature set — mirroring realistic practice of
not feeding protected or proxy-for-protected attributes into a clinical
risk model — while still being used to *generate* the label via a
simulated reduced-follow-up-access effect, and used *only* for the
post-hoc fairness audit. This reproduces, in miniature, the well-documented
phenomenon of an algorithm being disparate in its effect on a group
without that group's membership ever being a model input (see Obermeyer,
Powers, Vogeli & Mullainathan, "Dissecting racial bias in an algorithm
used to manage the health of populations," Science, 2019, for the
real-world finding this generator is illustrating, not reproducing).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent
SEED = 20260726
N_PATIENTS = 4000

REMOTENESS_LEVELS = ["easily_accessible", "accessible", "less_accessible", "remote", "very_remote"]
REMOTENESS_PROBABILITIES = [0.68, 0.16, 0.08, 0.05, 0.03]
DISCHARGE_DISPOSITIONS = ["home", "home_with_services", "skilled_nursing_facility"]

#: Model-input clinical/utilization features (protected/proxy attributes
#: deliberately excluded — see module docstring).
FEATURE_COLUMNS = [
    "age",
    "n_prior_admissions_12mo",
    "length_of_stay_days",
    "n_comorbidities",
    "discharge_disposition_home_with_services",
    "discharge_disposition_skilled_nursing_facility",
    "polypharmacy_count",
    "income_quintile",
]

LABEL_COLUMN = "readmitted_30d"

#: Injected, known disparity (see module docstring): remote patients and
#: the Indigenous-identity proxy group face reduced post-discharge
#: follow-up access in this simulation, raising true readmission risk
#: independent of clinical severity.
REMOTENESS_LOGIT_SHIFT = 0.55
INDIGENOUS_LOGIT_SHIFT = 0.45


def _logistic(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def generate_readmission_cohort(n_patients: int = N_PATIENTS, seed: int = SEED) -> pd.DataFrame:
    """Generate the synthetic 30-day readmission cohort.

    Parameters
    ----------
    n_patients : int, default 4000
        Number of synthetic patients.
    seed : int, default 20260726
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Model-input feature columns (see
        :data:`FEATURE_COLUMNS`), the binary label
        (:data:`LABEL_COLUMN`), and subgroup columns
        (``remoteness_level``, ``indigenous_identity_proxy``,
        ``income_quintile``) kept for fairness auditing but excluded from
        :data:`FEATURE_COLUMNS`.

    Examples
    --------
    >>> df = generate_readmission_cohort(n_patients=200, seed=1)
    >>> len(df)
    200
    >>> set(FEATURE_COLUMNS).issubset(df.columns)
    True
    """
    rng = np.random.default_rng(seed)
    n = n_patients

    age = rng.normal(66, 14, size=n).clip(18, 100)
    remoteness_level = rng.choice(REMOTENESS_LEVELS, size=n, p=REMOTENESS_PROBABILITIES)
    remoteness_rank = pd.Categorical(
        remoteness_level, categories=REMOTENESS_LEVELS, ordered=True
    ).codes
    remoteness_score = remoteness_rank / (len(REMOTENESS_LEVELS) - 1)

    indigenous_identity_proxy = rng.binomial(1, 0.05, size=n).astype(bool)
    income_quintile = rng.integers(1, 6, size=n)

    n_comorbidities = rng.poisson(1.5, size=n).clip(0, 8)
    n_prior_admissions_12mo = rng.poisson(0.6 + 0.3 * n_comorbidities, size=n)
    length_of_stay_days = (rng.gamma(shape=2.0, scale=1.5, size=n) + 0.5 * n_comorbidities).round(1)
    discharge_disposition = rng.choice(DISCHARGE_DISPOSITIONS, size=n, p=[0.65, 0.25, 0.10])
    polypharmacy_count = rng.poisson(3 + 0.8 * n_comorbidities, size=n)

    clinical_logit = (
        -2.2
        + 0.35 * n_prior_admissions_12mo
        + 0.25 * n_comorbidities
        + 0.05 * length_of_stay_days
        + 0.04 * polypharmacy_count
        + 0.15 * (discharge_disposition == "skilled_nursing_facility")
        + 0.08 * (discharge_disposition == "home_with_services")
        - 0.08 * (income_quintile - 3)
        + 0.01 * (age - 66) / 10
    )
    disparity_logit = (
        REMOTENESS_LOGIT_SHIFT * remoteness_score
        + INDIGENOUS_LOGIT_SHIFT * indigenous_identity_proxy.astype(float)
    )
    readmit_p = _logistic(clinical_logit + disparity_logit)
    readmitted_30d = rng.binomial(1, readmit_p)

    df = pd.DataFrame(
        {
            "patient_id": np.arange(1, n + 1),
            "age": age.round(1),
            "n_prior_admissions_12mo": n_prior_admissions_12mo,
            "length_of_stay_days": length_of_stay_days,
            "n_comorbidities": n_comorbidities,
            "discharge_disposition": discharge_disposition,
            "polypharmacy_count": polypharmacy_count,
            "income_quintile": income_quintile,
            "remoteness_level": remoteness_level,
            "indigenous_identity_proxy": indigenous_identity_proxy,
            LABEL_COLUMN: readmitted_30d,
        }
    )
    df["discharge_disposition_home_with_services"] = (
        df["discharge_disposition"] == "home_with_services"
    ).astype(int)
    df["discharge_disposition_skilled_nursing_facility"] = (
        df["discharge_disposition"] == "skilled_nursing_facility"
    ).astype(int)
    return df


if __name__ == "__main__":
    cohort = generate_readmission_cohort()
    cohort.to_parquet(OUT_DIR / "readmission_cohort.parquet", index=False)
    cohort.head(50).to_csv(OUT_DIR / "readmission_cohort_preview.csv", index=False)
    print(f"Wrote {len(cohort)}-row synthetic readmission cohort to {OUT_DIR}")
    print(f"Overall 30-day readmission rate: {cohort[LABEL_COLUMN].mean():.3f}")
    print(
        "By indigenous_identity_proxy:\n",
        cohort.groupby("indigenous_identity_proxy")[LABEL_COLUMN].mean(),
    )
    print(
        "By remoteness_level:\n",
        cohort.groupby("remoteness_level")[LABEL_COLUMN].mean(),
    )

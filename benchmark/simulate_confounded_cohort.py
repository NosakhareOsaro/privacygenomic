"""A confounded-treatment-assignment synthetic cohort with a known,
exact ground-truth causal effect.

Unlike ``data/synthetic/generate_equity_cohort.py`` (calibrated to real
public health-equity statistics for illustrative demonstration), this
generator exists purely to evaluate causal-adjustment *estimators*
against a DGP where the true effect is exactly known by construction —
the standard technique for benchmarking causal inference methods (there
is no "real" calibration target here; the ground truth is the simulation
parameters themselves).

Design: treatment assignment is confounded by two covariates
(``x1``, ``x2``) that also independently affect the survival outcome —
classic confounding by indication. Survival times follow a Weibull
proportional-hazards model, so the *exact* population hazard ratio for
treatment is ``exp(TRUE_LOG_HR_TREATMENT)`` by construction, with no
approximation. :func:`true_risk_ratio_at_horizon` additionally computes,
by Monte Carlo simulation under the same DGP, the population-standardized
risk ratio at a fixed horizon — the estimand DoWhy's binary-outcome
backdoor estimator targets, which is not the same quantity as the hazard
ratio (see ``docs/architecture/adr-0003-causal-model-choices.md``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

TRUE_HR_TREATMENT = 0.65
TRUE_LOG_HR_TREATMENT = float(np.log(TRUE_HR_TREATMENT))
LOG_HR_X1 = float(np.log(1.8))
LOG_HR_X2 = float(np.log(1.3))

PROPENSITY_INTERCEPT = 0.0
PROPENSITY_COEF_X1 = -0.9  # confounding by indication: higher x1 -> less likely treated

WEIBULL_SHAPE = 1.2
BASELINE_SCALE_DAYS = 400.0
ADMIN_CENSORING_DAYS = 730.0
HORIZON_DAYS = 365.0  # must be < ADMIN_CENSORING_DAYS (see module docstring)


def _draw_confounders(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    return x1, x2


def _draw_treatment(x1: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    logit = PROPENSITY_INTERCEPT + PROPENSITY_COEF_X1 * x1
    p_treat = 1 / (1 + np.exp(-logit))
    return rng.binomial(1, p_treat)


def _draw_survival_times(
    treat: np.ndarray, x1: np.ndarray, x2: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    log_hazard_ratio = TRUE_LOG_HR_TREATMENT * treat + LOG_HR_X1 * x1 + LOG_HR_X2 * x2
    scale = BASELINE_SCALE_DAYS * np.exp(-log_hazard_ratio / WEIBULL_SHAPE)
    u = rng.uniform(1e-9, 1 - 1e-9, size=len(treat))
    return scale * (-np.log(u)) ** (1 / WEIBULL_SHAPE)


def simulate_confounded_cohort(n: int = 3000, seed: int = 2026) -> pd.DataFrame:
    """Simulate a cohort with confounded treatment assignment and a known true effect.

    Parameters
    ----------
    n : int, default 3000
        Number of subjects.
    seed : int, default 2026
        Random seed.

    Returns
    -------
    pandas.DataFrame
        Columns: ``patient_id``, ``x1``, ``x2``, ``treat``,
        ``time_to_event_days``, ``event_observed``, ``event_by_horizon``
        (binary indicator of an event by :data:`HORIZON_DAYS`, for the
        DoWhy backdoor estimator).

    Examples
    --------
    >>> df = simulate_confounded_cohort(n=200, seed=1)
    >>> len(df)
    200
    >>> sorted(df["treat"].unique().tolist())
    [0, 1]
    """
    rng = np.random.default_rng(seed)
    x1, x2 = _draw_confounders(n, rng)
    treat = _draw_treatment(x1, rng)
    time_to_death = _draw_survival_times(treat, x1, x2, rng)

    time_to_event = np.minimum(time_to_death, ADMIN_CENSORING_DAYS)
    event_observed = time_to_death <= ADMIN_CENSORING_DAYS
    event_by_horizon = time_to_death <= HORIZON_DAYS

    return pd.DataFrame(
        {
            "patient_id": np.arange(1, n + 1),
            "x1": x1,
            "x2": x2,
            "treat": treat,
            "time_to_event_days": time_to_event.round(2),
            "event_observed": event_observed.astype(int),
            "event_by_horizon": event_by_horizon.astype(int),
        }
    )


def true_risk_ratio_at_horizon(
    horizon: float = HORIZON_DAYS, n_monte_carlo: int = 300_000, seed: int = 999
) -> float:
    """Population-standardized true risk ratio at ``horizon``, by Monte Carlo.

    Draws confounders from their marginal distribution, then simulates
    the *same* population's survival times under the counterfactual
    ``do(treat=1)`` and ``do(treat=0)`` interventions, and returns
    ``P(T <= horizon | do(treat=1)) / P(T <= horizon | do(treat=0))``.

    Examples
    --------
    >>> rr = true_risk_ratio_at_horizon(n_monte_carlo=20000, seed=1)
    >>> rr < 1
    True
    """
    rng = np.random.default_rng(seed)
    x1, x2 = _draw_confounders(n_monte_carlo, rng)

    t_treated = _draw_survival_times(np.ones(n_monte_carlo), x1, x2, rng)
    t_control = _draw_survival_times(np.zeros(n_monte_carlo), x1, x2, rng)

    risk_treated = float((t_treated <= horizon).mean())
    risk_control = float((t_control <= horizon).mean())
    return risk_treated / risk_control


if __name__ == "__main__":
    cohort = simulate_confounded_cohort()
    print(f"Simulated {len(cohort)} subjects.")
    print(f"True hazard ratio (treatment): {TRUE_HR_TREATMENT}")
    print(f"True risk ratio at {HORIZON_DAYS:.0f} days: {true_risk_ratio_at_horizon():.4f}")

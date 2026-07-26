import numpy as np
import pandas as pd
import pytest

import equitable_rwe
from equitable_rwe import fit_equitable_survival_estimate


@pytest.fixture
def cohort() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 1200
    x1 = rng.normal(size=n)
    treat = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x1)))
    subgroup = rng.choice(["majority", "minority"], size=n, p=[0.8, 0.2])
    duration = rng.exponential(30 * np.exp(0.3 * treat + 0.3 * x1))
    return pd.DataFrame(
        {
            "T": duration,
            "E": rng.binomial(1, 0.9, n),
            "treat": treat,
            "x1": x1,
            "group": subgroup,
        }
    )


def test_available_from_top_level_package():
    assert equitable_rwe.fit_equitable_survival_estimate is fit_equitable_survival_estimate


def test_returns_both_causal_estimate_and_fairness_report(cohort):
    result = fit_equitable_survival_estimate(cohort, "T", "E", "treat", ["x1"], "group")
    assert result.causal_estimate is not None
    assert result.fairness_report is not None


def test_hazard_ratio_shortcut_matches_causal_estimate(cohort):
    result = fit_equitable_survival_estimate(cohort, "T", "E", "treat", ["x1"], "group")
    assert result.hazard_ratio == result.causal_estimate.hazard_ratio


def test_is_equitable_matches_fairness_report(cohort):
    result = fit_equitable_survival_estimate(cohort, "T", "E", "treat", ["x1"], "group")
    assert result.is_equitable == result.fairness_report.all_subgroups_adequately_supported

import numpy as np
import pandas as pd
import pytest

from equitable_rwe.survival.royston_parmar import fit_royston_parmar


@pytest.fixture
def aft_df() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 500
    x1 = rng.normal(size=n)
    duration = rng.weibull(1.3, n) * 50 * np.exp(0.5 * x1)
    event = rng.binomial(1, 0.85, n)
    return pd.DataFrame({"T": duration, "E": event, "x1": x1})


def test_fit_royston_parmar_recovers_effect_sign(aft_df):
    result = fit_royston_parmar(aft_df, "T", "E", covariates=["x1"], n_baseline_knots=3)
    beta_x1 = result.summary.loc[("beta_", "x1"), "coef"]
    assert beta_x1 > 0
    assert result.n_baseline_knots == 3


def test_median_survival_time_is_positive(aft_df):
    result = fit_royston_parmar(aft_df, "T", "E", covariates=["x1"])
    assert result.median_survival_time() > 0


def test_fewer_knots_still_fits(aft_df):
    result = fit_royston_parmar(aft_df, "T", "E", covariates=["x1"], n_baseline_knots=2)
    assert result.n_baseline_knots == 2
    assert ("beta_", "x1") in result.summary.index

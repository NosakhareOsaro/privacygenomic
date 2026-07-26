import numpy as np
import pandas as pd

from equitable_rwe.survival.cox import check_proportional_hazards, fit_cox_ph


def test_fit_cox_ph_recovers_known_effect_sign():
    rng = np.random.default_rng(3)
    n = 600
    x1 = rng.normal(size=n)
    duration = rng.exponential(50 * np.exp(0.6 * x1))
    event = rng.binomial(1, 0.9, n)
    df = pd.DataFrame({"T": duration, "E": event, "x1": x1})
    cph = fit_cox_ph(df, "T", "E", covariates=["x1"])
    # numpy's exponential(scale=...) parameterizes by mean, so larger x1
    # here means *longer* mean survival, i.e. a *lower* hazard: the Cox
    # coefficient should be negative and significant.
    assert cph.params_["x1"] < 0
    assert cph.summary.loc["x1", "p"] < 0.01


def test_fit_cox_ph_defaults_to_all_other_columns_as_covariates(survival_df):
    cph = fit_cox_ph(survival_df, "T", "E")
    assert list(cph.params_.index) == ["x1"]


def test_fit_cox_ph_with_weights_and_robust():
    rng = np.random.default_rng(4)
    n = 400
    x1 = rng.normal(size=n)
    df = pd.DataFrame(
        {
            "T": rng.exponential(50, n),
            "E": rng.binomial(1, 0.8, n),
            "x1": x1,
            "w": rng.uniform(0.5, 2.0, n),
        }
    )
    cph = fit_cox_ph(df, "T", "E", covariates=["x1"], weights_col="w", robust=True)
    assert "x1" in cph.params_.index


def test_check_proportional_hazards_returns_result_for_every_covariate(survival_df):
    cph = fit_cox_ph(survival_df, "T", "E", covariates=["x1"])
    result = check_proportional_hazards(cph, survival_df, "T")
    assert set(result.p_values) == {"x1"}
    assert 0 <= result.p_values["x1"] <= 1
    assert isinstance(result.any_violated, bool)


def test_check_proportional_hazards_violation_is_detected():
    # Construct data with a strongly time-varying effect: x1 matters early
    # but not late, which should show up as a significant residual/time
    # correlation.
    rng = np.random.default_rng(5)
    n = 800
    x1 = rng.normal(size=n)
    early = rng.exponential(10 * np.exp(2.0 * x1))
    late = 200 + rng.exponential(10, n)
    duration = np.where(rng.random(n) < 0.5, early, late)
    event = np.ones(n)
    df = pd.DataFrame({"T": duration, "E": event, "x1": x1})
    cph = fit_cox_ph(df, "T", "E", covariates=["x1"])
    result = check_proportional_hazards(cph, df, "T")
    assert result.p_values["x1"] < 0.05

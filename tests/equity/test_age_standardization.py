import numpy as np
import pandas as pd
import pytest

from equitable_rwe.equity.age_standardization import direct_standardize, fit_poisson_rate_model


def test_direct_standardize_matches_manual_calculation():
    counts = pd.Series({"young": 10, "old": 50})
    populations = pd.Series({"young": 1000, "old": 1000})
    standard = pd.Series({"young": 500, "old": 500})
    result = direct_standardize(counts, populations, standard)
    expected = 0.5 * (10 / 1000) + 0.5 * (50 / 1000)
    assert result.rate == pytest.approx(expected)


def test_direct_standardize_weights_toward_larger_standard_group():
    counts = pd.Series({"young": 10, "old": 50})
    populations = pd.Series({"young": 1000, "old": 1000})
    standard_young_heavy = pd.Series({"young": 900, "old": 100})
    standard_old_heavy = pd.Series({"young": 100, "old": 900})
    young_heavy = direct_standardize(counts, populations, standard_young_heavy)
    old_heavy = direct_standardize(counts, populations, standard_old_heavy)
    assert young_heavy.rate < old_heavy.rate


def test_direct_standardize_ci_brackets_the_rate():
    counts = pd.Series({"young": 10, "old": 50})
    populations = pd.Series({"young": 1000, "old": 1000})
    standard = pd.Series({"young": 500, "old": 500})
    result = direct_standardize(counts, populations, standard)
    assert result.ci_lower <= result.rate <= result.ci_upper


def test_direct_standardize_ci_lower_is_nonnegative():
    counts = pd.Series({"young": 0, "old": 1})
    populations = pd.Series({"young": 1000, "old": 1000})
    standard = pd.Series({"young": 999, "old": 1})
    result = direct_standardize(counts, populations, standard)
    assert result.ci_lower >= 0


def test_fit_poisson_rate_model_recovers_known_rate_ratio():
    # group b's rate is exactly 2x group a's by construction; several
    # strata per group avoid a saturated (perfectly separated) model.
    rng = np.random.default_rng(0)
    n_strata = 20
    df = pd.DataFrame(
        {
            "count": np.concatenate(
                [
                    rng.poisson(10, n_strata),  # rate 0.01/person, group a
                    rng.poisson(20, n_strata),  # rate 0.02/person, group b
                ]
            ),
            "population": 1000,
            "group": ["a"] * n_strata + ["b"] * n_strata,
        }
    )
    result = fit_poisson_rate_model(df, "count", "population", ["group"])
    rate_ratio = np.exp(result.params["group[T.b]"])
    assert rate_ratio == pytest.approx(2.0, rel=0.2)


def test_fit_poisson_rate_model_adjusts_for_multiple_predictors():
    df = pd.DataFrame(
        {
            "count": [10, 20, 15, 30],
            "population": [1000, 1000, 1000, 1000],
            "group": ["a", "a", "b", "b"],
            "age_band": ["young", "old", "young", "old"],
        }
    )
    result = fit_poisson_rate_model(df, "count", "population", ["group", "age_band"])
    assert {"group[T.b]", "age_band[T.young]"}.issubset(set(result.params.index))

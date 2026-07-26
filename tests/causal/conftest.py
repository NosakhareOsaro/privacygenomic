import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def confounded_survival_df() -> pd.DataFrame:
    """A cohort with treatment confounded by x1: higher x1 -> more likely
    treated AND (independently) longer survival, so a naive comparison is
    biased and adjustment should move the estimate toward the true
    (negative, i.e. protective) treatment coefficient.
    """
    # numpy's exponential(scale=...) is parameterized by the mean, so a
    # *positive* coefficient inside the scale expression means *longer*
    # survival, i.e. a *lower* hazard (protective).
    rng = np.random.default_rng(11)
    n = 800
    x1 = rng.normal(size=n)
    treat = rng.binomial(1, 1 / (1 + np.exp(-1.0 * x1)))
    duration = rng.exponential(30 * np.exp(0.5 * treat + 0.8 * x1))
    event = rng.binomial(1, 0.9, n)
    return pd.DataFrame({"T": duration, "E": event, "treat": treat, "x1": x1})


@pytest.fixture
def confounded_binary_df() -> pd.DataFrame:
    rng = np.random.default_rng(12)
    n = 600
    x1 = rng.normal(size=n)
    treat = rng.binomial(1, 1 / (1 + np.exp(0.3 * x1)))
    outcome = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 * treat + 0.4 * x1))))
    return pd.DataFrame({"treat": treat, "x1": x1, "outcome": outcome})

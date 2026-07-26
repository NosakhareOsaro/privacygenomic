import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def survival_df() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    n = 400
    x1 = rng.normal(size=n)
    duration = rng.exponential(50 * np.exp(-0.5 * x1))
    event = rng.binomial(1, 0.85, n)
    return pd.DataFrame({"T": duration, "E": event, "x1": x1})


@pytest.fixture
def two_group_survival_df() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    n_per_group = 200
    group_a = pd.DataFrame(
        {
            "T": rng.exponential(150, n_per_group),
            "E": np.ones(n_per_group),
            "group": "A",
        }
    )
    group_b = pd.DataFrame(
        {
            "T": rng.exponential(90, n_per_group),
            "E": np.ones(n_per_group),
            "group": "B",
        }
    )
    return pd.concat([group_a, group_b], ignore_index=True)


@pytest.fixture
def competing_risks_df() -> pd.DataFrame:
    rng = np.random.default_rng(2)
    n = 500
    x1 = rng.normal(size=n)
    duration = rng.exponential(20, n)
    event = rng.choice([0, 1, 2], size=n, p=[0.2, 0.5, 0.3])
    return pd.DataFrame({"T": duration, "event": event, "x1": x1})

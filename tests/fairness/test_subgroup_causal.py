import numpy as np
import pandas as pd
import pytest

from equitable_rwe.fairness.subgroup_causal import (
    MIN_EVENTS_PER_SUBGROUP,
    audit_causal_subgroup_fairness,
)


@pytest.fixture
def well_supported_df():
    rng = np.random.default_rng(0)
    n = 1200
    x1 = rng.normal(size=n)
    treat = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x1)))
    subgroup = rng.choice(["majority", "minority"], size=n, p=[0.7, 0.3])
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


@pytest.fixture
def sparse_subgroup_df():
    rng = np.random.default_rng(0)
    n_majority = 800
    n_tiny = 8  # far fewer events than MIN_EVENTS_PER_SUBGROUP
    x1_majority = rng.normal(size=n_majority)
    treat_majority = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x1_majority)))
    duration_majority = rng.exponential(30 * np.exp(0.3 * treat_majority + 0.3 * x1_majority))

    x1_tiny = rng.normal(size=n_tiny)
    treat_tiny = rng.binomial(1, 0.5, n_tiny)
    duration_tiny = rng.exponential(30, n_tiny)

    df = pd.DataFrame(
        {
            "T": np.concatenate([duration_majority, duration_tiny]),
            "E": np.concatenate([rng.binomial(1, 0.9, n_majority), np.zeros(n_tiny)]),
            "treat": np.concatenate([treat_majority, treat_tiny]),
            "x1": np.concatenate([x1_majority, x1_tiny]),
            "group": ["majority"] * n_majority + ["tiny"] * n_tiny,
        }
    )
    return df


def test_per_subgroup_covers_every_group(well_supported_df):
    report = audit_causal_subgroup_fairness(well_supported_df, "T", "E", "treat", ["x1"], "group")
    assert set(report.per_subgroup["subgroup"]) == {"majority", "minority"}


def test_well_supported_subgroups_are_not_flagged(well_supported_df):
    report = audit_causal_subgroup_fairness(well_supported_df, "T", "E", "treat", ["x1"], "group")
    assert report.all_subgroups_adequately_supported


def test_sparse_subgroup_is_flagged_for_too_few_events(sparse_subgroup_df):
    report = audit_causal_subgroup_fairness(sparse_subgroup_df, "T", "E", "treat", ["x1"], "group")
    assert "tiny" in report.flagged_subgroups
    tiny_row = report.per_subgroup.set_index("subgroup").loc["tiny"]
    assert tiny_row["n_events"] < MIN_EVENTS_PER_SUBGROUP
    assert pd.isna(tiny_row["hazard_ratio"])


def test_overall_hazard_ratio_is_positive(well_supported_df):
    report = audit_causal_subgroup_fairness(well_supported_df, "T", "E", "treat", ["x1"], "group")
    assert report.overall_hazard_ratio > 0

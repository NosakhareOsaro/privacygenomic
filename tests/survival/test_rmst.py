import numpy as np

from equitable_rwe.survival.rmst import compute_rmst, rmst_difference


def test_compute_rmst_is_bounded_by_tau(survival_df):
    tau = 20.0
    rmst = compute_rmst(survival_df["T"], survival_df["E"], tau=tau)
    assert 0 < rmst <= tau


def test_compute_rmst_increases_with_tau(survival_df):
    small = compute_rmst(survival_df["T"], survival_df["E"], tau=10)
    large = compute_rmst(survival_df["T"], survival_df["E"], tau=50)
    assert large > small


def test_rmst_difference_detects_longer_survival_group(two_group_survival_df):
    df = two_group_survival_df
    a = df[df["group"] == "A"]
    b = df[df["group"] == "B"]
    result = rmst_difference(a["T"], a["E"], b["T"], b["E"], tau=100, n_bootstrap=300, seed=42)
    assert result.difference > 0
    assert result.ci_lower < result.difference < result.ci_upper
    assert result.p_value < 0.05


def test_rmst_difference_is_near_zero_for_identical_distributions():
    rng = np.random.default_rng(9)
    t_a = rng.exponential(100, 300)
    t_b = rng.exponential(100, 300)
    events = np.ones(300)
    result = rmst_difference(t_a, events, t_b, events, tau=80, n_bootstrap=300, seed=1)
    assert abs(result.difference) < 15

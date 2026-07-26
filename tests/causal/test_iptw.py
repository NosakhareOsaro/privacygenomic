import numpy as np

from equitable_rwe.causal.iptw import compute_iptw_weights, effective_sample_size


def test_weights_are_positive(confounded_survival_df):
    weights = compute_iptw_weights(confounded_survival_df, "treat", ["x1"])
    assert (weights > 0).all()


def test_truncation_caps_extreme_weights(confounded_survival_df):
    untruncated = compute_iptw_weights(confounded_survival_df, "treat", ["x1"], truncate_at=None)
    truncated = compute_iptw_weights(confounded_survival_df, "treat", ["x1"], truncate_at=0.9)
    assert truncated.max() <= untruncated.max()


def test_stabilized_weights_have_mean_near_one(confounded_survival_df):
    weights = compute_iptw_weights(
        confounded_survival_df, "treat", ["x1"], stabilized=True, truncate_at=None
    )
    assert abs(weights.mean() - 1.0) < 0.2


def test_effective_sample_size_equals_n_for_equal_weights():
    assert effective_sample_size(np.ones(50)) == 50.0


def test_effective_sample_size_is_less_than_n_for_unequal_weights():
    weights = np.array([1.0] * 49 + [50.0])
    assert effective_sample_size(weights) < 50

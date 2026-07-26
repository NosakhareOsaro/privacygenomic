from equitable_rwe.causal.dowhy_backdoor import fit_dowhy_backdoor


def test_recovers_protective_direction(confounded_binary_df):
    result = fit_dowhy_backdoor(
        confounded_binary_df, "treat", "outcome", common_causes=["x1"], seed=0
    )
    assert result.risk_difference < 0
    assert result.risk_ratio < 1


def test_runs_all_three_refutations(confounded_binary_df):
    result = fit_dowhy_backdoor(
        confounded_binary_df, "treat", "outcome", common_causes=["x1"], seed=0
    )
    names = {r.name for r in result.refutations}
    assert names == {
        "placebo_treatment_refuter",
        "random_common_cause",
        "data_subset_refuter",
    }


def test_passes_refutation_is_boolean(confounded_binary_df):
    result = fit_dowhy_backdoor(
        confounded_binary_df, "treat", "outcome", common_causes=["x1"], seed=0
    )
    assert isinstance(result.passes_refutation, bool)

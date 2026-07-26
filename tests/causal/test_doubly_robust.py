from equitable_rwe.causal.doubly_robust import fit_doubly_robust_survival


def test_recovers_protective_direction_under_confounding(confounded_survival_df):
    result = fit_doubly_robust_survival(
        confounded_survival_df, "T", "E", "treat", covariates=["x1"]
    )
    # True DGP: treatment is protective (reduces hazard).
    assert result.hazard_ratio < 1
    assert result.p_value < 0.05


def test_effective_sample_size_is_reported(confounded_survival_df):
    result = fit_doubly_robust_survival(
        confounded_survival_df, "T", "E", "treat", covariates=["x1"]
    )
    assert 0 < result.effective_sample_size <= len(confounded_survival_df)


def test_confidence_interval_brackets_hazard_ratio(confounded_survival_df):
    result = fit_doubly_robust_survival(
        confounded_survival_df, "T", "E", "treat", covariates=["x1"]
    )
    assert result.ci_lower < result.hazard_ratio < result.ci_upper

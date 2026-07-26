from equitable_rwe.causal.propensity import estimate_propensity_scores


def test_propensity_scores_are_bounded(confounded_survival_df):
    ps = estimate_propensity_scores(confounded_survival_df, "treat", ["x1"])
    assert (ps > 0).all()
    assert (ps < 1).all()


def test_propensity_scores_increase_with_positive_confounder(confounded_survival_df):
    df = confounded_survival_df
    ps = estimate_propensity_scores(df, "treat", ["x1"])
    high_x1_ps = ps[df["x1"] > df["x1"].median()].mean()
    low_x1_ps = ps[df["x1"] <= df["x1"].median()].mean()
    assert high_x1_ps > low_x1_ps

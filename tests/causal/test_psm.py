from equitable_rwe.causal.psm import match_on_propensity


def test_match_on_propensity_improves_balance(confounded_survival_df):
    result = match_on_propensity(confounded_survival_df, "treat", ["x1"])
    assert result.n_matched_pairs > 0
    before = abs(result.balance.loc["x1", "smd_before"])
    after = abs(result.balance.loc["x1", "smd_after"])
    assert after < before


def test_match_on_propensity_produces_equal_group_sizes(confounded_survival_df):
    result = match_on_propensity(confounded_survival_df, "treat", ["x1"])
    counts = result.matched_df["treat"].value_counts()
    assert counts[0] == counts[1] == result.n_matched_pairs


def test_match_on_propensity_every_pair_has_two_rows(confounded_survival_df):
    result = match_on_propensity(confounded_survival_df, "treat", ["x1"])
    pair_sizes = result.matched_df.groupby("_match_id").size()
    assert (pair_sizes == 2).all()


def test_tighter_caliper_matches_fewer_or_equal_pairs(confounded_survival_df):
    loose = match_on_propensity(confounded_survival_df, "treat", ["x1"], caliper=1.0)
    tight = match_on_propensity(confounded_survival_df, "treat", ["x1"], caliper=0.01)
    assert tight.n_matched_pairs <= loose.n_matched_pairs

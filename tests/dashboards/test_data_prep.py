def test_load_cohort_adds_age_group(equity_cohort):
    assert "age_group" in equity_cohort.columns
    assert equity_cohort["age_group"].notna().all()


def test_age_standardized_rates_cover_every_remoteness_level(equity_cohort, data_prep):
    results = data_prep.age_standardized_hospitalization_rate_by_remoteness(equity_cohort)
    assert set(results) == set(equity_cohort["remoteness_level"].unique())
    for result in results.values():
        assert result.rate >= 0
        assert result.ci_lower <= result.rate <= result.ci_upper


def test_province_choropleth_values_are_nonnegative(equity_cohort, data_prep):
    values = data_prep.province_choropleth_values(equity_cohort)
    assert (values >= 0).all()
    assert set(values.index).issubset(set(equity_cohort["province"].unique()))


def test_stage_distribution_rows_sum_to_one(equity_cohort, data_prep):
    table = data_prep.stage_distribution_by_indigenous_proxy(equity_cohort)
    row_sums = table.sum(axis=1)
    assert ((row_sums - 1.0).abs() < 1e-9).all()


def test_income_quintile_by_remoteness_is_ordered_by_the_dgp(equity_cohort, data_prep):
    # The synthetic generator's income_logit decreases with remoteness
    # score (data/synthetic/generate_equity_cohort.py), so mean income
    # quintile should be roughly decreasing from least to most remote.
    series = data_prep.income_quintile_by_remoteness(equity_cohort)
    assert list(series.index) == [
        level
        for level in ["easily_accessible", "accessible", "less_accessible", "remote", "very_remote"]
        if level in series.index
    ]
    assert series.iloc[0] > series.iloc[-1]

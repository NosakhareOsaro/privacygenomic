import pandas as pd
import pytest


@pytest.fixture(scope="module")
def cohort(equity_cohort_module):
    return equity_cohort_module.generate_equity_cohort(n_patients=3000, seed=7)


def test_generated_cohort_has_expected_columns(cohort):
    expected = {
        "patient_id",
        "age_at_diagnosis",
        "sex",
        "province",
        "remoteness_level",
        "remoteness_score",
        "indigenous_identity_proxy",
        "income_quintile",
        "condition",
        "diagnosis_date",
        "stage_at_diagnosis",
        "hospitalizations_per_year",
        "time_to_event_days",
        "event_observed",
    }
    assert expected.issubset(set(cohort.columns))
    assert len(cohort) == 3000


def test_is_deterministic_given_seed(equity_cohort_module):
    a = equity_cohort_module.generate_equity_cohort(n_patients=500, seed=42)
    b = equity_cohort_module.generate_equity_cohort(n_patients=500, seed=42)
    pd.testing.assert_frame_equal(a, b)


def test_different_seeds_differ(equity_cohort_module):
    a = equity_cohort_module.generate_equity_cohort(n_patients=500, seed=1)
    b = equity_cohort_module.generate_equity_cohort(n_patients=500, seed=2)
    assert not a["time_to_event_days"].equals(b["time_to_event_days"])


def test_remoteness_levels_are_valid(cohort, equity_cohort_module):
    assert set(cohort["remoteness_level"]).issubset(set(equity_cohort_module.REMOTENESS_LEVELS))


def test_indigenous_proxy_prevalence_near_calibration_target(cohort, equity_cohort_module):
    observed = cohort["indigenous_identity_proxy"].mean()
    target = equity_cohort_module.INDIGENOUS_POPULATION_SHARE
    assert abs(observed - target) / target < 0.25


def test_hospitalization_rate_ratio_near_calibration_target(cohort, equity_cohort_module):
    report = equity_cohort_module.calibration_report(cohort)
    ratio_stats = report["remote_to_accessible_hospitalization_rate_ratio"]
    assert ratio_stats["relative_error"] < 0.25


def test_event_observed_is_boolean_and_consistent_with_censoring(cohort, equity_cohort_module):
    assert cohort["event_observed"].dtype == bool
    censored = ~cohort["event_observed"]
    assert (
        cohort.loc[censored, "time_to_event_days"] <= equity_cohort_module.ADMIN_CENSORING_DAYS
    ).all()


def test_survival_time_is_never_negative(cohort):
    assert (cohort["time_to_event_days"] >= 0).all()


def test_indigenous_proxy_group_has_shorter_survival_on_average(cohort):
    # Directionally calibrated: see calibration_notes.md. This test locks
    # in the *sign* of the modeled disparity, not its exact magnitude.
    indigenous_mean = cohort.loc[cohort["indigenous_identity_proxy"], "time_to_event_days"].mean()
    other_mean = cohort.loc[~cohort["indigenous_identity_proxy"], "time_to_event_days"].mean()
    assert indigenous_mean < other_mean


def test_calibration_report_keys(cohort, equity_cohort_module):
    report = equity_cohort_module.calibration_report(cohort)
    assert set(report) == {
        "indigenous_population_share",
        "remote_to_accessible_hospitalization_rate_ratio",
    }
    for stats in report.values():
        assert {"observed", "target", "relative_error"}.issubset(stats)

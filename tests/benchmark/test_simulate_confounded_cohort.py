from simulate_confounded_cohort import (
    TRUE_HR_TREATMENT,
    simulate_confounded_cohort,
    true_risk_ratio_at_horizon,
)


def test_simulate_confounded_cohort_shape_and_columns():
    df = simulate_confounded_cohort(n=500, seed=1)
    assert len(df) == 500
    assert set(df.columns) == {
        "patient_id",
        "x1",
        "x2",
        "treat",
        "time_to_event_days",
        "event_observed",
        "event_by_horizon",
    }


def test_simulate_confounded_cohort_is_deterministic():
    a = simulate_confounded_cohort(n=200, seed=5)
    b = simulate_confounded_cohort(n=200, seed=5)
    assert a.equals(b)


def test_treatment_is_confounded_by_x1():
    df = simulate_confounded_cohort(n=3000, seed=1)
    treated_x1 = df.loc[df["treat"] == 1, "x1"].mean()
    control_x1 = df.loc[df["treat"] == 0, "x1"].mean()
    # PROPENSITY_COEF_X1 < 0: higher x1 -> less likely treated.
    assert treated_x1 < control_x1


def test_true_hr_is_protective():
    assert TRUE_HR_TREATMENT < 1


def test_true_risk_ratio_at_horizon_is_protective_and_reproducible():
    rr_a = true_risk_ratio_at_horizon(n_monte_carlo=20000, seed=1)
    rr_b = true_risk_ratio_at_horizon(n_monte_carlo=20000, seed=1)
    assert rr_a == rr_b
    assert rr_a < 1

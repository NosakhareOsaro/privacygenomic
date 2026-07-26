import numpy as np
import pytest

from equitable_rwe.fairness.audit import (
    CALIBRATION_GAP_THRESHOLD,
    FOUR_FIFTHS_THRESHOLD,
    audit_fairness,
)


@pytest.fixture
def biased_predictions():
    rng = np.random.default_rng(0)
    n = 600
    subgroup = rng.choice(["a", "b"], size=n)
    bias = np.where(subgroup == "b", 0.35, 0.0)
    y_pred_proba = np.clip(rng.uniform(0, 0.55, size=n) + bias, 0, 1)
    y_true = rng.binomial(1, y_pred_proba)
    return y_true, y_pred_proba, subgroup


@pytest.fixture
def unbiased_predictions():
    rng = np.random.default_rng(1)
    n = 800
    subgroup = rng.choice(["a", "b"], size=n)
    y_pred_proba = rng.uniform(0, 1, size=n)
    y_true = rng.binomial(1, y_pred_proba)
    return y_true, y_pred_proba, subgroup


def test_detects_injected_disparity(biased_predictions):
    y_true, y_pred_proba, subgroup = biased_predictions
    report = audit_fairness(y_true, y_pred_proba, subgroup)
    assert not report.passes_all_checks
    assert len(report.flagged_subgroups) > 0


def test_no_disparity_passes_checks(unbiased_predictions):
    y_true, y_pred_proba, subgroup = unbiased_predictions
    report = audit_fairness(y_true, y_pred_proba, subgroup)
    assert report.passes_all_checks
    assert report.flagged_subgroups == []


def test_demographic_parity_table_shape(biased_predictions):
    y_true, y_pred_proba, subgroup = biased_predictions
    report = audit_fairness(y_true, y_pred_proba, subgroup)
    assert set(report.demographic_parity["subgroup"]) == {"a", "b"}
    assert (report.demographic_parity["ratio_to_reference"] <= 1.0 + 1e-9).all()


def test_reference_group_has_ratio_one(biased_predictions):
    y_true, y_pred_proba, subgroup = biased_predictions
    report = audit_fairness(y_true, y_pred_proba, subgroup)
    ref_row = report.demographic_parity.set_index("subgroup").loc[report.reference_group]
    assert ref_row["ratio_to_reference"] == pytest.approx(1.0)


def test_four_fifths_threshold_is_the_standard_value():
    assert FOUR_FIFTHS_THRESHOLD == 0.8


def test_calibration_columns_present(biased_predictions):
    y_true, y_pred_proba, subgroup = biased_predictions
    report = audit_fairness(y_true, y_pred_proba, subgroup)
    assert {"observed_rate", "mean_predicted_probability", "calibration_gap"}.issubset(
        report.calibration.columns
    )


def test_calibration_gap_threshold_is_positive():
    assert CALIBRATION_GAP_THRESHOLD > 0

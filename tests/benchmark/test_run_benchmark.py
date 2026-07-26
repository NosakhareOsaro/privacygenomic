import numpy as np
import pytest
from run_benchmark import run_benchmark


@pytest.fixture(scope="module")
def results():
    return run_benchmark()


def test_all_four_estimators_present(results):
    assert set(results["estimator"]) == {"naive_cox", "psm", "iptw", "dowhy_backdoor"}


def test_cox_scale_estimators_are_positive_hazard_ratios(results):
    hr_rows = results[results["effect_measure"] == "hazard_ratio"]
    assert (hr_rows["estimate"] > 0).all()


def test_adjustment_reduces_bias_relative_to_naive(results):
    # The core benchmark claim: at least one adjustment method should
    # recover the true hazard ratio more closely than the naive,
    # unadjusted comparison, on this confounded DGP.
    naive_bias = abs(results.loc[results["estimator"] == "naive_cox", "log_bias"].iloc[0])
    psm_bias = abs(results.loc[results["estimator"] == "psm", "log_bias"].iloc[0])
    iptw_bias = abs(results.loc[results["estimator"] == "iptw", "log_bias"].iloc[0])
    assert min(psm_bias, iptw_bias) < naive_bias


def test_evalues_are_at_least_one(results):
    assert (results["evalue_point"] >= 1.0).all()


def test_dowhy_row_has_no_confidence_interval(results):
    dowhy_row = results[results["estimator"] == "dowhy_backdoor"].iloc[0]
    assert np.isnan(dowhy_row["ci_lower"]) or dowhy_row["ci_lower"] is None

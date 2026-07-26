import pytest

from equitable_rwe.causal.evalue import evalue


def test_evalue_of_null_is_one():
    result = evalue(1.0)
    assert result.estimate_evalue == pytest.approx(1.0)


def test_evalue_increases_with_effect_size_away_from_null():
    small = evalue(0.9)
    large = evalue(0.3)
    assert large.estimate_evalue > small.estimate_evalue


def test_evalue_is_symmetric_for_reciprocal_ratios():
    below = evalue(0.5)
    above = evalue(2.0)
    assert below.estimate_evalue == pytest.approx(above.estimate_evalue)


def test_evalue_matches_known_worked_example():
    # VanderWeele & Ding (2017): RR ~ 3.9 gives E-value ~ 7.26.
    result = evalue(3.9)
    assert result.estimate_evalue == pytest.approx(7.26, abs=0.01)


def test_ci_evalue_is_one_when_ci_crosses_null():
    result = evalue(0.7, ci_lower=0.6, ci_upper=1.1)
    assert result.ci_evalue == pytest.approx(1.0)


def test_ci_evalue_is_less_than_point_evalue_when_ci_excludes_null():
    result = evalue(0.65, ci_lower=0.55, ci_upper=0.77)
    assert 1.0 < result.ci_evalue < result.estimate_evalue

import numpy as np
import pandas as pd
import pytest

from equitable_rwe.survival.competing_risks import cumulative_incidence, fit_fine_gray
from equitable_rwe.survival.cox import fit_cox_ph


def test_fine_gray_reduces_to_ordinary_cox_without_competing_events():
    # When every non-censored event is the event of interest, the
    # Fine-Gray long-format construction degenerates to one row per
    # subject with weight 1 — mathematically identical to a plain Cox
    # partial likelihood. This is the core correctness check for the
    # IPCW long-format implementation.
    rng = np.random.default_rng(1)
    n = 500
    x1 = rng.normal(size=n)
    duration = rng.exponential(20 * np.exp(-0.5 * x1))
    event = rng.binomial(1, 0.85, n)
    df = pd.DataFrame({"T": duration, "event": event, "x1": x1})

    fg = fit_fine_gray(df, "T", "event", event_of_interest=1, covariates=["x1"])
    cph = fit_cox_ph(df, "T", "event", covariates=["x1"])

    assert fg.summary.loc["x1", "coef"] == pytest.approx(cph.params_["x1"], rel=1e-4)


def test_fit_fine_gray_counts_are_consistent(competing_risks_df):
    result = fit_fine_gray(competing_risks_df, "T", "event", event_of_interest=1, covariates=["x1"])
    assert result.n_subjects == len(competing_risks_df)
    assert result.n_events_of_interest == (competing_risks_df["event"] == 1).sum()
    assert result.n_competing_events == (competing_risks_df["event"] == 2).sum()
    assert "x1" in result.summary.index


def test_fit_fine_gray_max_event_times_gives_similar_estimate(competing_risks_df):
    exact = fit_fine_gray(
        competing_risks_df,
        "T",
        "event",
        event_of_interest=1,
        covariates=["x1"],
        max_event_times=None,
    )
    coarse = fit_fine_gray(
        competing_risks_df,
        "T",
        "event",
        event_of_interest=1,
        covariates=["x1"],
        max_event_times=50,
    )
    assert abs(exact.summary.loc["x1", "coef"] - coarse.summary.loc["x1", "coef"]) < 0.15


def test_cumulative_incidence_is_bounded_and_nondecreasing(competing_risks_df):
    ajf = cumulative_incidence(
        competing_risks_df["T"], competing_risks_df["event"], event_of_interest=1
    )
    cif = ajf.cumulative_density_.iloc[:, 0].to_numpy()
    assert np.all(cif >= -1e-9)
    assert np.all(cif <= 1 + 1e-9)
    assert np.all(np.diff(cif) >= -1e-9)


def test_covariate_with_strong_true_effect_recovers_correct_sign():
    rng = np.random.default_rng(6)
    n = 600
    x1 = rng.normal(size=n)
    # numpy's exponential(scale=...) parameterizes by mean: larger x1
    # here means a *longer* mean time to the event of interest, i.e. a
    # *lower* subdistribution hazard. Competing-event timing is
    # independent of x1.
    t_interest = rng.exponential(10 * np.exp(1.2 * x1))
    t_competing = rng.exponential(15, n)
    interest_first = t_interest < t_competing
    duration = np.where(interest_first, t_interest, t_competing)
    event = np.where(interest_first, 1, 2)
    censor_time = rng.exponential(40, n)
    censored = censor_time < duration
    duration = np.where(censored, censor_time, duration)
    event = np.where(censored, 0, event)

    df = pd.DataFrame({"T": duration, "event": event, "x1": x1})
    result = fit_fine_gray(df, "T", "event", event_of_interest=1, covariates=["x1"])
    assert result.summary.loc["x1", "coef"] < 0
    assert result.summary.loc["x1", "p"] < 0.01

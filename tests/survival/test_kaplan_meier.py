import numpy as np

from equitable_rwe.survival.kaplan_meier import fit_km, plot_km


def test_fit_km_basic_counts(survival_df):
    result = fit_km(survival_df["T"], survival_df["E"], label="cohort")
    assert result.n == len(survival_df)
    assert result.n_events == survival_df["E"].sum()
    assert result.label == "cohort"


def test_median_survival_is_positive_or_infinite(survival_df):
    result = fit_km(survival_df["T"], survival_df["E"])
    assert result.median_survival_days > 0


def test_survival_at_is_monotonically_decreasing(survival_df):
    result = fit_km(survival_df["T"], survival_df["E"])
    times = [0, 10, 20, 40, 80]
    values = result.survival_at(times).to_numpy()
    assert np.all(np.diff(values) <= 1e-9)


def test_survival_at_zero_is_one(survival_df):
    result = fit_km(survival_df["T"], survival_df["E"])
    assert result.survival_at([0]).iloc[0] == 1.0


def test_plot_km_returns_axes_with_legend(two_group_survival_df):
    df = two_group_survival_df
    a = fit_km(df.loc[df["group"] == "A", "T"], df.loc[df["group"] == "A", "E"], label="A")
    b = fit_km(df.loc[df["group"] == "B", "T"], df.loc[df["group"] == "B", "E"], label="B")
    ax = plot_km([a, b])
    assert ax.get_legend() is not None
    assert ax.get_ylim()[1] > 1.0


def test_longer_survival_group_has_higher_median(two_group_survival_df):
    df = two_group_survival_df
    a = fit_km(df.loc[df["group"] == "A", "T"], df.loc[df["group"] == "A", "E"], label="A")
    b = fit_km(df.loc[df["group"] == "B", "T"], df.loc[df["group"] == "B", "E"], label="B")
    assert a.median_survival_days > b.median_survival_days

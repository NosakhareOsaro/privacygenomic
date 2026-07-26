"""Data-preparation logic for the equity dashboard, kept separate from
the Streamlit UI script (``equity_dashboard.py``) so it can be unit
tested directly rather than only exercised by clicking through the app.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from equitable_rwe.equity.age_standardization import StandardizedRateResult, direct_standardize

COHORT_PATH = Path(__file__).parents[1] / "data" / "synthetic" / "equity_cohort.parquet"

AGE_BINS = [0, 50, 65, 75, 200]
AGE_LABELS = ["<50", "50-64", "65-74", "75+"]


def load_cohort(path: Path = COHORT_PATH) -> pd.DataFrame:
    """Load the synthetic equity cohort and add an age-group column."""
    df = pd.read_parquet(path)
    df["age_group"] = pd.cut(df["age_at_diagnosis"], bins=AGE_BINS, labels=AGE_LABELS, right=False)
    return df


def age_standardized_hospitalization_rate_by_remoteness(
    df: pd.DataFrame,
) -> dict[str, StandardizedRateResult]:
    """Age-standardized mean hospitalizations/year per remoteness level.

    Uses the cohort's own overall age distribution as the standard
    population (see :mod:`equitable_rwe.equity.age_standardization`).

    Examples
    --------
    >>> df = load_cohort()
    >>> results = age_standardized_hospitalization_rate_by_remoteness(df)
    >>> set(results) == set(df["remoteness_level"].unique())
    True
    """
    standard_population = df.groupby("age_group", observed=True).size()

    results = {}
    for level, group in df.groupby("remoteness_level", observed=True):
        counts = group.groupby("age_group", observed=True)["hospitalizations_per_year"].sum()
        populations = group.groupby("age_group", observed=True).size()
        counts = counts.reindex(standard_population.index, fill_value=0)
        populations = populations.reindex(standard_population.index, fill_value=1)
        results[str(level)] = direct_standardize(counts, populations, standard_population)
    return results


def province_choropleth_values(
    df: pd.DataFrame, value_col: str = "hospitalizations_per_year"
) -> pd.Series:
    """Mean of ``value_col`` per province, for the choropleth.

    Examples
    --------
    >>> df = load_cohort()
    >>> values = province_choropleth_values(df)
    >>> bool((values >= 0).all())
    True
    """
    return df.groupby("province")[value_col].mean()


def stage_distribution_by_indigenous_proxy(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-tabulation of stage at diagnosis by Indigenous-identity proxy status.

    Examples
    --------
    >>> df = load_cohort()
    >>> table = stage_distribution_by_indigenous_proxy(df)
    >>> bool((abs(table.sum(axis=1) - 1.0) < 1e-6).all())
    True
    """
    counts = pd.crosstab(df["indigenous_identity_proxy"], df["stage_at_diagnosis"])
    return counts.div(counts.sum(axis=1), axis=0)


def income_quintile_by_remoteness(df: pd.DataFrame) -> pd.DataFrame:
    """Mean income quintile per remoteness level, sorted least to most remote.

    Examples
    --------
    >>> df = load_cohort()
    >>> table = income_quintile_by_remoteness(df)
    >>> bool(table.iloc[0] >= table.iloc[-1])
    True
    """
    order = ["easily_accessible", "accessible", "less_accessible", "remote", "very_remote"]
    means = df.groupby("remoteness_level", observed=True)["income_quintile"].mean()
    return means.reindex([level for level in order if level in means.index])

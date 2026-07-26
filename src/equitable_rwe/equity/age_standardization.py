"""Age-standardized rates: direct standardization and Poisson regression.

Two complementary ways to compare a rate (e.g. hospitalization,
readmission) across groups without age structure confounding the
comparison:

1. **Direct standardization** (:func:`direct_standardize`) applies each
   group's age-specific rates to a common standard population's age
   distribution, producing one standardized rate per group.
2. **Poisson rate regression** (:func:`fit_poisson_rate_model`) instead
   models the rate directly, with age (and any other covariates) as
   regression terms and a log-population offset — a model-based
   alternative that also yields adjusted rate ratios between groups with
   confidence intervals from the regression itself.

This module does not ship an external reference standard population
(e.g. the WHO World Standard Population) because this project has no
verified source for its exact published age-group weights to cite
correctly. Callers must supply their own ``standard_population`` — the
dashboard in this project uses the cohort's own overall age distribution
as the standard (valid for comparing subgroups *within* this study; not
appropriate for comparing this project's rates to externally published
age-standardized rates, which would require the same external standard).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


@dataclass
class StandardizedRateResult:
    """A directly age-standardized rate with a Poisson-based confidence interval.

    Attributes
    ----------
    rate : float
        The directly standardized rate (same units as the input
        age-specific rates, e.g. events per person).
    variance : float
        Variance of the standardized rate estimate, assuming Poisson
        counts within each age group.
    ci_lower, ci_upper : float
        95% confidence interval (normal approximation).
    """

    rate: float
    variance: float
    ci_lower: float
    ci_upper: float


def direct_standardize(
    counts: pd.Series, populations: pd.Series, standard_population: pd.Series
) -> StandardizedRateResult:
    """Directly age-standardize a rate against a standard population.

    Parameters
    ----------
    counts : pandas.Series
        Event counts per age group, indexed by age group.
    populations : pandas.Series
        Population (person-time or person-count) per age group, same
        index as ``counts``.
    standard_population : pandas.Series
        Standard population weight per age group, same index. Only the
        relative weights matter (they are normalized internally).

    Returns
    -------
    StandardizedRateResult

    Examples
    --------
    >>> import pandas as pd
    >>> counts = pd.Series({"young": 5, "old": 40}, name="counts")
    >>> pops = pd.Series({"young": 1000, "old": 500})
    >>> standard = pd.Series({"young": 600, "old": 400})
    >>> result = direct_standardize(counts, pops, standard)
    >>> round(result.rate, 4)
    0.035
    """
    age_groups = counts.index
    weights = standard_population.loc[age_groups]
    normalized_weights = weights / weights.sum()

    age_specific_rates = counts / populations
    standardized_rate = float((normalized_weights * age_specific_rates).sum())

    variance = float(((normalized_weights**2) * (counts / (populations**2))).sum())
    se = np.sqrt(variance)

    return StandardizedRateResult(
        rate=standardized_rate,
        variance=variance,
        ci_lower=max(0.0, standardized_rate - 1.96 * se),
        ci_upper=standardized_rate + 1.96 * se,
    )


def fit_poisson_rate_model(
    df: pd.DataFrame, count_col: str, exposure_col: str, predictors: list[str]
):
    """Fit a Poisson rate regression with a log-exposure offset.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per group/stratum (e.g. age group x remoteness level).
    count_col : str
        Event count column.
    exposure_col : str
        Population or person-time column, used as a log offset so
        coefficients are interpreted as rate ratios.
    predictors : list[str]
        Covariates (e.g. age group, remoteness level) to include.

    Returns
    -------
    statsmodels.genmod.generalized_linear_model.GLMResultsWrapper
        Fitted Poisson GLM. Use ``np.exp(result.params)`` for rate
        ratios and ``result.conf_int()`` for their log-scale CIs.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame(
    ...     {
    ...         "count": [5, 40, 8, 60],
    ...         "population": [1000, 500, 900, 480],
    ...         "group": ["a", "a", "b", "b"],
    ...     }
    ... )
    >>> result = fit_poisson_rate_model(df, "count", "population", ["group"])
    >>> "group[T.b]" in result.params.index
    True
    """
    formula = f"{count_col} ~ " + " + ".join(predictors)
    model = smf.glm(
        formula=formula,
        data=df,
        family=sm.families.Poisson(),
        offset=np.log(df[exposure_col]),
    )
    return model.fit()

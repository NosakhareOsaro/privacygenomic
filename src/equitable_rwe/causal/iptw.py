"""Inverse probability of treatment weighting (IPTW)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from equitable_rwe.causal.propensity import estimate_propensity_scores


def compute_iptw_weights(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    stabilized: bool = True,
    truncate_at: float | None = 0.99,
) -> np.ndarray:
    """Compute (optionally stabilized) inverse probability of treatment weights.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject.
    treatment_col : str
        Binary (0/1) treatment indicator column.
    covariates : list[str]
        Covariates for the propensity model.
    stabilized : bool, default True
        Use stabilized weights (multiply by the marginal treatment
        prevalence), which have lower variance than unstabilized weights
        while targeting the same estimand.
    truncate_at : float, optional, default 0.99
        Truncate weights at this upper percentile to limit the influence
        of extreme weights from near-zero or near-one propensity scores.
        Set to ``None`` to disable truncation.

    Returns
    -------
    numpy.ndarray
        Weight per subject.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> treat = rng.binomial(1, 1 / (1 + np.exp(-x1)))
    >>> df = pd.DataFrame({"treat": treat, "x1": x1})
    >>> weights = compute_iptw_weights(df, "treat", ["x1"])
    >>> bool((weights > 0).all())
    True
    """
    propensity = estimate_propensity_scores(df, treatment_col, covariates)
    treatment = df[treatment_col].to_numpy()

    weights = np.where(treatment == 1, 1 / propensity, 1 / (1 - propensity))
    if stabilized:
        marginal_prevalence = treatment.mean()
        stabilizer = np.where(treatment == 1, marginal_prevalence, 1 - marginal_prevalence)
        weights = weights * stabilizer

    if truncate_at is not None:
        cap = np.quantile(weights, truncate_at)
        weights = np.minimum(weights, cap)

    return weights


def effective_sample_size(weights: np.ndarray) -> float:
    """Kish's effective sample size for a set of weights.

    A large gap between this and ``len(weights)`` signals that a small
    number of extreme weights dominate the analysis — a standard IPTW
    diagnostic.

    Examples
    --------
    >>> import numpy as np
    >>> effective_sample_size(np.ones(100))
    100.0
    """
    return float(weights.sum() ** 2 / (weights**2).sum())

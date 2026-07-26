"""Propensity score estimation shared by the PSM and IPTW modules."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


def estimate_propensity_scores(
    df: pd.DataFrame, treatment_col: str, covariates: list[str]
) -> np.ndarray:
    """Estimate P(treatment=1 | covariates) via logistic regression.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject.
    treatment_col : str
        Binary (0/1) treatment indicator column.
    covariates : list[str]
        Covariates to include in the propensity model.

    Returns
    -------
    numpy.ndarray
        Estimated propensity score per subject, in ``(0, 1)``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x1 = rng.normal(size=n)
    >>> treat = rng.binomial(1, 1 / (1 + np.exp(-x1)))
    >>> df = pd.DataFrame({"treat": treat, "x1": x1})
    >>> ps = estimate_propensity_scores(df, "treat", ["x1"])
    >>> bool(((ps > 0) & (ps < 1)).all())
    True
    """
    model = LogisticRegression(max_iter=1000)
    model.fit(df[covariates], df[treatment_col])
    propensity = model.predict_proba(df[covariates])[:, 1]
    return np.clip(propensity, 1e-6, 1 - 1e-6)

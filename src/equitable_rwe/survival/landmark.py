"""Landmark analysis for time-varying covariate effects.

Landmarking (van Houwelingen, 2007) sidesteps the proportional hazards
assumption for effects that plausibly change over follow-up by
re-anchoring the analysis at a chosen "landmark" time: restrict to
subjects still event-free and under observation at that time, reset the
time origin there, and fit a fresh model on the remainder of follow-up.
Repeating this at a grid of landmark times shows how an effect estimate
evolves — see :func:`landmark_table`.
"""

from __future__ import annotations

import pandas as pd
from lifelines import CoxPHFitter


def landmark_dataset(
    df: pd.DataFrame, duration_col: str, event_col: str, landmark_time: float
) -> pd.DataFrame:
    """Restrict to subjects at risk at ``landmark_time`` and reset the time origin.

    Subjects whose observed time (event or censoring) is at or before
    ``landmark_time`` are dropped — they were not both alive and under
    observation at the landmark. Remaining subjects' duration is measured
    from the landmark rather than from baseline.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject.
    duration_col, event_col : str
        Follow-up time and event indicator columns.
    landmark_time : float
        The landmark time, in the same units as ``duration_col``.

    Returns
    -------
    pandas.DataFrame
        Filtered copy of ``df`` with ``duration_col`` replaced by time
        since the landmark.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({"T": [10, 50, 100], "E": [1, 1, 0]})
    >>> out = landmark_dataset(df, "T", "E", landmark_time=30)
    >>> len(out)
    2
    >>> out["T"].tolist()
    [20, 70]
    """
    at_risk = df[df[duration_col] > landmark_time].copy()
    at_risk[duration_col] = at_risk[duration_col] - landmark_time
    return at_risk


def fit_landmark_cox(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    landmark_time: float,
    covariates: list[str] | None = None,
) -> CoxPHFitter:
    """Fit a Cox model on the landmark cohort at a single landmark time.

    Parameters
    ----------
    df : pandas.DataFrame
        Full cohort, in baseline-time units.
    duration_col, event_col : str
        Follow-up time and event indicator columns.
    landmark_time : float
        The landmark time.
    covariates : list[str], optional
        Columns to include as covariates. Defaults to every column other
        than ``duration_col`` and ``event_col``.

    Returns
    -------
    lifelines.CoxPHFitter
        Fitted on the landmark-restricted, time-reset cohort.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> df = pd.DataFrame(
    ...     {
    ...         "T": rng.exponential(100, n),
    ...         "E": rng.binomial(1, 0.7, n),
    ...         "x1": rng.normal(size=n),
    ...     }
    ... )
    >>> cph = fit_landmark_cox(df, "T", "E", landmark_time=20, covariates=["x1"])
    >>> "x1" in cph.params_.index
    True
    """
    landmark_df = landmark_dataset(df, duration_col, event_col, landmark_time)
    keep = covariates or [c for c in df.columns if c not in {duration_col, event_col}]
    cph = CoxPHFitter()
    cph.fit(
        landmark_df[[duration_col, event_col, *keep]],
        duration_col=duration_col,
        event_col=event_col,
    )
    return cph


def landmark_table(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    landmark_times: list[float],
    covariate: str,
) -> pd.DataFrame:
    """Estimate a single covariate's hazard ratio across a grid of landmark times.

    Parameters
    ----------
    df : pandas.DataFrame
        Full cohort, in baseline-time units.
    duration_col, event_col : str
        Follow-up time and event indicator columns.
    landmark_times : list[float]
        Landmark times to fit at.
    covariate : str
        The single covariate whose evolving effect is being examined; it
        is the only covariate included in each landmark model.

    Returns
    -------
    pandas.DataFrame
        One row per landmark time: ``landmark_time``, ``n_at_risk``,
        ``hazard_ratio``, ``ci_lower``, ``ci_upper``, ``p_value``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> df = pd.DataFrame(
    ...     {
    ...         "T": rng.exponential(100, n),
    ...         "E": rng.binomial(1, 0.7, n),
    ...         "x1": rng.normal(size=n),
    ...     }
    ... )
    >>> table = landmark_table(df, "T", "E", landmark_times=[0, 20, 40], covariate="x1")
    >>> list(table["landmark_time"])
    [0, 20, 40]
    """
    rows = []
    for t in landmark_times:
        landmark_df = landmark_dataset(df, duration_col, event_col, t)
        cph = CoxPHFitter()
        cph.fit(
            landmark_df[[duration_col, event_col, covariate]],
            duration_col=duration_col,
            event_col=event_col,
        )
        summary = cph.summary.loc[covariate]
        rows.append(
            {
                "landmark_time": t,
                "n_at_risk": len(landmark_df),
                "hazard_ratio": float(summary["exp(coef)"]),
                "ci_lower": float(summary["exp(coef) lower 95%"]),
                "ci_upper": float(summary["exp(coef) upper 95%"]),
                "p_value": float(summary["p"]),
            }
        )
    return pd.DataFrame(rows)

"""Competing-risks survival analysis: Fine-Gray subdistribution hazard
regression and the Aalen-Johansen cumulative incidence estimator.

Most Python survival libraries stop at cause-specific Cox models, which
censor competing events and so answer a different question than the one
usually intended ("what is this covariate's effect on the risk of *this*
event, accounting for the fact that *other* events remove people from
risk?"). Fine & Gray's (1999) subdistribution hazard model answers that
question directly. lifelines does not ship it, so this module implements
the standard "long format" construction (as used by R's
``survival::finegray``): competing-event subjects are kept in the risk
set past their event time with an inverse-probability-of-censoring
(IPCW) weight that decays according to the censoring distribution, then
a weighted Cox model is fit on the resulting counting-process data via
:class:`lifelines.CoxTimeVaryingFitter`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from lifelines import AalenJohansenFitter, CoxTimeVaryingFitter, KaplanMeierFitter


@dataclass
class FineGrayResult:
    """A fitted Fine-Gray subdistribution hazard model.

    Standard errors in ``summary`` are model-based (naive), not the
    robust/sandwich variance a reference Fine-Gray implementation would
    report, because lifelines' ``CoxTimeVaryingFitter`` does not yet
    support ``robust=True`` with weights. Point estimates (``coef``,
    ``exp(coef)``) are unaffected. See
    ``docs/architecture/adr-0002-survival-model-choices.md``.

    Attributes
    ----------
    fitter : lifelines.CoxTimeVaryingFitter
        The underlying fitted weighted Cox model, for direct access to
        anything not wrapped here (e.g. ``.print_summary()``).
    summary : pandas.DataFrame
        Coefficient table: ``coef``, ``exp(coef)``, standard errors, and
        p-values, one row per covariate.
    n_subjects : int
        Number of subjects in the original (short-format) cohort.
    n_events_of_interest : int
        Number of subjects who experienced the event of interest.
    n_competing_events : int
        Number of subjects who experienced a competing event.
    """

    fitter: CoxTimeVaryingFitter
    summary: pd.DataFrame
    n_subjects: int
    n_events_of_interest: int
    n_competing_events: int


def _censoring_survival_function(
    durations: np.ndarray, is_censored: np.ndarray
) -> KaplanMeierFitter:
    """KM estimate of the censoring distribution G(t) = P(censoring time > t)."""
    kmf = KaplanMeierFitter()
    kmf.fit(durations, event_observed=is_censored)
    return kmf


def _build_fine_gray_long_format(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    event_of_interest,
    id_col: str,
    max_event_times: int | None = None,
) -> pd.DataFrame:
    """Construct the IPCW long-format counting-process dataset for Fine-Gray.

    Subjects with the event of interest or who are censored contribute a
    single ``(0, T]`` row with weight 1. Subjects with a competing event
    at time ``T_i`` contribute a row up to ``T_i`` with weight 1, plus one
    additional row per later event-of-interest time ``t_j > T_i`` (up to
    the last observed time), each with weight ``G(t_j) / G(T_i)`` — the
    standard Fine & Gray (1999) risk-set-extension weighting.

    This construction is exact but has ``O(n_competing * n_event_times)``
    output rows: for a few thousand subjects it can reach the low
    millions, and fitting slows accordingly. ``max_event_times`` trades a
    small amount of precision for tractability by coarsening the
    event-of-interest time grid to an evenly spaced quantile subset — a
    standard practical mitigation for this method (the same idea R's
    ``finegray()`` supports via its own coarser time grids), rather than
    evaluating every unique event time.
    """
    is_censored = df[event_col] == 0
    censoring_km = _censoring_survival_function(df[duration_col].to_numpy(), is_censored.to_numpy())

    event_of_interest_times = np.sort(
        df.loc[df[event_col] == event_of_interest, duration_col].unique()
    )
    if max_event_times is not None and len(event_of_interest_times) > max_event_times:
        quantile_positions = np.linspace(0, len(event_of_interest_times) - 1, max_event_times)
        event_of_interest_times = event_of_interest_times[np.round(quantile_positions).astype(int)]
        event_of_interest_times = np.unique(event_of_interest_times)

    is_event_of_interest = df[event_col] == event_of_interest
    is_competing = ~is_event_of_interest & ~is_censored

    # Subjects with the event of interest, or censored: a single row each.
    base = pd.DataFrame(
        {
            id_col: df.loc[~is_competing, id_col].to_numpy(),
            "start": 0.0,
            "stop": df.loc[~is_competing, duration_col].to_numpy(),
            "event": is_event_of_interest.loc[~is_competing].astype(int).to_numpy(),
            "weight": 1.0,
        }
    )

    # Competing-event subjects: one "up to T_i" row (weight 1), plus one
    # extension row per later event-of-interest time, vectorized as a
    # boolean (m competing subjects x k event-of-interest times) mask
    # rather than a per-subject, per-time Python loop.
    competing_ids = df.loc[is_competing, id_col].to_numpy()
    competing_times = df.loc[is_competing, duration_col].to_numpy()
    up_to_competing_time = pd.DataFrame(
        {
            id_col: competing_ids,
            "start": 0.0,
            "stop": competing_times,
            "event": 0,
            "weight": 1.0,
        }
    )

    extension_rows = pd.DataFrame(columns=[id_col, "start", "stop", "event", "weight"])
    if len(competing_ids) and len(event_of_interest_times):
        mask = event_of_interest_times[np.newaxis, :] > competing_times[:, np.newaxis]
        row_idx, col_idx = np.nonzero(mask)
        if len(row_idx):
            flat_ids = competing_ids[row_idx]
            flat_stops = event_of_interest_times[col_idx]
            flat_competing_times = competing_times[row_idx]

            extension = pd.DataFrame(
                {id_col: flat_ids, "stop": flat_stops, "_competing_time": flat_competing_times}
            ).sort_values([id_col, "stop"])
            extension["start"] = extension.groupby(id_col)["stop"].shift(1)
            extension["start"] = extension["start"].fillna(extension["_competing_time"])

            g_stop = censoring_km.survival_function_at_times(
                extension["stop"].to_numpy()
            ).to_numpy()
            g_competing = censoring_km.survival_function_at_times(
                extension["_competing_time"].to_numpy()
            ).to_numpy()
            extension["weight"] = np.where(g_competing > 0, g_stop / g_competing, 0.0)
            extension["event"] = 0
            extension_rows = extension[[id_col, "start", "stop", "event", "weight"]]

    parts = [part for part in (base, up_to_competing_time, extension_rows) if not part.empty]
    return pd.concat(parts, ignore_index=True)


def fit_fine_gray(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    event_of_interest,
    covariates: list[str],
    id_col: str = "_subject_id",
    max_event_times: int | None = 200,
) -> FineGrayResult:
    """Fit a Fine-Gray subdistribution hazard model.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject. ``id_col`` is created automatically if not
        already present.
    duration_col : str
        Follow-up time column.
    event_col : str
        Event-type column: ``0`` for censored, ``event_of_interest`` for
        the event of interest, any other value for a competing event.
    event_of_interest : Any
        The value in ``event_col`` denoting the event being modeled.
    covariates : list[str]
        Covariate columns to include.
    id_col : str, default "_subject_id"
        Subject identifier column name (created from the row index if
        absent).
    max_event_times : int, optional, default 200
        Coarsen the event-of-interest time grid to at most this many
        points before building the long-format dataset (see
        :func:`_build_fine_gray_long_format`). Set to ``None`` for the
        exact construction over every unique event time, at the cost of
        the row count (and fit time) growing with cohort size — see
        ``docs/architecture/adr-0002-survival-model-choices.md``.

    Returns
    -------
    FineGrayResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> df = pd.DataFrame(
    ...     {
    ...         "T": rng.exponential(20, n),
    ...         "event": rng.choice([0, 1, 2], size=n, p=[0.2, 0.5, 0.3]),
    ...         "x1": x1,
    ...     }
    ... )
    >>> result = fit_fine_gray(df, "T", "event", event_of_interest=1, covariates=["x1"])
    >>> "x1" in result.summary.index
    True
    """
    df = df.copy()
    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))

    long_df = _build_fine_gray_long_format(
        df, duration_col, event_col, event_of_interest, id_col, max_event_times=max_event_times
    )
    covariate_lookup = df.set_index(id_col)[covariates]
    long_df = long_df.merge(covariate_lookup, left_on=id_col, right_index=True, how="left")
    long_df = long_df[long_df["stop"] > long_df["start"]].reset_index(drop=True)

    # NOTE: lifelines' CoxTimeVaryingFitter does not yet implement robust
    # (sandwich) variance (raises NotImplementedError if robust=True), so
    # standard errors below are the naive model-based Cox estimate. This
    # understates uncertainty relative to a reference Fine-Gray
    # implementation (e.g. R's crr, or coxph(..., robust=TRUE,
    # cluster=id) on finegray() output) — point estimates are unaffected.
    # See docs/architecture/adr-0002-survival-model-choices.md.
    ctv = CoxTimeVaryingFitter()
    ctv.fit(
        long_df,
        event_col="event",
        start_col="start",
        stop_col="stop",
        weights_col="weight",
        id_col=id_col,
        robust=False,
    )

    n_events_of_interest = int((df[event_col] == event_of_interest).sum())
    n_competing = int(((df[event_col] != event_of_interest) & (df[event_col] != 0)).sum())

    return FineGrayResult(
        fitter=ctv,
        summary=ctv.summary,
        n_subjects=len(df),
        n_events_of_interest=n_events_of_interest,
        n_competing_events=n_competing,
    )


def cumulative_incidence(
    durations, event_observed, event_of_interest, timeline=None
) -> AalenJohansenFitter:
    """Nonparametric cumulative incidence function via Aalen-Johansen.

    A model-free complement to :func:`fit_fine_gray`, useful as a sanity
    check: the Fine-Gray model's baseline cumulative incidence should
    track this nonparametric estimate reasonably closely.

    Parameters
    ----------
    durations : array-like
        Follow-up times.
    event_observed : array-like
        Event-type indicator (0 = censored, otherwise the event type).
    event_of_interest : Any
        The event type to compute cumulative incidence for.
    timeline : array-like, optional
        Times at which to evaluate the estimate.

    Returns
    -------
    lifelines.AalenJohansenFitter
        Fitted estimator; see ``.cumulative_density_`` for the CIF.

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> durations = rng.exponential(20, n)
    >>> events = rng.choice([0, 1, 2], size=n, p=[0.2, 0.5, 0.3])
    >>> ajf = cumulative_incidence(durations, events, event_of_interest=1)
    >>> float(ajf.cumulative_density_.iloc[-1, 0]) <= 1.0
    True
    """
    ajf = AalenJohansenFitter()
    ajf.fit(durations, event_observed, event_of_interest=event_of_interest, timeline=timeline)
    return ajf

"""Propensity score matching (a scikit-learn-based MatchIt equivalent).

Nearest-neighbor matching on the logit of the propensity score, with an
optional caliper, plus the standardized-mean-difference balance
diagnostics that are MatchIt's signature output — implemented directly on
top of ``sklearn.neighbors.NearestNeighbors`` rather than via an R
interop layer, so this project has no R dependency anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from equitable_rwe.causal.propensity import estimate_propensity_scores


@dataclass
class MatchResult:
    """Result of :func:`match_on_propensity`.

    Attributes
    ----------
    matched_df : pandas.DataFrame
        The matched cohort (treated subjects plus their matched controls),
        with a ``_match_id`` column linking each matched pair.
    balance : pandas.DataFrame
        Standardized mean difference (SMD) per covariate, before and
        after matching. Values under ~0.1 are conventionally considered
        well balanced.
    n_matched_pairs : int
        Number of treated subjects successfully matched.
    n_treated_unmatched : int
        Number of treated subjects dropped (no control within the
        caliper).
    """

    matched_df: pd.DataFrame
    balance: pd.DataFrame
    n_matched_pairs: int
    n_treated_unmatched: int


def _standardized_mean_difference(treated: pd.Series, control: pd.Series) -> float:
    pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((treated.mean() - control.mean()) / pooled_sd)


def _balance_table(
    df: pd.DataFrame, treatment_col: str, covariates: list[str], matched_df: pd.DataFrame
) -> pd.DataFrame:
    rows = []
    for cov in covariates:
        before = _standardized_mean_difference(
            df.loc[df[treatment_col] == 1, cov], df.loc[df[treatment_col] == 0, cov]
        )
        after = _standardized_mean_difference(
            matched_df.loc[matched_df[treatment_col] == 1, cov],
            matched_df.loc[matched_df[treatment_col] == 0, cov],
        )
        rows.append({"covariate": cov, "smd_before": before, "smd_after": after})
    return pd.DataFrame(rows).set_index("covariate")


def match_on_propensity(
    df: pd.DataFrame,
    treatment_col: str,
    covariates: list[str],
    caliper: float = 0.2,
) -> MatchResult:
    """1:1 nearest-neighbor matching on the logit propensity score.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject, with a unique default index (used to build
        ``_match_id`` pairs).
    treatment_col : str
        Binary (0/1) treatment indicator column.
    covariates : list[str]
        Covariates for the propensity model and for balance diagnostics.
    caliper : float, default 0.2
        Maximum allowed distance on the logit-propensity scale, expressed
        as a multiple of the pooled standard deviation of the logit
        propensity score (Austin, 2011's recommended default of 0.2 SDs).
        Treated subjects with no control within the caliper are dropped.

    Returns
    -------
    MatchResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> treat = rng.binomial(1, 1 / (1 + np.exp(-x1)))
    >>> df = pd.DataFrame({"treat": treat, "x1": x1})
    >>> result = match_on_propensity(df, "treat", ["x1"])
    >>> result.n_matched_pairs > 0
    True
    """
    df = df.reset_index(drop=True)
    propensity = estimate_propensity_scores(df, treatment_col, covariates)
    logit_ps = np.log(propensity / (1 - propensity))

    treated_idx = df.index[df[treatment_col] == 1].to_numpy()
    control_idx = df.index[df[treatment_col] == 0].to_numpy()
    caliper_width = caliper * logit_ps.std()

    nn = NearestNeighbors(n_neighbors=1).fit(logit_ps[control_idx].reshape(-1, 1))
    distances, neighbor_pos = nn.kneighbors(logit_ps[treated_idx].reshape(-1, 1))

    matched_rows = []
    match_id = 0
    n_unmatched = 0
    used_controls: set[int] = set()
    for t_pos, (dist, ctrl_pos) in enumerate(zip(distances[:, 0], neighbor_pos[:, 0], strict=True)):
        control_row_idx = control_idx[ctrl_pos]
        if dist > caliper_width or control_row_idx in used_controls:
            n_unmatched += 1
            continue
        used_controls.add(control_row_idx)
        treated_row_idx = treated_idx[t_pos]
        matched_rows.append((match_id, treated_row_idx))
        matched_rows.append((match_id, control_row_idx))
        match_id += 1

    match_ids, row_positions = zip(*matched_rows, strict=True) if matched_rows else ((), ())
    matched_df = df.loc[list(row_positions)].copy()
    matched_df["_match_id"] = list(match_ids)

    balance = _balance_table(df, treatment_col, covariates, matched_df)

    return MatchResult(
        matched_df=matched_df.reset_index(drop=True),
        balance=balance,
        n_matched_pairs=match_id,
        n_treated_unmatched=n_unmatched,
    )

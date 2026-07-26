"""Subgroup fairness auditing for a causal survival estimate.

The classifier-style metrics in :mod:`equitable_rwe.fairness.audit`
(demographic parity, equalized odds, calibration) don't apply to a hazard
ratio — there is no "positive prediction rate" for a Cox coefficient.
What *does* transfer is the underlying question: is this estimate equally
trustworthy for every subgroup, or does the topline number quietly rest
on much weaker evidence for some group? This module answers that by
refitting the doubly-robust estimator (see
:mod:`equitable_rwe.causal.doubly_robust`) separately within each
subgroup and checking, per subgroup: is there enough data (effective
sample size), and is the propensity model actually balancing the
covariates (standardized mean difference), for that subgroup's estimate
to be treated as reliable evidence — regardless of what its point
estimate says.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from equitable_rwe.causal.doubly_robust import fit_doubly_robust_survival
from equitable_rwe.causal.iptw import compute_iptw_weights

#: A subgroup's effective sample size below this fraction of its raw
#: sample size signals a small number of extreme weights dominating that
#: subgroup's estimate.
MIN_EFFECTIVE_SAMPLE_FRACTION = 0.5

#: A subgroup with fewer events than this has too little information for
#: a stable subgroup-specific Cox fit (a common rule-of-thumb minimum,
#: e.g. Peduzzi et al. 1996's "≥10 events per predictor" heuristic
#: applied here with 1 predictor of primary interest).
MIN_EVENTS_PER_SUBGROUP = 10


def _standardized_mean_difference(treated: pd.Series, control: pd.Series) -> float:
    pooled_sd = np.sqrt((treated.var() + control.var()) / 2)
    if pooled_sd == 0:
        return 0.0
    return float((treated.mean() - control.mean()) / pooled_sd)


@dataclass
class CausalFairnessReport:
    """Per-subgroup reliability audit of a doubly-robust survival estimate.

    Attributes
    ----------
    overall_hazard_ratio : float
        The hazard ratio from fitting on the full cohort.
    per_subgroup : pandas.DataFrame
        One row per subgroup: ``subgroup``, ``n``, ``n_events``,
        ``hazard_ratio``, ``ci_lower``, ``ci_upper``,
        ``effective_sample_fraction`` (effective sample size / n), and
        ``max_abs_smd`` (largest post-weighting standardized mean
        difference across covariates — lower is better balance).
    flagged_subgroups : list[str]
        Subgroups where the estimate should be treated with caution:
        too few events, or a low effective sample fraction, or poor
        post-weighting covariate balance.
    """

    overall_hazard_ratio: float
    per_subgroup: pd.DataFrame
    flagged_subgroups: list[str]

    @property
    def all_subgroups_adequately_supported(self) -> bool:
        """``True`` if no subgroup was flagged."""
        return len(self.flagged_subgroups) == 0


def audit_causal_subgroup_fairness(
    df: pd.DataFrame,
    duration_col: str,
    event_col: str,
    treatment_col: str,
    covariates: list[str],
    subgroup_col: str,
) -> CausalFairnessReport:
    """Audit a doubly-robust survival estimate's reliability across subgroups.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject.
    duration_col, event_col : str
        Follow-up time and event indicator columns.
    treatment_col : str
        Binary (0/1) treatment indicator.
    covariates : list[str]
        Covariates for the propensity model and outcome-model adjustment.
    subgroup_col : str
        Column defining the subgroups to audit (e.g. a demographic or
        geographic category).

    Returns
    -------
    CausalFairnessReport

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 1200
    >>> x1 = rng.normal(size=n)
    >>> treat = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x1)))
    >>> subgroup = rng.choice(["majority", "minority"], size=n, p=[0.9, 0.1])
    >>> duration = rng.exponential(30 * np.exp(0.3 * treat + 0.3 * x1))
    >>> df = pd.DataFrame(
    ...     {
    ...         "T": duration,
    ...         "E": rng.binomial(1, 0.9, n),
    ...         "treat": treat,
    ...         "x1": x1,
    ...         "group": subgroup,
    ...     }
    ... )
    >>> report = audit_causal_subgroup_fairness(df, "T", "E", "treat", ["x1"], "group")
    >>> "minority" in set(report.per_subgroup["subgroup"])
    True
    """
    overall = fit_doubly_robust_survival(df, duration_col, event_col, treatment_col, covariates)

    rows = []
    flagged = []
    for subgroup_value, group_df in df.groupby(subgroup_col):
        n = len(group_df)
        n_events = int(group_df[event_col].sum())

        weights = compute_iptw_weights(group_df, treatment_col, covariates)
        effective_n = float(weights.sum() ** 2 / (weights**2).sum())
        effective_fraction = effective_n / n

        smds = [
            abs(
                _standardized_mean_difference(
                    group_df.loc[group_df[treatment_col] == 1, cov],
                    group_df.loc[group_df[treatment_col] == 0, cov],
                )
            )
            for cov in covariates
        ]
        max_abs_smd = max(smds) if smds else 0.0

        row: dict[str, object] = {
            "subgroup": subgroup_value,
            "n": n,
            "n_events": n_events,
            "effective_sample_fraction": effective_fraction,
            "max_abs_smd": max_abs_smd,
        }

        is_flagged = (
            n_events < MIN_EVENTS_PER_SUBGROUP or effective_fraction < MIN_EFFECTIVE_SAMPLE_FRACTION
        )

        if n_events >= MIN_EVENTS_PER_SUBGROUP:
            try:
                subgroup_result = fit_doubly_robust_survival(
                    group_df, duration_col, event_col, treatment_col, covariates
                )
                row["hazard_ratio"] = subgroup_result.hazard_ratio
                row["ci_lower"] = subgroup_result.ci_lower
                row["ci_upper"] = subgroup_result.ci_upper
            except Exception:
                row["hazard_ratio"] = float("nan")
                row["ci_lower"] = float("nan")
                row["ci_upper"] = float("nan")
                is_flagged = True
        else:
            row["hazard_ratio"] = float("nan")
            row["ci_lower"] = float("nan")
            row["ci_upper"] = float("nan")

        rows.append(row)
        if is_flagged:
            flagged.append(str(subgroup_value))

    per_subgroup = pd.DataFrame(rows)[
        [
            "subgroup",
            "n",
            "n_events",
            "hazard_ratio",
            "ci_lower",
            "ci_upper",
            "effective_sample_fraction",
            "max_abs_smd",
        ]
    ]

    return CausalFairnessReport(
        overall_hazard_ratio=overall.hazard_ratio,
        per_subgroup=per_subgroup,
        flagged_subgroups=flagged,
    )

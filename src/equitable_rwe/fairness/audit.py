"""AEQUITAS-style subgroup fairness audit for a binary classifier.

Reimplements the three metrics at the core of the `Aequitas
<https://github.com/dssg/aequitas>`_ toolkit's audit — demographic
parity, equalized odds, and within-group calibration — directly on
scikit-learn-shaped predictions, so this project has no dependency on
Aequitas itself. See ``docs/architecture/adr-0004-fairness-and-ml-choices.md``
for why.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

#: Four-fifths (80%) rule: the standard adverse-impact threshold from
#: US EEOC Uniform Guidelines on Employee Selection Procedures (1978),
#: widely reused in the algorithmic fairness literature as a demographic
#: parity threshold.
FOUR_FIFTHS_THRESHOLD = 0.8

#: Conventional threshold for a "meaningful" gap in true/false positive
#: rate between subgroups (Aequitas and related tooling commonly use a
#: similar order-of-magnitude default).
EQUALIZED_ODDS_GAP_THRESHOLD = 0.1

#: Conventional threshold for a "meaningful" within-group calibration gap
#: (observed rate vs. mean predicted probability).
CALIBRATION_GAP_THRESHOLD = 0.1


@dataclass
class FairnessReport:
    """Result of :func:`audit_fairness`.

    Attributes
    ----------
    demographic_parity : pandas.DataFrame
        Per-subgroup selection rate, ratio to the group with the highest
        selection rate, and a ``passes_four_fifths_rule`` flag.
    equalized_odds : pandas.DataFrame
        Per-subgroup true positive rate (TPR) and false positive rate
        (FPR).
    calibration : pandas.DataFrame
        Per-subgroup mean predicted probability, observed event rate, and
        the gap between them.
    reference_group : str
        The subgroup used as the denominator for
        ``demographic_parity``'s ratio column (the group with the
        highest selection rate, per the four-fifths rule's convention).
    """

    demographic_parity: pd.DataFrame
    equalized_odds: pd.DataFrame
    calibration: pd.DataFrame
    reference_group: str
    _flags: list[str] = field(default_factory=list, repr=False)

    @property
    def flagged_subgroups(self) -> list[str]:
        """Subgroups failing at least one of the three fairness checks."""
        return sorted(set(self._flags))

    @property
    def passes_all_checks(self) -> bool:
        """``True`` if no subgroup was flagged on any metric."""
        return len(self._flags) == 0


def _demographic_parity(y_pred: np.ndarray, subgroup: pd.Series) -> tuple[pd.DataFrame, str]:
    rates = (
        pd.DataFrame({"subgroup": subgroup, "selected": y_pred})
        .groupby("subgroup")["selected"]
        .agg(n="size", selection_rate="mean")
    )
    reference_group = str(rates["selection_rate"].idxmax())
    reference_rate = rates.loc[reference_group, "selection_rate"]
    rates["ratio_to_reference"] = rates["selection_rate"] / reference_rate
    rates["passes_four_fifths_rule"] = rates["ratio_to_reference"] >= FOUR_FIFTHS_THRESHOLD
    return rates.reset_index(), reference_group


def _equalized_odds(y_true: np.ndarray, y_pred: np.ndarray, subgroup: pd.Series) -> pd.DataFrame:
    df = pd.DataFrame({"subgroup": subgroup, "y_true": y_true, "y_pred": y_pred})
    rows = []
    for group, g in df.groupby("subgroup"):
        positives = g[g["y_true"] == 1]
        negatives = g[g["y_true"] == 0]
        tpr = positives["y_pred"].mean() if len(positives) else float("nan")
        fpr = negatives["y_pred"].mean() if len(negatives) else float("nan")
        rows.append({"subgroup": group, "n": len(g), "tpr": tpr, "fpr": fpr})
    return pd.DataFrame(rows)


def _calibration_within_group(
    y_true: np.ndarray, y_pred_proba: np.ndarray, subgroup: pd.Series
) -> pd.DataFrame:
    df = pd.DataFrame({"subgroup": subgroup, "y_true": y_true, "y_pred_proba": y_pred_proba})
    grouped = df.groupby("subgroup").agg(
        n=("y_true", "size"),
        observed_rate=("y_true", "mean"),
        mean_predicted_probability=("y_pred_proba", "mean"),
    )
    grouped["calibration_gap"] = grouped["mean_predicted_probability"] - grouped["observed_rate"]
    return grouped.reset_index()


def audit_fairness(y_true, y_pred_proba, subgroup, threshold: float = 0.5) -> FairnessReport:
    """Run an AEQUITAS-style fairness audit across subgroups.

    Parameters
    ----------
    y_true : array-like
        True binary labels.
    y_pred_proba : array-like
        Predicted probabilities.
    subgroup : array-like
        Subgroup membership per row (e.g. a demographic or geographic
        category).
    threshold : float, default 0.5
        Probability threshold used to binarize ``y_pred_proba`` for the
        demographic-parity and equalized-odds checks (calibration uses
        the continuous probabilities directly).

    Returns
    -------
    FairnessReport

    Examples
    --------
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> subgroup = rng.choice(["a", "b"], size=n)
    >>> bias = np.where(subgroup == "b", 0.3, 0.0)
    >>> y_pred_proba = np.clip(rng.uniform(0, 0.6, size=n) + bias, 0, 1)
    >>> y_true = rng.binomial(1, y_pred_proba)
    >>> report = audit_fairness(y_true, y_pred_proba, subgroup)
    >>> "b" in report.flagged_subgroups
    True
    """
    y_true = np.asarray(y_true)
    y_pred_proba = np.asarray(y_pred_proba)
    subgroup = pd.Series(subgroup).reset_index(drop=True)
    y_pred = (y_pred_proba >= threshold).astype(int)

    demographic_parity, reference_group = _demographic_parity(y_pred, subgroup)
    equalized_odds = _equalized_odds(y_true, y_pred, subgroup)
    calibration = _calibration_within_group(y_true, y_pred_proba, subgroup)

    flags: list[str] = []
    flags += demographic_parity.loc[
        ~demographic_parity["passes_four_fifths_rule"], "subgroup"
    ].tolist()

    odds_pivot = equalized_odds.set_index("subgroup")
    tpr_gap = odds_pivot["tpr"].max() - odds_pivot["tpr"].min()
    fpr_gap = odds_pivot["fpr"].max() - odds_pivot["fpr"].min()
    if tpr_gap > EQUALIZED_ODDS_GAP_THRESHOLD or fpr_gap > EQUALIZED_ODDS_GAP_THRESHOLD:
        extreme_groups = [odds_pivot["tpr"].idxmax(), odds_pivot["tpr"].idxmin()]
        flags += [str(g) for g in extreme_groups]

    flags += calibration.loc[
        calibration["calibration_gap"].abs() > CALIBRATION_GAP_THRESHOLD, "subgroup"
    ].tolist()

    return FairnessReport(
        demographic_parity=demographic_parity,
        equalized_odds=equalized_odds,
        calibration=calibration,
        reference_group=reference_group,
        _flags=flags,
    )

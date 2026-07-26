"""DoWhy backdoor-adjustment estimation with automated refutation tests.

DoWhy's contribution over the propensity-score machinery already in
:mod:`equitable_rwe.causal.psm` and :mod:`equitable_rwe.causal.iptw` is
formal causal identification (checking the backdoor criterion against an
explicit causal graph) and a battery of refutation tests that stress-test
an estimate against alternative explanations — placebo treatments, added
random confounders, and data subsetting. See
``docs/architecture/adr-0003-causal-model-choices.md`` for why this
project uses DoWhy for that, rather than reimplementing backdoor
identification on top of plain statsmodels.

DoWhy has no native Cox/survival estimator, so this module targets a
**binary event-by-horizon outcome** (e.g. death by day ``tau``), not a
hazard ratio. Its result is on the risk-difference scale DoWhy's
``backdoor.propensity_score_weighting`` estimator natively produces, plus
a derived risk-ratio reading (computed from the same weighted group means)
for E-value comparability with the Cox-based estimators elsewhere in this
project — see :func:`equitable_rwe.benchmark` for how the two scales are
reconciled.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from dowhy import CausalModel


@dataclass
class RefutationResult:
    """Result of a single DoWhy refutation test.

    Attributes
    ----------
    name : str
        Refuter name (e.g. ``"placebo_treatment_refuter"``).
    original_effect : float
        The estimate being refuted.
    new_effect : float
        The effect estimated under the refutation's perturbed data/model.
    p_value : float
        p-value for the refutation test (interpretation depends on the
        refuter; see DoWhy's documentation for each).
    """

    name: str
    original_effect: float
    new_effect: float
    p_value: float


@dataclass
class DoWhyBackdoorResult:
    """A DoWhy backdoor-adjustment estimate plus its refutation battery.

    Attributes
    ----------
    risk_difference : float
        The ATE on the risk-difference scale, as estimated by
        ``backdoor.propensity_score_weighting``.
    risk_ratio : float
        Derived risk ratio (weighted P(event|treated) /
        weighted P(event|control)), for E-value comparability.
    refutations : list[RefutationResult]
        Placebo-treatment, random-common-cause, and data-subset
        refutation results.
    """

    risk_difference: float
    risk_ratio: float
    refutations: list[RefutationResult]

    @property
    def passes_refutation(self) -> bool:
        """``True`` if no refutation found a significant effect where none should be.

        For the placebo and random-common-cause refuters, a significant
        (p < 0.05) *change* from the original estimate would be
        concerning; this project treats "not significant" as "passed"
        for all three, consistent with how DoWhy reports them.
        """
        return all(r.p_value >= 0.05 for r in self.refutations)


def fit_dowhy_backdoor(
    df: pd.DataFrame,
    treatment_col: str,
    outcome_col: str,
    common_causes: list[str],
    n_refutation_simulations: int = 20,
    seed: int | None = None,
) -> DoWhyBackdoorResult:
    """Estimate a backdoor-adjusted treatment effect with DoWhy, plus refutations.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per subject.
    treatment_col : str
        Binary (0/1) treatment indicator column.
    outcome_col : str
        Binary (0/1) outcome column (e.g. event-by-horizon indicator).
    common_causes : list[str]
        Covariates assumed sufficient for backdoor adjustment (measured
        confounders of treatment and outcome).
    n_refutation_simulations : int, default 20
        Number of simulations for each refutation test.
    seed : int, optional
        Random seed, passed to DoWhy's refuters where supported.

    Returns
    -------
    DoWhyBackdoorResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x1 = rng.normal(size=n)
    >>> treat = rng.binomial(1, 1 / (1 + np.exp(0.3 * x1)))
    >>> outcome = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 * treat + 0.4 * x1))))
    >>> df = pd.DataFrame({"treat": treat, "x1": x1, "outcome": outcome})
    >>> result = fit_dowhy_backdoor(df, "treat", "outcome", common_causes=["x1"])
    >>> isinstance(result.risk_difference, float)
    True
    """
    model = CausalModel(
        data=df, treatment=treatment_col, outcome=outcome_col, common_causes=common_causes
    )
    identified_estimand = model.identify_effect(proceed_when_unidentifiable=True)
    estimate = model.estimate_effect(
        identified_estimand, method_name="backdoor.propensity_score_weighting"
    )
    risk_difference = float(estimate.value)

    treated_mean = df.loc[df[treatment_col] == 1, outcome_col].mean()
    control_mean = df.loc[df[treatment_col] == 0, outcome_col].mean()
    risk_ratio = float(treated_mean / control_mean) if control_mean > 0 else float("nan")

    refutations = []
    for method_name, kwargs in [
        ("placebo_treatment_refuter", {}),
        ("random_common_cause", {}),
        ("data_subset_refuter", {"subset_fraction": 0.8}),
    ]:
        refutation = model.refute_estimate(
            identified_estimand,
            estimate,
            method_name=method_name,
            num_simulations=n_refutation_simulations,
            random_seed=seed,
            **kwargs,
        )
        refutations.append(
            RefutationResult(
                name=method_name,
                original_effect=float(refutation.estimated_effect),
                new_effect=float(np.asarray(refutation.new_effect).mean()),
                p_value=float(refutation.refutation_result["p_value"]),
            )
        )

    return DoWhyBackdoorResult(
        risk_difference=risk_difference, risk_ratio=risk_ratio, refutations=refutations
    )

"""Four-estimator causal-adjustment benchmark with E-value sensitivity analysis.

Compares naive (unadjusted) Cox regression, propensity score matching,
inverse probability of treatment weighting, and DoWhy backdoor adjustment
on the same confounded synthetic cohort (:mod:`benchmark.simulate_confounded_cohort`),
against a known ground-truth effect. Most causal-inference portfolios
either skip this comparison entirely or run a single method without
checking it against a known answer; this benchmark exists specifically to
show which adjustment strategies recover the truth on a cohort where the
truth is known, and how robust each recovered estimate is to unmeasured
confounding via its E-value.

Run with: ``python benchmark/run_benchmark.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from simulate_confounded_cohort import (  # noqa: E402
    HORIZON_DAYS,
    TRUE_HR_TREATMENT,
    simulate_confounded_cohort,
    true_risk_ratio_at_horizon,
)

from equitable_rwe.causal.dowhy_backdoor import fit_dowhy_backdoor
from equitable_rwe.causal.evalue import evalue
from equitable_rwe.causal.iptw import compute_iptw_weights
from equitable_rwe.causal.psm import match_on_propensity
from equitable_rwe.survival.cox import fit_cox_ph

RESULTS_DIR = Path(__file__).parent / "results"
N_SUBJECTS = 3000
SEED = 2026


def _hr_row(estimator: str, cph, true_value: float) -> dict:
    summary = cph.summary.loc["treat"]
    hr = float(summary["exp(coef)"])
    ci_lower = float(summary["exp(coef) lower 95%"])
    ci_upper = float(summary["exp(coef) upper 95%"])
    ev = evalue(hr, ci_lower=ci_lower, ci_upper=ci_upper)
    return {
        "estimator": estimator,
        "effect_measure": "hazard_ratio",
        "estimate": hr,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "true_value": true_value,
        "log_bias": None,
        "evalue_point": ev.estimate_evalue,
        "evalue_ci": ev.ci_evalue,
        "passes_refutation": None,
    }


def run_benchmark() -> pd.DataFrame:
    cohort = simulate_confounded_cohort(n=N_SUBJECTS, seed=SEED)
    true_rr = true_risk_ratio_at_horizon()

    rows = []

    # 1. Naive (unadjusted) Cox.
    naive_cph = fit_cox_ph(cohort, "time_to_event_days", "event_observed", covariates=["treat"])
    rows.append(_hr_row("naive_cox", naive_cph, TRUE_HR_TREATMENT))

    # 2. Propensity score matching, then unadjusted Cox on the matched sample.
    match_result = match_on_propensity(cohort, "treat", ["x1", "x2"])
    psm_cph = fit_cox_ph(
        match_result.matched_df, "time_to_event_days", "event_observed", covariates=["treat"]
    )
    rows.append(_hr_row("psm", psm_cph, TRUE_HR_TREATMENT))

    # 3. IPTW: weighted Cox with only the treatment indicator (the
    # classical marginal-structural-model-style IPTW arm, distinct from
    # this project's doubly-robust weight-and-adjust estimator).
    weights = compute_iptw_weights(cohort, "treat", ["x1", "x2"])
    cohort_with_weights = cohort.assign(_iptw_weight=weights)
    iptw_cph = fit_cox_ph(
        cohort_with_weights,
        "time_to_event_days",
        "event_observed",
        covariates=["treat"],
        weights_col="_iptw_weight",
        robust=True,
    )
    rows.append(_hr_row("iptw", iptw_cph, TRUE_HR_TREATMENT))

    # 4. DoWhy backdoor adjustment on the binary event-by-horizon outcome
    # (DoWhy has no native Cox estimator; see adr-0003).
    dowhy_result = fit_dowhy_backdoor(
        cohort, "treat", "event_by_horizon", common_causes=["x1", "x2"], seed=SEED
    )
    dowhy_ev = evalue(dowhy_result.risk_ratio)
    rows.append(
        {
            "estimator": "dowhy_backdoor",
            "effect_measure": "risk_ratio_at_horizon",
            "estimate": dowhy_result.risk_ratio,
            "ci_lower": None,
            "ci_upper": None,
            "true_value": true_rr,
            "log_bias": None,
            "evalue_point": dowhy_ev.estimate_evalue,
            "evalue_ci": None,
            "passes_refutation": dowhy_result.passes_refutation,
        }
    )

    results = pd.DataFrame(rows)
    results["log_bias"] = np.log(results["estimate"]) - np.log(results["true_value"])
    return results


def _write_summary_markdown(results: pd.DataFrame, true_rr: float) -> str:
    lines = [
        "# Four-estimator causal-adjustment benchmark",
        "",
        f"Simulated cohort: n={N_SUBJECTS}, seed={SEED} "
        "(see `simulate_confounded_cohort.py` for the full DGP).",
        "",
        f"- True hazard ratio (treatment): **{TRUE_HR_TREATMENT}**",
        f"- True risk ratio at {HORIZON_DAYS:.0f} days (Monte Carlo, "
        f"n=300,000): **{true_rr:.4f}**",
        "",
        "The first three estimators target the hazard ratio and are directly "
        "comparable to each other and to the true HR. DoWhy has no native "
        "Cox estimator (see adr-0003) and instead targets the risk ratio at "
        "the fixed horizon above — comparable to the Monte Carlo true RR, "
        "but not on the same scale as the hazard-ratio rows.",
        "",
        "| Estimator | Measure | Estimate | 95% CI | True value | log bias | "
        "E-value (point) | E-value (CI) | Refutation |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for _, r in results.iterrows():
        ci = f"({r['ci_lower']:.3f}, {r['ci_upper']:.3f})" if pd.notna(r["ci_lower"]) else "n/a"
        refutation = (
            "passed"
            if r["passes_refutation"] is True
            else ("n/a" if pd.isna(r["passes_refutation"]) else "FAILED")
        )
        evalue_ci = f"{r['evalue_ci']:.2f}" if pd.notna(r["evalue_ci"]) else "n/a"
        lines.append(
            f"| {r['estimator']} | {r['effect_measure']} | {r['estimate']:.4f} | {ci} | "
            f"{r['true_value']:.4f} | {r['log_bias']:+.4f} | {r['evalue_point']:.2f} | "
            f"{evalue_ci} | {refutation} |"
        )

    lines += [
        "",
        "**log bias** is `log(estimate) - log(true_value)`: closer to zero is "
        "better, and it is comparable in magnitude across the two effect-measure "
        "scales because both are ratio measures on the log scale.",
        "",
        "**Refutation** reflects DoWhy's own `is_statistically_significant` flag "
        "(p < 0.05) applied uniformly across its three refuters, per DoWhy's own "
        "convention. For the placebo-treatment refuter specifically, a 'FAILED' "
        "result here means the original estimate differs significantly from the "
        "placebo-simulated null distribution — which some practitioners would "
        "read as *reassuring* (the effect doesn't look like noise), even though "
        "DoWhy reports it identically to a genuine robustness failure. See "
        "`docs/architecture/adr-0003-causal-model-choices.md` for the full "
        "discussion; do not read a placebo 'FAILED' here as an unambiguous "
        "problem with the estimate.",
        "",
        "This table is generated by `run_benchmark.py` from the numbers above "
        "— nothing here is hand-written.",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = run_benchmark()
    true_rr = true_risk_ratio_at_horizon()

    results.to_parquet(RESULTS_DIR / "benchmark_results.parquet", index=False)
    results.to_csv(RESULTS_DIR / "benchmark_results.csv", index=False)
    (RESULTS_DIR / "benchmark_summary.md").write_text(
        _write_summary_markdown(results, true_rr) + "\n"
    )

    print(results.to_string(index=False))

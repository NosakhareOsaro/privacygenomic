# ADR-0003: Causal inference module implementation choices

## Status

Accepted

## Context

`equitable_rwe.causal` needed propensity score matching, IPTW, a
DoWhy-based backdoor adjustment with refutation tests, E-value
sensitivity analysis, and a doubly-robust combined estimator. Three
decisions were non-obvious enough to record.

## Decision 1: DoWhy over pure statsmodels/scikit-learn IPTW

The propensity-score machinery in `psm.py` and `iptw.py` is built
directly on `scikit-learn` (`LogisticRegression`, `NearestNeighbors`) —
no R interop, no DoWhy needed for those. DoWhy is used specifically for
`dowhy_backdoor.py`, and specifically for what it adds *beyond*
propensity-score adjustment: **formal identification** (checking a
proposed adjustment set against the backdoor criterion on an explicit
causal graph, rather than assuming the analyst's covariate list is
sufficient) and a **standardized refutation battery** (placebo treatment,
random common cause, data subset) that would otherwise have to be
hand-implemented per project. Reimplementing backdoor identification from
scratch on top of statsmodels was considered and rejected: it is a solved
problem with a mature, widely used implementation already available, and
hand-rolling it would add correctness risk (subtle graph-theoretic bugs)
for no benefit over depending on the library that specializes in it.

**Consequence:** DoWhy has no native Cox/survival estimator, so
`fit_dowhy_backdoor` targets a binary event-by-horizon outcome rather
than a hazard ratio, and reports a risk difference (DoWhy's native scale
for `backdoor.propensity_score_weighting`) plus a derived risk ratio for
E-value comparability with the Cox-based estimators. This means DoWhy's
result in the four-estimator benchmark (`benchmark/`) is not on the
identical scale as the other three, and the benchmark reports say so
explicitly rather than presenting a misleadingly uniform table.

## Decision 2: doubly robust = weight-and-adjust, not full AIPW

`fit_doubly_robust_survival` combines IPTW weights with covariate
adjustment in the same weighted Cox model. This is the standard, widely
used "doubly robust" construction for a marginal treatment effect (Austin
2013; Funk et al. 2011): consistent if either the propensity model or the
outcome model is correctly specified. It is **not** a fully augmented
AIPW/TMLE estimator with an explicit augmentation term for the survival
setting (e.g. Bang & Robins, 2005-style augmented IPCW). A full survival
AIPW estimator was considered and rejected for this release: it requires
estimating and correctly combining a censoring model, an outcome model,
and a propensity model with a specific augmentation term whose derivation
and validation is substantial additional work with real correctness risk
if rushed. The simpler weight-and-adjust device is well established,
correctly cited as such, and not overstated as more than it is anywhere
in this project's docstrings, README, or paper draft.

## Decision 3: E-value applied to hazard ratios as an approximate risk ratio

`evalue.py` implements the VanderWeele & Ding (2017) E-value formula for
a risk-ratio-scale input. This project applies it directly to Cox hazard
ratios, which is standard practice but is only a good approximation to
the risk-ratio E-value when cumulative outcome incidence is low (roughly
≲15%) in both compared groups. This project does not implement the
correction needed for common outcomes (converting HR to RR under high
incidence) because doing so from memory, without a verified reference
formula to check it against, risked introducing a subtly wrong
"corrected" number that would be worse than clearly stating the
approximation's limits. `evalue.py`'s docstring states this limitation
directly.

## A refutation-interpretation nuance found while running the benchmark

DoWhy's `test_significance` uses one uniform convention across all three
refuters: the null hypothesis is "the original estimate is part of the
refuter's simulated null distribution," and `is_statistically_significant
= True` (p < 0.05) means that hypothesis is rejected — DoWhy's own code
comments label this outcome "fails the test" for every refuter, including
the placebo-treatment refuter. This project's `passes_refutation` follows
that same convention (small p ⇒ fail) for consistency with DoWhy's own
labeling, rather than inventing a different rule.

Worth being explicit about: for the **placebo-treatment refuter**
specifically, "the original estimate differs significantly from the
placebo-simulated null distribution" is arguably a *reassuring* signal in
substantive terms — the real estimate doesn't look like noise — even
though DoWhy's own convention reports it identically to a genuine
robustness failure. Reasonable practitioners read this refuter's result
differently depending on which framing they start from. This project
reports DoWhy's raw `is_statistically_significant` flag as-is
(`benchmark/results/benchmark_summary.md`) rather than silently
resolving that ambiguity one way, and this paragraph exists so a reader
of the benchmark results isn't misled by a bare "FAILED" into assuming
something is wrong with the underlying causal estimate.

## Consequences

- The four-estimator benchmark compares naive Cox, PSM+Cox, and IPTW+Cox
  hazard ratios directly against a known simulated ground-truth hazard
  ratio, and separately compares DoWhy's risk-difference/risk-ratio
  estimate against a Monte-Carlo-derived ground-truth risk ratio at the
  same horizon — two comparisons on two scales, not one table pretending
  they're identical.
- Anyone extending this project to a fully augmented survival AIPW
  estimator has a clear, documented starting point and a clear reason
  the simpler estimator was chosen first (time and correctness-risk
  budget for a first release, not a belief that weight-and-adjust is
  sufficient for every use case).

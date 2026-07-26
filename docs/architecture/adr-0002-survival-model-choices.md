# ADR-0002: Survival module implementation choices

## Status

Accepted

## Context

`equitable_rwe.survival` needed to cover Kaplan-Meier, Cox PH with
diagnostics, Fine-Gray competing risks, a Royston-Parmar-family flexible
parametric model, landmark analysis, and RMST. `lifelines` provides solid,
well-tested implementations of the first, second, and last of these, plus
a nonparametric competing-risks estimator (Aalen-Johansen) — but not
Fine-Gray regression, and not the classical proportional-hazards-scale
Royston-Parmar spline model. Three decisions followed from that gap.

## Decision 1: Fine-Gray via IPCW long-format Cox, not a from-scratch subdistribution likelihood

Fine & Gray's (1999) subdistribution hazard model can be fit by
constructing a modified risk set — competing-event subjects remain
"at risk" past their event time with an inverse-probability-of-censoring
weight — and running an ordinary (weighted) Cox partial likelihood on the
resulting counting-process data. This is exactly what R's
`survival::finegray()` + `coxph()` does, and it lets
`equitable_rwe.survival.competing_risks.fit_fine_gray` reuse lifelines'
`CoxTimeVaryingFitter` rather than hand-deriving and optimizing a novel
subdistribution partial likelihood from scratch. **Correctness check
performed during development:** on data with no competing events at all,
this implementation's coefficient matches a plain `CoxPHFitter` fit to
7 significant figures — the expected behavior, since the two constructions
are identical once there is nothing to reweight.

Two consequences of this choice are recorded rather than hidden:

- **Standard errors are model-based, not robust/sandwich.** lifelines'
  `CoxTimeVaryingFitter.fit(..., robust=True)` currently raises
  `NotImplementedError` in the installed version (0.30.x). A reference
  Fine-Gray implementation reports cluster-robust standard errors,
  because the IPCW-reweighted "observations" are not independent within
  subject. This implementation's point estimates (`coef`, `exp(coef)`)
  are unaffected, but its p-values and confidence intervals are somewhat
  too narrow; `FineGrayResult` documents this directly rather than
  presenting the naive SEs without qualification.
- **The exact construction is `O(n_competing × n_event_times)` rows.** For
  a few thousand subjects this reaches the low millions and fitting slows
  accordingly (empirically: ~40s at n=4000 with the exact grid). `fit_fine_gray`
  defaults to `max_event_times=200`, coarsening the event-of-interest time
  grid to an evenly spaced subset — a standard practical mitigation (the
  same idea `finegray()` supports via a coarser time argument) — bringing
  n=4000 down to ~5s. Callers who need the exact construction can pass
  `max_event_times=None`.

## Decision 2: Royston-Parmar via lifelines' CRCSplineFitter (AFT scale), not a hand-rolled PH-scale spline model

The original Royston & Parmar (2002) model works on the proportional
hazards (or proportional odds) scale. lifelines does not ship that
variant, but does ship `CRCSplineFitter` — Crowther, Royston & Clements'
(2014) accelerated-failure-time generalization of the same restricted-cubic-spline
idea. Hand-rolling a custom PH-scale spline model via lifelines'
`ParametricRegressionFitter` base class was considered and rejected: it
would require deriving and differentiating (via `autograd`) a novel
cumulative hazard function with no reference implementation in this
project to validate against, which is a correctness risk this project is
not willing to carry for a portfolio-quality release. `fit_royston_parmar`
wraps `CRCSplineFitter` directly, documented honestly as the AFT-scale
member of the Royston-Parmar family rather than mislabeled as the
original PH-scale model.

A known caveat, observed during development on some simulated datasets:
`CRCSplineFitter` can raise a `StatisticalWarning` about a negative
variance-matrix diagonal, a documented numerical fragility of this spline
model family, more likely with more baseline knots relative to the number
of events. `fit_royston_parmar`'s docstring recommends reducing
`n_baseline_knots` if this occurs.

## Decision 3: a structured Schoenfeld-residual test, not lifelines' `check_assumptions`

`CoxPHFitter.check_assumptions()` prints a human-readable diagnostic and
returns `None`. Since this project wants CI-checkable proportional-hazards
diagnostics (and to test its own correctness), `equitable_rwe.survival.cox.test_proportional_hazards`
reimplements the same idea — correlating scaled Schoenfeld residuals
(computed via lifelines' `compute_residuals`) against rank-transformed
event time, per Grambsch & Therneau (1994) — as a plain
`ProportionalHazardsTestResult` dataclass with per-covariate correlations
and p-values.

## Consequences

- Fine-Gray results in this project should be read as "point estimates
  reliable, uncertainty intervals somewhat optimistic" until lifelines
  ships robust variance for `CoxTimeVaryingFitter`, or this project
  implements a manual cluster-robust sandwich estimator (not done, to
  avoid re-deriving variance machinery lifelines already owns elsewhere).
- The Royston-Parmar model in this project answers an AFT-scale question
  ("how does this covariate accelerate or decelerate time to event")
  rather than a PH-scale one ("what is this covariate's hazard ratio") —
  those are different, not interchangeable, quantities, and any report
  generated from `fit_royston_parmar` should say so explicitly.

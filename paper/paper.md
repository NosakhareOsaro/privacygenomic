---
title: 'equitable_rwe: doubly-robust causal survival estimation with integrated subgroup fairness auditing'
tags:
  - Python
  - causal inference
  - survival analysis
  - health equity
  - algorithmic fairness
  - real-world evidence
  - OMOP CDM
authors:
  - name: Nosakhare Osaro
    affiliation: 1
affiliations:
  - name: Independent Researcher
    index: 1
date: 26 July 2026
bibliography: paper.bib
---

## Summary

`equitable_rwe` is a Python library for causally valid, fairness-audited
analysis of real-world clinical outcome data. It combines survival
analysis (Kaplan-Meier, Cox proportional hazards with a structured
proportional-hazards diagnostic, Fine-Gray competing-risks regression,
a Royston-Parmar-family flexible parametric model, landmark analysis,
and restricted mean survival time), causal inference (propensity score
matching, inverse probability of treatment weighting, a doubly-robust
IPTW-weighted and covariate-adjusted Cox estimator, DoWhy-based backdoor
adjustment with automated refutation testing, and E-value sensitivity
analysis for unmeasured confounding), and subgroup fairness auditing
(an AEQUITAS-equivalent classifier audit — demographic parity, equalized
odds, within-group calibration — and a causal-estimator reliability
audit that checks per-subgroup effective sample size and covariate
balance rather than a classifier-style metric that does not apply to a
hazard ratio). Its headline API, `fit_equitable_survival_estimate`,
returns a doubly-robust causal treatment-effect estimate and a subgroup
fairness report from a single function call.

The library is one component of PrivacyGenomic, a reproducible pipeline
that also includes an OMOP Common Data Model extraction and
Achilles-style quality-control layer for MIMIC-IV-shaped electronic
health record data, a synthetic Canadian health-equity cohort generator
calibrated against two cited public statistics (2021 Census Indigenous
population share; CIHI's 2024-2025 national rural/urban hospitalization
rate ratio), an XGBoost 30-day readmission model with SHAP
explainability, and an open-source (Streamlit/Plotly) health-equity
dashboard with age-standardized rates and a GIS choropleth.

## Statement of need

Observational (real-world) clinical data is confounded by design: the
patients who receive a given treatment differ systematically from those
who do not, and those differences are frequently also associated with
outcomes and with membership in equity-relevant subgroups. Two
methodological gaps recur across the tools researchers reach for when
analyzing this kind of data.

First, causal-adjustment methods are usually evaluated in isolation. A
typical analysis picks one adjustment strategy — propensity matching,
say, or inverse probability weighting — fits it once, and reports the
result, without checking whether that method recovers a known answer on
comparable data, and without a standardized robustness check against
unmeasured confounding. `equitable_rwe` packages a four-estimator
benchmark (naive Cox, propensity score matching, IPTW, and DoWhy backdoor
adjustment) against a synthetic cohort with an exact, known
ground-truth hazard ratio, alongside E-value sensitivity analysis
[@vanderweele2017] for every estimator it produces. On the benchmark
cohort distributed with this package, propensity matching and IPTW each
cut the naive (unadjusted) estimator's log-scale bias relative to the
true hazard ratio by roughly half or more — a concrete, reproducible
illustration of why adjustment method choice matters, not just
adjustment itself (see `benchmark/results/benchmark_summary.md` in the
repository for the exact, code-generated numbers).

Second, subgroup fairness auditing and causal inference are usually
built as separate tools answering separate questions, even when a single
analysis needs both — is this treatment effect estimate valid, *and* is
it equally well supported by evidence across the subgroups a health
system cares about, such as remoteness or an equity-relevant demographic
category? Existing fairness toolkits such as Aequitas [@saleiro2018] are
built for classifier outputs (selection rates, true/false positive
rates) that have no direct analogue for a Cox hazard ratio. `equitable_rwe`
treats these as two distinct questions requiring two distinct audits: a
classifier-style AEQUITAS-equivalent audit for its readmission-risk
model, and a causal-estimator reliability audit — per-subgroup effective
sample size and post-weighting covariate balance — for its doubly-robust
survival estimator, integrated into one combined API call rather than
requiring two disconnected tools.

## Implementation notes

Several components fill gaps in the existing Python survival/causal
ecosystem rather than duplicating it. `lifelines` [@davidsonpilon2019]
provides Kaplan-Meier and Cox estimation but no Fine-Gray subdistribution
hazard model [@finegray1999]; `equitable_rwe` implements the standard
inverse-probability-of-censoring-weighted long-format construction on top
of `lifelines`' `CoxTimeVaryingFitter`, validated during development by
confirming its estimate matches a plain Cox fit to seven significant
figures in the degenerate case with no competing events. `lifelines` also
ships a spline-based flexible parametric model on the accelerated-failure-time
scale (`CRCSplineFitter`, after Crowther, Royston & Clements) rather than
the original Royston-Parmar proportional-hazards-scale model
[@roystonparmar2002]; `equitable_rwe` wraps that estimator directly and
documents the scale distinction rather than mislabeling an AFT-scale
result as a hazard ratio. DoWhy [@sharma2020] supplies formal backdoor
identification and a standardized refutation battery (placebo treatment,
random common cause, data subsetting) that `equitable_rwe` uses for its
backdoor-adjustment estimator rather than reimplementing causal graph
identification from scratch. The XGBoost [@chen2016] readmission model
and its SHAP [@lundberg2017] explanations deliberately exclude protected
and proxy-for-protected attributes from the feature set while still using
them for the post-hoc fairness audit, illustrating — at small,
synthetic scale — the same finding as Obermeyer et al.
[@obermeyer2019]: that a model can be disparate in its effect on a group
without that group's membership ever being a model input.

Every architectural trade-off in the library — including the ones above
— is recorded as a short Architecture Decision Record in the source
repository (`docs/architecture/`), each stating not just what was
implemented but what was considered and rejected, and why.

## Data and reproducibility

No real patient data is used anywhere in this project. Test fixtures and
demonstration cohorts are either fully synthetic or match a public
dataset's schema without containing any data derived from it (see
`docs/data-governance/ocap-and-privacy.md` for the project's data
governance rationale, including why it does not use or approximate real
Indigenous-identity patient data). Every dataset generator is committed
alongside a script that reproduces its output deterministically from a
fixed seed.

## Acknowledgements

This library builds on `lifelines`, `DoWhy`, `scikit-learn`
[@pedregosa2011], `XGBoost`, and `SHAP`; it would not exist without that
foundation.

## References

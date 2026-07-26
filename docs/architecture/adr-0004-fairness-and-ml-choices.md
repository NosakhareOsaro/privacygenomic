# ADR-0004: Fairness auditing and readmission model choices

## Status

Accepted

## Context

The project needed an XGBoost 30-day readmission model with SHAP
explainability and an AEQUITAS-equivalent fairness audit, plus this
project's headline feature: a single API call returning both a causal
estimate and a fairness report. Three decisions shaped how that came
together.

## Decision 1: reimplement Aequitas's three core metrics, don't depend on the package

`equitable_rwe.fairness.audit` reimplements demographic parity, equalized
odds, and within-group calibration directly on `numpy`/`pandas`, rather
than depending on the `aequitas` package. Aequitas is oriented around its
own `Audit` class and a specific input-table shape; this project's
predictions already exist as plain arrays from scikit-learn-style model
output (`fit_readmission_model` and this project's other classifiers),
and the three metrics themselves are simple enough (a handful of
`groupby` aggregations) that adapting to Aequitas's API surface would add
a dependency and an integration layer for less code than reimplementing
them directly. The four-fifths rule threshold and the general metric
definitions are taken directly from the standard practice Aequitas itself
follows, so results should agree in substance, not just in spirit.

## Decision 2: exclude protected/proxy attributes from the readmission model's features, but not from its label

`data/synthetic/generate_readmission_cohort.py` deliberately excludes
`indigenous_identity_proxy` and `remoteness_level` from
`FEATURE_COLUMNS`, mirroring the realistic practice of not feeding
protected or proxy-for-protected attributes into a clinical risk model —
while still using them to generate the label (via a simulated
reduced-post-discharge-follow-up-access effect) and reserving them
*purely* for the post-hoc fairness audit. This means the benchmark
scenario for `equitable_rwe.fairness.audit_fairness` is the harder, more
realistic one: can the audit detect a disparity that the model was never
told to reproduce, but reproduces anyway through proxy structure in its
legitimate clinical features (prior admissions, comorbidity count, and
so on, which are themselves correlated with the excluded attributes)?
This mirrors, in miniature, the real and well-documented finding in
Obermeyer, Powers, Vogeli & Mullainathan (2019, *Science*) that a
widely-used commercial health risk algorithm exhibited significant
racial bias despite never using race as an input. This project's
synthetic version is illustrative, not a reproduction of that study's
methodology or findings.

## Decision 3: the causal flagship's fairness report is a reliability audit, not a demographic-parity check

`equitable_rwe.fairness.audit_fairness`'s three metrics are inherently
classifier-shaped (a selection rate, a true/false positive rate) and do
not have a meaningful analogue for a Cox hazard ratio — there is no
"positive prediction rate" for a continuous regression coefficient.
Rather than force a mismatched metric onto
`fit_equitable_survival_estimate` (e.g. binarizing the hazard ratio
somehow), `equitable_rwe.fairness.subgroup_causal.audit_causal_subgroup_fairness`
asks the question that *does* transfer to a causal estimate: is the
evidence behind this estimate equally strong for every subgroup — enough
events, adequate effective sample size after weighting, adequate
post-weighting covariate balance — or does the topline number quietly
rest on much weaker support for some group? This is a genuinely different
kind of "fairness" than demographic parity (evidence adequacy, not
outcome parity), and `EquitableSurvivalEstimate.is_equitable`'s docstring
says so explicitly, rather than implying it answers "does this treatment
work equally well for everyone" — a question this project's data
(subgroup-stratified refits) does support via `fairness_report.per_subgroup`,
but which is a different claim from `is_equitable`.

## Consequences

- `fit_equitable_survival_estimate` currently refits the doubly-robust
  model on the full cohort twice (once directly, once inside the
  fairness audit's `overall` computation) — a known, minor inefficiency
  accepted in exchange for `audit_causal_subgroup_fairness` remaining
  usable standalone, without requiring a pre-fitted result to be threaded
  through.
- A subgroup with too few events (< 10, see
  `MIN_EVENTS_PER_SUBGROUP`) gets no subgroup-specific hazard ratio at
  all, by design — reporting an unstable point estimate from a handful
  of events would be worse than reporting none.

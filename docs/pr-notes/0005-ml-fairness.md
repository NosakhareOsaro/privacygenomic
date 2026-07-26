# PR: XGBoost readmission model, fairness audit, and the flagship combined API

**Branch:** `feat/ml-fairness` → `main`

## Summary

Adds `equitable_rwe.ml` (XGBoost 30-day readmission model + SHAP),
`equitable_rwe.fairness` (an AEQUITAS-equivalent classifier audit and a
causal-estimator subgroup reliability audit), and this project's headline
deliverable: `equitable_rwe.fit_equitable_survival_estimate`, returning a
doubly-robust causal estimate and a subgroup fairness report from one
call.

## Why

The project's stated differentiator is a combined causal-plus-fairness
API most tutorials don't build. Getting there required first admitting
that "fairness" means something different for a classifier (does this
model select/err on people differently by group?) than for a hazard
ratio (is the *evidence* behind this estimate equally strong across
groups?) — see adr-0004 for why this PR builds two distinct audits
instead of forcing one metric onto both.

## What's included

- `data/synthetic/generate_readmission_cohort.py` — a synthetic cohort
  with a **known, injected** disparity by `remoteness_level` and
  `indigenous_identity_proxy`, used to test the fairness tooling against
  a ground truth. Protected/proxy attributes are excluded from the
  model's feature set but still drive the label, illustrating (not
  reproducing) the real, published finding that an algorithm can be
  disparate without ever using a protected attribute as an input
  (Obermeyer et al., 2019, *Science*).
- `ml/readmission.py`, `ml/explain.py` — XGBoost classifier + SHAP
  TreeExplainer-based feature importance.
- `fairness/audit.py` — demographic parity (four-fifths rule), equalized
  odds, and within-group calibration, reimplemented directly rather than
  depending on the `aequitas` package (adr-0004).
- `fairness/subgroup_causal.py` — per-subgroup effective sample size and
  post-weighting covariate balance for the doubly-robust causal
  estimator: a reliability audit, not a demographic-parity check, since
  a hazard ratio has no "selection rate."
- `api.py` — `fit_equitable_survival_estimate`, exported at the top level
  of the package (`equitable_rwe.fit_equitable_survival_estimate`).
- 28 new tests, 97% coverage on `src/equitable_rwe`.

## Testing

```
uv run ruff check src tests data benchmark
uv run black --check src tests data benchmark
uv run mypy src
uv run pytest
```

150 passed, 97% coverage.

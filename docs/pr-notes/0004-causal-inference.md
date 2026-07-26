# PR: causal inference module + four-estimator benchmark

**Branch:** `feat/causal-inference` → `main`

## Summary

Adds `equitable_rwe.causal` (PSM, IPTW, a doubly-robust IPTW-weighted-and-adjusted
Cox estimator, DoWhy backdoor adjustment with automated refutation, and
E-value sensitivity analysis) and `benchmark/`, comparing all four
adjustment strategies against a known, exact ground-truth effect on a
purpose-built confounded synthetic cohort.

## Why

A causal estimate without a robustness check (E-value) and without
evidence it can recover a known answer (the benchmark) is not much more
trustworthy than an unadjusted comparison. Most portfolio-grade causal
inference code stops at "here is my adjusted estimate"; this PR adds both
missing pieces.

## What's included

- `propensity.py`, `psm.py`, `iptw.py` — scikit-learn-based propensity
  modeling, nearest-neighbor matching with balance diagnostics (a MatchIt
  equivalent with no R dependency), and stabilized/truncated IPTW weights
  with Kish's effective sample size diagnostic.
- `doubly_robust.py` — this project's flagship causal estimator: IPTW
  weights combined with covariate adjustment in the same Cox model
  (the standard "weight-and-adjust" doubly robust device — explicitly
  *not* claimed to be a full AIPW/TMLE estimator; see adr-0003).
- `dowhy_backdoor.py` — DoWhy-based backdoor identification, a
  propensity-score-weighting estimate, and all three of DoWhy's
  refutation tests (placebo treatment, random common cause, data subset).
- `evalue.py` — VanderWeele & Ding (2017) E-value, validated against
  their own published worked example (RR=3.9 → E-value≈7.26) in the test
  suite.
- `benchmark/simulate_confounded_cohort.py` — a Weibull-proportional-hazards
  DGP with confounding by indication and an *exact* known hazard ratio
  (0.65), plus a Monte Carlo ground-truth risk ratio at a fixed horizon.
- `benchmark/run_benchmark.py` — runs all four estimators, computes
  E-values, and writes `benchmark/results/benchmark_results.{parquet,csv}`
  and a generated `benchmark_summary.md` — every number in that file is
  produced by the code, not hand-written.
- `docs/architecture/adr-0003-causal-model-choices.md` — three trade-offs
  (DoWhy vs. hand-rolled backdoor identification; weight-and-adjust vs.
  full AIPW; E-value on HR as an RR approximation), plus a documented
  nuance found *while running the benchmark*: DoWhy's placebo-refuter
  "significance" flag is genuinely ambiguous to interpret, and this PR
  says so explicitly rather than presenting a bare pass/fail.
- 42 new tests, 97% coverage on `src/equitable_rwe`.

## A finding from the benchmark itself

On the current cohort, the naive (unadjusted) estimate is biased away
from the true hazard ratio by confounding by indication; PSM and IPTW
each cut that log-scale bias roughly in half or more. This is a real,
generated result — see `benchmark/results/benchmark_summary.md` — not an
assumed outcome; the benchmark's own regression tests
(`tests/benchmark/test_run_benchmark.py`) assert this bias-reduction
property so a future change that breaks it fails CI.

## Testing

```
uv run ruff check src tests benchmark
uv run black --check src tests benchmark
uv run mypy src
uv run pytest
```

122 passed, 97% coverage.

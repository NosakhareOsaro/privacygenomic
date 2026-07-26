# PR: survival analysis module

**Branch:** `feat/survival-models` → `main`

## Summary

Adds `equitable_rwe.survival`: Kaplan-Meier, Cox PH with a structured
proportional-hazards test, Fine-Gray competing risks, a Royston-Parmar-family
flexible parametric model, landmark analysis, and bootstrap RMST
differences.

## Why

Every causal and fairness estimate downstream of this module needs a
survival backbone that doesn't silently assume proportional hazards holds
or that competing events don't exist. Most tutorial-grade Python survival
work stops at Kaplan-Meier and a single Cox fit; this PR adds the pieces
that are usually skipped — a real, structured PH diagnostic; genuine
Fine-Gray subdistribution hazard regression (which no mainstream Python
survival library implements); and a flexible parametric alternative for
when PH is doubtful.

## What's included

- `kaplan_meier.py`, `rmst.py`, `landmark.py` — thin, well-tested wrappers
  and small original additions (bootstrap RMST difference test, a
  landmark-time grid table) on top of lifelines primitives.
- `cox.py` — `check_proportional_hazards`, a structured
  (non-print-only) Schoenfeld-residual PH test lifelines doesn't provide
  as a return value.
- `competing_risks.py` — `fit_fine_gray`, a from-scratch implementation
  of the Fine & Gray (1999) IPCW long-format construction on top of
  `lifelines.CoxTimeVaryingFitter`, vectorized for practical performance
  (~5s at n=4000 with the default coarsened time grid, vs. ~40s exact).
  **Correctness validated** during development: with no competing events,
  its coefficient matches a plain Cox fit to 7 significant figures.
- `royston_parmar.py` — wraps lifelines' `CRCSplineFitter` (the AFT-scale
  member of the Royston-Parmar family), honestly labeled as such rather
  than presented as the original PH-scale model lifelines doesn't ship.
- `docs/architecture/adr-0002-survival-model-choices.md` — records three
  genuine trade-offs: why Fine-Gray reuses `CoxTimeVaryingFitter` instead
  of a from-scratch subdistribution likelihood (and the resulting
  non-robust-SE caveat, since lifelines doesn't yet support
  `robust=True` there), why the Royston-Parmar wrapper uses the AFT-scale
  `CRCSplineFitter` instead of a hand-rolled PH-scale spline model, and
  why the PH test is reimplemented as structured output.
- 42 new tests (unit, doctest, and directional-correctness checks against
  simulated ground truth), 96% coverage on `src/equitable_rwe`.

## A bug the test suite caught worth flagging in review

`check_proportional_hazards` was originally named `test_proportional_hazards`
— which pytest's `--doctest-modules` collection over `src/` tried to
collect as a test function itself (`fixture 'cph' not found`). Renamed;
this is a real, generalizable pitfall (never prefix a library function
`test_` in a pytest-scanned source tree), not just a one-off fix.

## Testing

```
uv run ruff check src tests
uv run black --check src tests
uv run mypy src
uv run pytest
```

80 passed, 96% coverage.

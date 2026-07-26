# PR: synthetic health-equity cohort generator + OCAP governance doc

**Branch:** `feat/synthetic-equity-cohort` → `main`

## Summary

Adds the synthetic Canadian chronic-disease cohort generator
(`data/synthetic/generate_equity_cohort.py`) used by every downstream
statistical module in this project, and the OCAP® data-governance
document explaining why this project uses only synthetic and
public-aggregate data.

## Why

The project's differentiators require a cohort with realistic,
citable health-equity structure — not just a generic simulated dataset.
Building it required answering, up front, a governance question this
project cannot sidestep: is it appropriate to use or approximate
Indigenous-identity patient data at all, for an independent portfolio
project outside any relationship with a First Nations governing data
authority? The answer is documented in
`docs/data-governance/ocap-and-privacy.md`: no — synthetic and
public-aggregate data is the only appropriate input here, not a
convenience shortcut.

## What's included

- `generate_equity_cohort()` — synthetic cohort with demographic,
  geographic, clinical, and survival fields.
- Two precisely calibrated targets (Indigenous-proxy population share,
  remote/accessible hospitalization rate ratio), each backed by a cited
  public source and verified against the *actual generated sample* via
  `calibration_report()` — see `data/synthetic/calibration_notes.md` for
  the full observed-vs-directional-vs-illustrative breakdown.
- `docs/data-governance/ocap-and-privacy.md` — OCAP® principles and this
  project's practical governance checklist.
- 10 tests covering determinism, schema, calibration tolerance, and the
  sign (not just presence) of the modeled survival disparity.

## Testing

```
uv run ruff check src tests data
uv run black --check src tests data
uv run mypy src
uv run pytest
```

40 passed, 95% coverage on `src/equitable_rwe`.

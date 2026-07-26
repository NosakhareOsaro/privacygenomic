# PR: health-equity dashboard (age-standardized rates, choropleth, Streamlit)

**Branch:** `feat/equity-dashboard` → `main`

## Summary

Adds `equitable_rwe.equity` (direct age standardization, Poisson rate
regression, a simplified GIS choropleth) and `dashboards/equity_dashboard.py`,
an open-source Streamlit + Plotly dashboard over the synthetic equity
cohort.

## Why

Open-source over Power BI specifically so a reviewer without a Power BI
license can run this — see adr-0005. Building the choropleth also forced
an honest decision this PR documents rather than fudges: this project has
no verified source for real Canadian province boundary polygons, and
approximating them from memory as bounding boxes produces visibly
overlapping, broken-looking regions (provinces' lat/lon bounding boxes
overlap even though the provinces themselves don't). The choropleth
renders a tile-grid cartogram instead — a real, recognized cartographic
technique for exactly this situation, not a made-up workaround.

## What's included

- `age_standardization.py` — `direct_standardize` (classic direct
  method, Poisson-based CI) and `fit_poisson_rate_model` (Poisson GLM
  with a log-population offset, for adjusted rate ratios). No default
  external standard population is shipped, since this project has no
  verified source for one's exact published weights to cite correctly —
  callers supply their own.
- `choropleth.py` — `build_province_grid` / `render_choropleth_html`,
  a non-overlapping 13-region tile-grid cartogram via geopandas +
  folium's `GeoDataFrame.explore()`.
- `dashboards/data_prep.py` — all data-prep logic, kept separate from
  the Streamlit script so it's unit tested directly.
- `dashboards/equity_dashboard.py` — the dashboard itself: age-standardized
  hospitalization rates by remoteness, the choropleth, stage-at-diagnosis
  and income distributions by equity subgroup, and KM curves by
  Indigenous-identity proxy. Verified with Streamlit's `AppTest` utility
  (no exceptions, all expected elements render) in addition to a manual
  `streamlit run` smoke test.
- `docs/architecture/adr-0005-equity-dashboard-choices.md`.
- 19 new tests, 97% coverage on `src/equitable_rwe`.

## Testing

```
uv run ruff check src tests dashboards data benchmark
uv run black --check src tests dashboards data benchmark
uv run mypy src
uv run pytest
```

173 passed, 97% coverage. Dashboard additionally verified via
`streamlit.testing.v1.AppTest` and a live `streamlit run` + HTTP health
check.

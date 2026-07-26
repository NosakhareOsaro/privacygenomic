# ADR-0005: Equity dashboard implementation choices

## Status

Accepted

## Context

The equity dashboard needed age-standardized rates, a GIS choropleth, and
an interactive front end, per the original spec's "Power BI or an
open-source Streamlit/Plotly Dash equivalent (prefer open-source)."

## Decision 1: Streamlit + Plotly, not Power BI

Power BI requires a license and, for `.pbix` files, the Power BI Desktop
application to open — a real barrier for anyone reviewing this portfolio
without an existing Microsoft/BI toolchain. Streamlit plus Plotly is
fully open-source, runs with `streamlit run dashboards/equity_dashboard.py`,
and needs nothing beyond this project's own dependencies.

## Decision 2: a tile-grid cartogram, not survey-accurate province boundaries

`equitable_rwe.equity.choropleth` renders provinces/territories as a
non-overlapping tile grid (one unit square per region, in its
approximate compass position) rather than actual province outline
polygons. Real province boundaries are highly irregular; approximating
them from memory as simplified shapes (e.g. bounding boxes) was
considered and rejected specifically because bounding boxes of
neighboring Canadian provinces overlap substantially in raw latitude/longitude
extent even though the provinces themselves do not overlap — a naive
bounding-box map would render as broken, visually overlapping regions.
Fetching a real boundary file from an external source was also considered
and rejected to avoid a runtime network dependency and unverified license
terms for geographic data this project would then redistribute. A
tile-grid cartogram is a real, recognized cartographic technique for
exactly this trade-off (e.g. NPR's tile-grid maps of US states): it
preserves relative geographic intuition (which province is where, roughly)
without asserting boundary precision this project cannot back up.

## Decision 3: internal (cohort-own) standard population, not an external reference standard

`equitable_rwe.equity.age_standardization.direct_standardize` requires
the caller to supply a standard population rather than shipping a default
external one (e.g. the WHO World Standard Population). This project does
not have a verified source for that standard's exact published age-group
weights, and citing a plausible-sounding but unverified number would
violate this project's own rule that no statistical claim is stated
without the code (or a checkable citation) actually backing it. The
dashboard instead uses the cohort's own overall age distribution as the
standard population — valid for comparing subgroups *within* this study,
clearly documented as not comparable to externally published
age-standardized rates.

## Consequences

- The choropleth is explicitly unsuitable for any real geographic or
  spatial analysis; every place it appears (module docstring, dashboard
  caption) says so.
- Age-standardized rates in this dashboard should not be compared to
  externally published age-standardized statistics (e.g. from CIHI or
  Statistics Canada), only to each other within this cohort — the
  dashboard's caption states this directly rather than leaving it
  implicit.

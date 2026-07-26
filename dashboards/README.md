# Equity dashboard

Open-source (Streamlit + Plotly) alternative to a Power BI dashboard —
see `docs/architecture/adr-0005-equity-dashboard-choices.md` for why.

## Run it

```bash
uv sync --extra dev
uv run streamlit run dashboards/equity_dashboard.py
```

## Structure

- `data_prep.py` — all data-preparation logic, kept separate from the UI
  script so it's unit tested directly (`tests/dashboards/`) rather than
  only exercised by clicking through the app.
- `equity_dashboard.py` — the Streamlit UI script itself. Thin: it calls
  `data_prep.py` functions and `equitable_rwe.equity`/`equitable_rwe.survival`
  APIs, and renders the results.

Every figure is computed from the synthetic, OCAP-governed cohort in
`data/synthetic/equity_cohort.parquet` — see
`docs/data-governance/ocap-and-privacy.md`.

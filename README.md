# PrivacyGenomic

**Real-World Evidence, Health Equity & Causal Outcomes Analytics Platform**

An open, reproducible pipeline that turns OMOP-CDM-formatted EHR data into
causally valid, fairness-audited clinical outcome estimates and public
health-equity intelligence — built for Canadian health systems
(CIHI/Statistics Canada-aligned) and PhysioNet-derived cohorts.

> Status: under active development. See [CHANGELOG.md](CHANGELOG.md) for
> progress and [docs/architecture](docs/architecture) for design decisions.

## What this is

This repository is the home of `equitable-rwe`, a Python library that
combines doubly-robust causal effect estimation (IPTW-adjusted Cox
regression) with integrated subgroup fairness auditing in a single API call
— returning both a causal estimate and a fairness report from one function.
It also hosts the supporting pipeline used to validate that library:

- OMOP CDM cohort extraction with Achilles-style quality control.
- A synthetic, Canadian health-equity-calibrated cohort generator (no real
  patient data — see [OCAP® and data governance](docs/data-governance/ocap-and-privacy.md)).
- Survival analysis: Kaplan-Meier, Cox PH with Schoenfeld diagnostics,
  Fine-Gray competing risks, Royston-Parmar flexible parametric models,
  landmark analysis, and RMST.
- A four-estimator causal-adjustment benchmark (naive Cox, PSM, IPTW, DoWhy
  backdoor adjustment) with E-value sensitivity analysis.
- An XGBoost 30-day readmission model with SHAP explainability and a
  fairness audit (demographic parity, equalized odds, within-group
  calibration).
- An open-source health-equity dashboard (age-standardized rates, GIS
  choropleth).

## Installation

This project targets Python 3.11–3.12 and uses [uv](https://docs.astral.sh/uv/)
for environment management.

```bash
uv sync --extra dev
```

## Documentation

Full documentation (once published) will live at ReadTheDocs; in the
meantime, see [docs/](docs/) and the module docstrings under
[src/equitable_rwe](src/equitable_rwe).

## License

MIT — see [LICENSE](LICENSE).

## Citation

See [CITATION.cff](CITATION.cff).

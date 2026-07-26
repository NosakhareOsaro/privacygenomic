# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Project scaffold: `pyproject.toml` (uv-managed, Python 3.11–3.12), MIT
  license, pre-commit tooling (ruff, black, mypy), package skeleton for
  `equitable_rwe` (`cohort`, `survival`, `causal`, `fairness`, `ml`,
  `equity`), and repository documentation structure.
- `equitable_rwe.cohort`: MIMIC-IV to OMOP CDM v5.4 ETL
  (`person`, `observation_period`, `visit_occurrence`,
  `condition_occurrence`, `drug_exposure`, `measurement`, `death`) and an
  Achilles-style data quality module (referential integrity, date
  plausibility judged against the dataset's own clock, and standard
  vocabulary coverage checks). See `docs/architecture/adr-0001-omop-etl.md`
  for scope decisions.
- `data/synthetic/generate_equity_cohort.py`: synthetic Canadian
  chronic-disease cohort generator with two health-equity gaps calibrated
  to cited public statistics (2021 Census Indigenous population share;
  CIHI 2024–2025 rural/urban hospitalization rate ratio) and several
  further gaps directionally modeled and explicitly labeled as such — see
  `data/synthetic/calibration_notes.md`.
- `docs/data-governance/ocap-and-privacy.md`: OCAP® principles and the
  rationale for using only synthetic and public-aggregate data throughout
  this project.

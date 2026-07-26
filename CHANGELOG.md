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

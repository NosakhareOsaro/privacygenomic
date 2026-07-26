# PR: OMOP CDM cohort extraction with Achilles-style QC

**Branch:** `feat/omop-cohort` → `main`

## Summary

Adds `equitable_rwe.cohort`: an ETL from MIMIC-IV `hosp` module tables into
an OMOP CDM v5.4 cohort (`person`, `observation_period`,
`visit_occurrence`, `condition_occurrence`, `drug_exposure`,
`measurement`, `death`), plus an Achilles-style data quality module.

## Why

Every downstream module in this project (survival, causal, fairness) needs
a standard cohort shape to be written once and reused across source EHRs.
OMOP CDM is that standard in observational health research; MIMIC-IV is
this project's initial source. See `docs/architecture/adr-0001-omop-etl.md`
for the three scope decisions this PR makes (synthetic test fixture in
place of downloaded demo data, deferred standard-vocabulary mapping via an
explicit `concept_map` extension point, and date-plausibility checks
computed against the dataset's own clock rather than wall-clock time — the
last of which is a genuine, non-obvious bug this PR's own test suite
caught during development: MIMIC-IV date-shifts every timestamp into the
2100s for de-identification, which a naive "future date" check would flag
as universally implausible).

## What's included

- `cohort/schema.py` — OMOP CDM table/column definitions and standard
  vocabulary concept IDs (gender, race, ethnicity, visit type).
- `cohort/omop_etl.py` — `load_mimic_tables`, `build_omop_cdm`, and one
  `transform_*` function per target table.
- `cohort/qc.py` — `run_qc`, producing a `QCReport` with per-table/person
  counts, concept frequency tables, and referential-integrity /
  date-plausibility / vocabulary-coverage issues.
- `tests/fixtures/generate_mimic_demo_fixture.py` — deterministic
  generator for a schema-matched, fully synthetic stand-in for the
  MIMIC-IV Clinical Database Demo (see ADR-0001 for why this isn't
  downloaded).
- 28 tests (unit + doctest), 95% line coverage on the new code.

## What's explicitly out of scope (by design, not oversight)

- Standard vocabulary mapping (ICD/NDC/LOINC → SNOMED/RxNorm/LOINC
  concept IDs) — requires the licensed OHDSI Athena vocabulary.
- Validation against the real MIMIC-IV Clinical Database Demo — the ETL is
  written to run against it unmodified, but that run has not been
  performed in this repository.

## Testing

```
uv run ruff check src tests
uv run black --check src tests
uv run mypy src
uv run pytest
```

All pass locally (28 passed, 95% coverage on touched files).

# ADR-0001: OMOP CDM ETL scope and test-data strategy

## Status

Accepted

## Context

The cohort extraction module (`equitable_rwe.cohort`) needs to convert a
source EHR — MIMIC-IV — into a standard OMOP CDM v5.4 shape, so that
downstream survival, causal, and fairness modules can be written once
against a standard schema. Three design questions came up immediately:

1. **What do we test against?** Full MIMIC-IV requires PhysioNet
   credentialing; the MIMIC-IV Clinical Database Demo does not, but
   downloading and committing even the openly licensed demo data into a
   public repository raises redistribution questions under PhysioNet's
   data use agreement, and makes CI depend on network access to a
   third-party host.
2. **How far do we take standard vocabulary mapping?** A "real" OMOP ETL
   maps every source code (ICD-9/10, NDC, lab `itemid`) to a standard
   concept ID (SNOMED, RxNorm, LOINC) using the OHDSI Athena vocabulary.
   That vocabulary is itself a multi-gigabyte, separately licensed
   download, not something to vendor into this repository.
3. **What counts as an "implausible" date?** MIMIC-IV is de-identified in
   part by shifting every date for a given patient into an arbitrary,
   consistent future range (e.g. admissions recorded in the 2100s). A
   naive plausibility check ("is this date after today?") is wrong for
   this kind of source data.

## Decision

1. **Test fixture is synthetic, not downloaded.** `tests/fixtures/generate_mimic_demo_fixture.py`
   generates a small (30-patient) fixture that matches the real MIMIC-IV
   demo `hosp` module schema exactly (table names, column names, dtypes)
   but contains no data derived from PhysioNet. The ETL functions in
   `omop_etl.py` are written generically against that schema, so pointing
   `load_mimic_tables` at a real (credentialed or demo) MIMIC-IV directory
   requires no code change — only the input path changes. This keeps CI
   fully offline and avoids any redistribution question, at the cost of
   the fixture not reflecting real-world coding irregularities (e.g. rare
   ICD codes, missingness patterns) the way genuine demo data would.

2. **Standard vocabulary mapping is an explicit extension point, not a
   hardcoded partial table.** `transform_condition_occurrence`,
   `transform_drug_exposure`, and `transform_measurement` each accept an
   optional `concept_map` argument. Unmapped codes are set to
   `UNMAPPED_CONCEPT_ID` (0) — the official OMOP convention — rather than
   a fabricated or guessed concept ID. `qc.check_unmapped_concepts` then
   surfaces vocabulary coverage as a first-class, visible data-quality
   signal rather than silently producing analyses on partially-mapped
   data. A production deployment would populate `concept_map` from an
   OHDSI Usagi/Athena mapping run; this project deliberately does not
   embed a hand-picked subset of concept IDs it cannot verify against the
   licensed vocabulary.

3. **Date plausibility is judged against the dataset's own clock, not
   wall-clock time.** `qc._latest_observed_date` derives a reference
   "now" from the latest date actually present in the cohort (visit and
   observation-period end dates), and `check_date_plausibility` defaults
   to that instead of `pd.Timestamp.now()`. This correctly handles
   date-shifted sources like MIMIC-IV without requiring every caller to
   know and pass the shift.

## Consequences

- Anyone with PhysioNet access can validate this ETL against the real
  MIMIC-IV Clinical Database Demo (or full MIMIC-IV) by pointing
  `load_mimic_tables` at the real directory — no code changes needed —
  but that validation has not been performed as part of this repository's
  CI, only against the synthetic fixture.
- Every `condition_concept_id` / `drug_concept_id` / `measurement_concept_id`
  produced by this ETL is `0` (unmapped) unless the caller supplies a
  `concept_map`, which `run_qc` will flag. This is intentional: silence on
  this point would be worse than an explicit, visible warning.
- Achilles-style QC here is a lightweight, pandas-native reimplementation
  of a small slice of what the R `Achilles` package computes, not a
  drop-in replacement for it.

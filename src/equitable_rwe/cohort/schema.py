"""Minimal OMOP Common Data Model (CDM) v5.4 table definitions.

This module defines the subset of the OMOP CDM used by this project's ETL:
``person``, ``observation_period``, ``visit_occurrence``,
``condition_occurrence``, ``drug_exposure``, ``measurement``, and ``death``.
Column names follow the official OHDSI CDM v5.4 specification
(https://ohdsi.github.io/CommonDataModel/cdm54.html) so downstream code
(Achilles-style QC, survival/causal modules) can be written once against a
standard shape regardless of the source EHR.

Standard vocabulary concept IDs used for demographic fields (gender, race,
ethnicity, visit type) are fixed values from the OHDSI Standardized
Vocabularies and are stable across CDM instances. Clinical concept mapping
(ICD/NDC/LOINC source codes to standard SNOMED/RxNorm/LOINC concept IDs)
requires the OHDSI Athena vocabulary and is deliberately out of scope here;
see :mod:`equitable_rwe.cohort.omop_etl` and
``docs/architecture/adr-0001-omop-etl.md`` for the rationale.
"""

from __future__ import annotations

# OMOP standard vocabulary concept IDs (stable, from OHDSI Athena).
GENDER_CONCEPT_MALE = 8507
GENDER_CONCEPT_FEMALE = 8532
GENDER_CONCEPT_UNKNOWN = 0

RACE_CONCEPT_WHITE = 8527
RACE_CONCEPT_BLACK = 8516
RACE_CONCEPT_ASIAN = 8515
RACE_CONCEPT_UNKNOWN = 0

ETHNICITY_CONCEPT_HISPANIC = 38003563
ETHNICITY_CONCEPT_NOT_HISPANIC = 38003564

VISIT_CONCEPT_INPATIENT = 9201
DEATH_TYPE_CONCEPT_EHR = 32817
DRUG_TYPE_CONCEPT_PRESCRIPTION = 32838
CONDITION_TYPE_CONCEPT_EHR = 32020
MEASUREMENT_TYPE_CONCEPT_LAB = 32856

#: Value used for any concept that has not been mapped to the standard
#: vocabulary, per the official OMOP CDM convention.
UNMAPPED_CONCEPT_ID = 0

PERSON_COLUMNS = [
    "person_id",
    "gender_concept_id",
    "year_of_birth",
    "race_concept_id",
    "ethnicity_concept_id",
    "person_source_value",
    "gender_source_value",
    "race_source_value",
]

OBSERVATION_PERIOD_COLUMNS = [
    "observation_period_id",
    "person_id",
    "observation_period_start_date",
    "observation_period_end_date",
    "period_type_concept_id",
]

VISIT_OCCURRENCE_COLUMNS = [
    "visit_occurrence_id",
    "person_id",
    "visit_concept_id",
    "visit_start_date",
    "visit_end_date",
    "visit_source_value",
]

CONDITION_OCCURRENCE_COLUMNS = [
    "condition_occurrence_id",
    "person_id",
    "visit_occurrence_id",
    "condition_concept_id",
    "condition_start_date",
    "condition_type_concept_id",
    "condition_source_value",
    "condition_status_source_value",
]

DRUG_EXPOSURE_COLUMNS = [
    "drug_exposure_id",
    "person_id",
    "visit_occurrence_id",
    "drug_concept_id",
    "drug_exposure_start_date",
    "drug_exposure_end_date",
    "drug_type_concept_id",
    "drug_source_value",
]

MEASUREMENT_COLUMNS = [
    "measurement_id",
    "person_id",
    "visit_occurrence_id",
    "measurement_concept_id",
    "measurement_date",
    "measurement_type_concept_id",
    "value_as_number",
    "unit_source_value",
    "measurement_source_value",
]

DEATH_COLUMNS = [
    "person_id",
    "death_date",
    "death_type_concept_id",
]

#: Required (non-nullable, referentially significant) columns per table,
#: used by :mod:`equitable_rwe.cohort.qc`.
REQUIRED_COLUMNS = {
    "person": ["person_id", "gender_concept_id", "year_of_birth"],
    "observation_period": [
        "person_id",
        "observation_period_start_date",
        "observation_period_end_date",
    ],
    "visit_occurrence": ["visit_occurrence_id", "person_id", "visit_start_date", "visit_end_date"],
    "condition_occurrence": ["person_id", "condition_start_date"],
    "drug_exposure": ["person_id", "drug_exposure_start_date"],
    "measurement": ["person_id", "measurement_date"],
    "death": ["person_id", "death_date"],
}

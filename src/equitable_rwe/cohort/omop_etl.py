"""ETL from MIMIC-IV ``hosp`` module tables into an OMOP CDM v5.4 cohort.

The transform functions in this module are written against the real
MIMIC-IV table and column names (``patients``, ``admissions``,
``diagnoses_icd``, ``prescriptions``, ``labevents``, ``d_labitems``) as
published for both the credentialed MIMIC-IV database and the openly
licensed MIMIC-IV Clinical Database Demo. They will run unmodified against
either — point :func:`load_mimic_tables` at a directory containing the real
demo CSVs (or their gzip-compressed originals) once you have them.

Clinical source codes (ICD-9/10 diagnosis codes, NDC drug codes, lab
``itemid``/LOINC codes) are carried through as
``*_source_value`` but are **not** mapped to standard OMOP concept IDs,
since that mapping requires the OHDSI Athena vocabulary, which is a
separate licensed download outside this project's scope. Mapped
``*_concept_id`` columns are set to
:data:`equitable_rwe.cohort.schema.UNMAPPED_CONCEPT_ID` unless a concept is
supplied via ``concept_map``. See ``docs/architecture/adr-0001-omop-etl.md``.

Examples
--------
>>> from pathlib import Path
>>> tables = load_mimic_tables(Path("tests/fixtures/mimic_demo"))
>>> cdm = build_omop_cdm(tables)
>>> "person" in cdm and "condition_occurrence" in cdm
True
>>> bool(cdm["person"]["gender_concept_id"].isin([8507, 8532, 0]).all())
True
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from equitable_rwe.cohort import schema

_MIMIC_TABLE_FILES = {
    "patients": "patients.csv",
    "admissions": "admissions.csv",
    "diagnoses_icd": "diagnoses_icd.csv",
    "prescriptions": "prescriptions.csv",
    "labevents": "labevents.csv",
    "d_labitems": "d_labitems.csv",
}

_DATE_COLUMNS = {
    "admissions": ["admittime", "dischtime", "deathtime"],
    "prescriptions": ["starttime", "stoptime"],
    "labevents": ["charttime"],
}

#: Medical source codes must be read as strings even when every code in a
#: given file happens to be numeric-looking (e.g. ICD-9 "4019"), otherwise
#: pandas silently infers an integer dtype and drops meaning like leading
#: zeros, breaking any later exact-match concept mapping.
_DTYPE_OVERRIDES: dict[str, dict[str, type]] = {
    "diagnoses_icd": {"icd_code": str},
}


def load_mimic_tables(source_dir: str | Path) -> dict[str, pd.DataFrame]:
    """Load raw MIMIC-IV ``hosp`` module CSVs from a directory.

    Parameters
    ----------
    source_dir : str or Path
        Directory containing the MIMIC-IV (or MIMIC-IV demo) ``hosp``
        module CSVs. Both plain ``.csv`` and ``.csv.gz`` files are
        supported.

    Returns
    -------
    dict[str, pandas.DataFrame]
        Raw tables keyed by MIMIC table name, with date columns parsed.

    Examples
    --------
    >>> from pathlib import Path
    >>> tables = load_mimic_tables(Path("tests/fixtures/mimic_demo"))
    >>> "patients" in tables
    True
    """
    source_dir = Path(source_dir)
    tables: dict[str, pd.DataFrame] = {}
    for name, filename in _MIMIC_TABLE_FILES.items():
        path = source_dir / filename
        if not path.exists():
            gz_path = source_dir / f"{filename}.gz"
            if gz_path.exists():
                path = gz_path
            else:
                raise FileNotFoundError(
                    f"Expected MIMIC-IV table '{filename}' (or .gz) under {source_dir}"
                )
        tables[name] = pd.read_csv(
            path,
            parse_dates=_DATE_COLUMNS.get(name),
            dtype=_DTYPE_OVERRIDES.get(name),
        )
    return tables


def transform_person(patients: pd.DataFrame) -> pd.DataFrame:
    """Map MIMIC-IV ``patients`` to the OMOP ``person`` table.

    Parameters
    ----------
    patients : pandas.DataFrame
        Raw MIMIC-IV ``patients`` table with columns ``subject_id``,
        ``gender``, ``anchor_age``, ``anchor_year``.

    Returns
    -------
    pandas.DataFrame
        OMOP ``person`` table.

    Examples
    --------
    >>> import pandas as pd
    >>> raw = pd.DataFrame(
    ...     {"subject_id": [1], "gender": ["F"], "anchor_age": [50], "anchor_year": [2150]}
    ... )
    >>> int(transform_person(raw)["gender_concept_id"].iloc[0])
    8532
    """
    gender_map = {
        "M": schema.GENDER_CONCEPT_MALE,
        "F": schema.GENDER_CONCEPT_FEMALE,
    }
    out = pd.DataFrame(
        {
            "person_id": patients["subject_id"],
            "gender_concept_id": patients["gender"]
            .map(gender_map)
            .fillna(schema.GENDER_CONCEPT_UNKNOWN)
            .astype(int),
            "year_of_birth": patients["anchor_year"] - patients["anchor_age"],
            "race_concept_id": schema.RACE_CONCEPT_UNKNOWN,
            "ethnicity_concept_id": schema.ETHNICITY_CONCEPT_NOT_HISPANIC,
            "person_source_value": patients["subject_id"].astype(str),
            "gender_source_value": patients["gender"],
            "race_source_value": pd.NA,
        }
    )
    return out[schema.PERSON_COLUMNS]


def transform_visit_occurrence(admissions: pd.DataFrame) -> pd.DataFrame:
    """Map MIMIC-IV ``admissions`` to the OMOP ``visit_occurrence`` table."""
    out = pd.DataFrame(
        {
            "visit_occurrence_id": admissions["hadm_id"],
            "person_id": admissions["subject_id"],
            "visit_concept_id": schema.VISIT_CONCEPT_INPATIENT,
            "visit_start_date": admissions["admittime"].dt.normalize(),
            "visit_end_date": admissions["dischtime"].dt.normalize(),
            "visit_source_value": admissions.get("admission_type", pd.NA),
        }
    )
    return out[schema.VISIT_OCCURRENCE_COLUMNS]


def transform_observation_period(visit_occurrence: pd.DataFrame) -> pd.DataFrame:
    """Derive one OMOP ``observation_period`` per person spanning their visits.

    Each person's observation period is taken as the span from their
    earliest visit start to their latest visit end, which is the standard
    approximation used when a payer/EHR-coverage table is unavailable.
    """
    spans = visit_occurrence.groupby("person_id").agg(
        observation_period_start_date=("visit_start_date", "min"),
        observation_period_end_date=("visit_end_date", "max"),
    )
    out = spans.reset_index()
    out.insert(0, "observation_period_id", range(1, len(out) + 1))
    out["period_type_concept_id"] = schema.CONDITION_TYPE_CONCEPT_EHR
    return out[schema.OBSERVATION_PERIOD_COLUMNS]


def transform_condition_occurrence(
    diagnoses_icd: pd.DataFrame,
    admissions: pd.DataFrame,
    concept_map: dict[str, int] | None = None,
) -> pd.DataFrame:
    """Map MIMIC-IV ``diagnoses_icd`` to the OMOP ``condition_occurrence`` table.

    Parameters
    ----------
    diagnoses_icd : pandas.DataFrame
        Raw MIMIC-IV ``diagnoses_icd`` table.
    admissions : pandas.DataFrame
        Raw MIMIC-IV ``admissions`` table, used to date-stamp each
        diagnosis at the admission's start date (MIMIC records diagnoses
        per-stay without a separate onset date).
    concept_map : dict[str, int], optional
        Mapping from ``icd_code`` source values to standard OMOP condition
        ``concept_id``. Codes absent from the map are set to
        :data:`equitable_rwe.cohort.schema.UNMAPPED_CONCEPT_ID`.
    """
    concept_map = concept_map or {}
    merged = diagnoses_icd.merge(
        admissions[["subject_id", "hadm_id", "admittime"]],
        on=["subject_id", "hadm_id"],
        how="left",
    )
    out = pd.DataFrame(
        {
            "person_id": merged["subject_id"],
            "visit_occurrence_id": merged["hadm_id"],
            "condition_concept_id": merged["icd_code"]
            .map(concept_map)
            .fillna(schema.UNMAPPED_CONCEPT_ID)
            .astype(int),
            "condition_start_date": merged["admittime"].dt.normalize(),
            "condition_type_concept_id": schema.CONDITION_TYPE_CONCEPT_EHR,
            "condition_source_value": merged["icd_code"],
            "condition_status_source_value": "icd" + merged["icd_version"].astype(str),
        }
    )
    out.insert(0, "condition_occurrence_id", range(1, len(out) + 1))
    return out[schema.CONDITION_OCCURRENCE_COLUMNS]


def transform_drug_exposure(
    prescriptions: pd.DataFrame, concept_map: dict[str, int] | None = None
) -> pd.DataFrame:
    """Map MIMIC-IV ``prescriptions`` to the OMOP ``drug_exposure`` table."""
    concept_map = concept_map or {}
    out = pd.DataFrame(
        {
            "person_id": prescriptions["subject_id"],
            "visit_occurrence_id": prescriptions["hadm_id"],
            "drug_concept_id": prescriptions["drug"]
            .map(concept_map)
            .fillna(schema.UNMAPPED_CONCEPT_ID)
            .astype(int),
            "drug_exposure_start_date": prescriptions["starttime"].dt.normalize(),
            "drug_exposure_end_date": prescriptions["stoptime"].dt.normalize(),
            "drug_type_concept_id": schema.DRUG_TYPE_CONCEPT_PRESCRIPTION,
            "drug_source_value": prescriptions["drug"],
        }
    )
    out.insert(0, "drug_exposure_id", range(1, len(out) + 1))
    return out[schema.DRUG_EXPOSURE_COLUMNS]


def transform_measurement(
    labevents: pd.DataFrame,
    d_labitems: pd.DataFrame,
    concept_map: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Map MIMIC-IV ``labevents`` to the OMOP ``measurement`` table."""
    concept_map = concept_map or {}
    merged = labevents.merge(d_labitems[["itemid", "label"]], on="itemid", how="left")
    out = pd.DataFrame(
        {
            "person_id": merged["subject_id"],
            "visit_occurrence_id": merged["hadm_id"],
            "measurement_concept_id": merged["itemid"]
            .map(concept_map)
            .fillna(schema.UNMAPPED_CONCEPT_ID)
            .astype(int),
            "measurement_date": merged["charttime"].dt.normalize(),
            "measurement_type_concept_id": schema.MEASUREMENT_TYPE_CONCEPT_LAB,
            "value_as_number": pd.to_numeric(merged["valuenum"], errors="coerce"),
            "unit_source_value": merged.get("valueuom", pd.NA),
            "measurement_source_value": merged["label"],
        }
    )
    out.insert(0, "measurement_id", range(1, len(out) + 1))
    return out[schema.MEASUREMENT_COLUMNS]


def transform_death(patients: pd.DataFrame) -> pd.DataFrame:
    """Map MIMIC-IV ``patients.dod`` to the OMOP ``death`` table.

    Only rows with a non-null date of death are included, matching the
    OMOP convention that ``death`` has at most one row per deceased person.
    """
    deceased = patients.dropna(subset=["dod"]).copy()
    out = pd.DataFrame(
        {
            "person_id": deceased["subject_id"],
            "death_date": pd.to_datetime(deceased["dod"]).dt.normalize(),
            "death_type_concept_id": schema.DEATH_TYPE_CONCEPT_EHR,
        }
    )
    return out[schema.DEATH_COLUMNS]


def build_omop_cdm(
    raw_tables: dict[str, pd.DataFrame],
    condition_concept_map: dict[str, int] | None = None,
    drug_concept_map: dict[str, int] | None = None,
    measurement_concept_map: dict[int, int] | None = None,
) -> dict[str, pd.DataFrame]:
    """Run the full MIMIC-IV to OMOP CDM transform.

    Parameters
    ----------
    raw_tables : dict[str, pandas.DataFrame]
        Output of :func:`load_mimic_tables`.
    condition_concept_map, drug_concept_map, measurement_concept_map : dict, optional
        Optional source-value to standard-concept-ID mappings (see
        :func:`transform_condition_occurrence`,
        :func:`transform_drug_exposure`, :func:`transform_measurement`).

    Returns
    -------
    dict[str, pandas.DataFrame]
        OMOP CDM tables keyed by table name: ``person``,
        ``observation_period``, ``visit_occurrence``,
        ``condition_occurrence``, ``drug_exposure``, ``measurement``,
        ``death``.

    Examples
    --------
    >>> from pathlib import Path
    >>> tables = load_mimic_tables(Path("tests/fixtures/mimic_demo"))
    >>> cdm = build_omop_cdm(tables)
    >>> len(cdm["person"]) == len(tables["patients"])
    True
    """
    person = transform_person(raw_tables["patients"])
    visit_occurrence = transform_visit_occurrence(raw_tables["admissions"])
    observation_period = transform_observation_period(visit_occurrence)
    condition_occurrence = transform_condition_occurrence(
        raw_tables["diagnoses_icd"], raw_tables["admissions"], condition_concept_map
    )
    drug_exposure = transform_drug_exposure(raw_tables["prescriptions"], drug_concept_map)
    measurement = transform_measurement(
        raw_tables["labevents"], raw_tables["d_labitems"], measurement_concept_map
    )
    death = transform_death(raw_tables["patients"])

    return {
        "person": person,
        "observation_period": observation_period,
        "visit_occurrence": visit_occurrence,
        "condition_occurrence": condition_occurrence,
        "drug_exposure": drug_exposure,
        "measurement": measurement,
        "death": death,
    }

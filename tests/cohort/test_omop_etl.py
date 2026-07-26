import pandas as pd
import pytest

from equitable_rwe.cohort import schema
from equitable_rwe.cohort.omop_etl import load_mimic_tables


def test_load_mimic_tables_returns_expected_tables(raw_mimic_tables):
    assert set(raw_mimic_tables) == {
        "patients",
        "admissions",
        "diagnoses_icd",
        "prescriptions",
        "labevents",
        "d_labitems",
    }
    assert len(raw_mimic_tables["patients"]) == 30


def test_load_mimic_tables_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_mimic_tables(tmp_path)


def test_person_row_count_matches_source(omop_cdm, raw_mimic_tables):
    assert len(omop_cdm["person"]) == len(raw_mimic_tables["patients"])


def test_person_gender_concept_ids_are_valid(omop_cdm):
    valid = {schema.GENDER_CONCEPT_MALE, schema.GENDER_CONCEPT_FEMALE}
    assert set(omop_cdm["person"]["gender_concept_id"]).issubset(valid)


def test_person_year_of_birth_matches_anchor_arithmetic(raw_mimic_tables, omop_cdm):
    patients = raw_mimic_tables["patients"].set_index("subject_id")
    person = omop_cdm["person"].set_index("person_id")
    expected = patients["anchor_year"] - patients["anchor_age"]
    pd.testing.assert_series_equal(person["year_of_birth"], expected, check_names=False)


def test_visit_occurrence_dates_never_end_before_start(omop_cdm):
    vo = omop_cdm["visit_occurrence"]
    assert (vo["visit_end_date"] >= vo["visit_start_date"]).all()


def test_observation_period_spans_all_visits(omop_cdm):
    vo = omop_cdm["visit_occurrence"]
    op = omop_cdm["observation_period"].set_index("person_id")
    per_person_min = vo.groupby("person_id")["visit_start_date"].min()
    per_person_max = vo.groupby("person_id")["visit_end_date"].max()
    assert (op.loc[per_person_min.index, "observation_period_start_date"] == per_person_min).all()
    assert (op.loc[per_person_max.index, "observation_period_end_date"] == per_person_max).all()


def test_condition_occurrence_row_count_matches_source(omop_cdm, raw_mimic_tables):
    assert len(omop_cdm["condition_occurrence"]) == len(raw_mimic_tables["diagnoses_icd"])


def test_condition_occurrence_concept_map_is_applied(raw_mimic_tables):
    from equitable_rwe.cohort.omop_etl import build_omop_cdm

    concept_map = {"4019": 999999}
    cdm = build_omop_cdm(raw_mimic_tables, condition_concept_map=concept_map)
    mapped = cdm["condition_occurrence"]
    hits = mapped.loc[mapped["condition_source_value"] == "4019", "condition_concept_id"]
    assert (hits == 999999).all()
    assert len(hits) > 0


def test_drug_exposure_row_count_matches_source(omop_cdm, raw_mimic_tables):
    assert len(omop_cdm["drug_exposure"]) == len(raw_mimic_tables["prescriptions"])


def test_measurement_row_count_matches_source(omop_cdm, raw_mimic_tables):
    assert len(omop_cdm["measurement"]) == len(raw_mimic_tables["labevents"])


def test_death_only_includes_deceased_patients(omop_cdm, raw_mimic_tables):
    n_deceased = raw_mimic_tables["patients"]["dod"].notna().sum()
    assert len(omop_cdm["death"]) == n_deceased
    assert omop_cdm["death"]["death_date"].notna().all()

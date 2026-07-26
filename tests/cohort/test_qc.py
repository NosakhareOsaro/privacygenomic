import pandas as pd

from equitable_rwe.cohort.qc import (
    QCIssue,
    check_date_plausibility,
    check_referential_integrity,
    check_unmapped_concepts,
    run_qc,
    summarize_table_counts,
)


def test_run_qc_passes_on_clean_cdm(omop_cdm):
    report = run_qc(omop_cdm)
    assert report.passed
    assert report.issues_frame().empty or all(
        issue.severity != "critical" for issue in report.issues
    )


def test_table_counts_cover_every_table(omop_cdm):
    counts = summarize_table_counts(omop_cdm)
    assert set(counts.index) == set(omop_cdm)
    for table, df in omop_cdm.items():
        assert counts.loc[table, "n_rows"] == len(df)


def test_referential_integrity_flags_orphan_visit(omop_cdm):
    broken = {**omop_cdm}
    condition_occurrence = broken["condition_occurrence"].copy()
    condition_occurrence.loc[0, "visit_occurrence_id"] = -1
    broken["condition_occurrence"] = condition_occurrence

    issues = check_referential_integrity(broken)
    assert any(
        issue.table == "condition_occurrence" and issue.severity == "critical" for issue in issues
    )


def test_referential_integrity_clean_on_valid_cdm(omop_cdm):
    assert check_referential_integrity(omop_cdm) == []


def test_date_plausibility_flags_visit_ending_before_start(omop_cdm):
    broken = {**omop_cdm}
    visit_occurrence = broken["visit_occurrence"].copy()
    visit_occurrence.loc[0, "visit_end_date"] = visit_occurrence.loc[
        0, "visit_start_date"
    ] - pd.Timedelta(days=1)
    broken["visit_occurrence"] = visit_occurrence

    issues = check_date_plausibility(broken)
    assert any(issue.check == "visit_end_before_start" for issue in issues)


def test_date_plausibility_flags_future_birth_year(omop_cdm):
    from equitable_rwe.cohort.qc import _latest_observed_date

    broken = {**omop_cdm}
    person = broken["person"].copy()
    person.loc[0, "year_of_birth"] = _latest_observed_date(omop_cdm).year + 5
    broken["person"] = person

    issues = check_date_plausibility(broken)
    assert any(issue.check == "birth_year_in_future" for issue in issues)


def test_date_plausibility_uses_dataset_clock_not_wall_clock(omop_cdm):
    # MIMIC-style de-identified data is date-shifted into the future, so a
    # birth year far beyond real "now" but consistent with the shifted
    # dataset must not be treated as implausible.
    issues = check_date_plausibility(omop_cdm)
    assert not any(issue.check == "birth_year_in_future" for issue in issues)


def test_unmapped_concepts_warns_when_everything_unmapped(omop_cdm):
    issues = check_unmapped_concepts(omop_cdm, warn_threshold=0.5)
    assert any(
        issue.table == "condition_occurrence" and issue.check == "high_unmapped_concept_fraction"
        for issue in issues
    )


def test_unmapped_concepts_silent_once_mapped(raw_mimic_tables):
    from equitable_rwe.cohort.omop_etl import build_omop_cdm

    concept_map = {"4019": 1, "4280": 2, "25000": 3, "5849": 4, "41401": 5, "2724": 6, "496": 7}
    cdm = build_omop_cdm(raw_mimic_tables, condition_concept_map=concept_map)
    issues = check_unmapped_concepts(cdm, warn_threshold=0.5)
    assert not any(issue.table == "condition_occurrence" for issue in issues)


def test_qc_issue_is_a_plain_dataclass():
    issue = QCIssue("check", "table", "warning", 1, "detail")
    assert issue.severity == "warning"

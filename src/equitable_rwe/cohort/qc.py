"""Achilles-style data quality checks for an OMOP CDM cohort.

`OHDSI Achilles <https://github.com/OHDSI/Achilles>`_ characterizes a CDM
instance by computing per-table row/person counts, concept-level frequency
distributions, and a set of plausibility rules (e.g. visits ending before
they start, birth years in the future, orphan foreign keys). This module
implements a lightweight, pure-pandas equivalent of that pattern: a summary
table analogous to Achilles' ``achilles_results``, plus a rule-based issue
list analogous to its data-quality dashboard, without requiring a database
or the R Achilles package.

Examples
--------
>>> from pathlib import Path
>>> from equitable_rwe.cohort.omop_etl import load_mimic_tables, build_omop_cdm
>>> cdm = build_omop_cdm(load_mimic_tables(Path("tests/fixtures/mimic_demo")))
>>> report = run_qc(cdm)
>>> report.passed
True
>>> int(report.table_counts.loc["person", "n_rows"]) > 0
True
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from equitable_rwe.cohort import schema

#: The clinically meaningful standard-vocabulary concept column per table
#: (excludes demographic and ``*_type_concept_id`` bookkeeping columns).
_PRIMARY_CONCEPT_COLUMN = {
    "visit_occurrence": "visit_concept_id",
    "condition_occurrence": "condition_concept_id",
    "drug_exposure": "drug_concept_id",
    "measurement": "measurement_concept_id",
}

#: (child_table, foreign_key_column, parent_table, parent_key_column)
_REFERENTIAL_LINKS = [
    ("observation_period", "person_id", "person", "person_id"),
    ("visit_occurrence", "person_id", "person", "person_id"),
    ("condition_occurrence", "person_id", "person", "person_id"),
    ("condition_occurrence", "visit_occurrence_id", "visit_occurrence", "visit_occurrence_id"),
    ("drug_exposure", "person_id", "person", "person_id"),
    ("drug_exposure", "visit_occurrence_id", "visit_occurrence", "visit_occurrence_id"),
    ("measurement", "person_id", "person", "person_id"),
    ("measurement", "visit_occurrence_id", "visit_occurrence", "visit_occurrence_id"),
    ("death", "person_id", "person", "person_id"),
]


@dataclass
class QCIssue:
    """A single data quality finding."""

    check: str
    table: str
    severity: str  # "critical" or "warning"
    n_affected: int
    detail: str


@dataclass
class QCReport:
    """Result of :func:`run_qc`: table-level counts plus any issues found."""

    table_counts: pd.DataFrame
    concept_frequencies: dict[str, pd.DataFrame]
    issues: list[QCIssue] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """``True`` if no ``critical`` severity issues were found."""
        return not any(issue.severity == "critical" for issue in self.issues)

    def issues_frame(self) -> pd.DataFrame:
        """Issues as a flat :class:`pandas.DataFrame`, for reporting."""
        if not self.issues:
            return pd.DataFrame(columns=["check", "table", "severity", "n_affected", "detail"])
        return pd.DataFrame([issue.__dict__ for issue in self.issues])


def summarize_table_counts(cdm: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Row and distinct-person counts per CDM table (Achilles analysis 1-2 equivalent)."""
    rows = []
    for table_name, df in cdm.items():
        n_persons = df["person_id"].nunique() if "person_id" in df.columns else pd.NA
        rows.append({"table": table_name, "n_rows": len(df), "n_persons": n_persons})
    return pd.DataFrame(rows).set_index("table")


def summarize_concept_frequencies(cdm: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    """Frequency of each ``*_concept_id`` / ``*_source_value`` per table.

    Mirrors Achilles' per-concept prevalence analyses, reporting how many
    records and distinct persons are associated with each concept.
    """
    frequencies: dict[str, pd.DataFrame] = {}
    for table_name, col in _PRIMARY_CONCEPT_COLUMN.items():
        df = cdm.get(table_name)
        if df is None or col not in df.columns:
            continue
        grouped = (
            df.groupby(col)
            .agg(n_records=(col, "size"), n_persons=("person_id", "nunique"))
            .sort_values("n_records", ascending=False)
        )
        frequencies[f"{table_name}.{col}"] = grouped
    return frequencies


def check_referential_integrity(cdm: dict[str, pd.DataFrame]) -> list[QCIssue]:
    """Flag foreign keys in child tables with no matching parent row."""
    issues: list[QCIssue] = []
    for child, fk, parent, pk in _REFERENTIAL_LINKS:
        if child not in cdm or parent not in cdm:
            continue
        child_df, parent_df = cdm[child], cdm[parent]
        if fk not in child_df.columns or pk not in parent_df.columns:
            continue
        orphans = ~child_df[fk].isin(parent_df[pk])
        n_orphans = int(orphans.sum())
        if n_orphans:
            issues.append(
                QCIssue(
                    check="referential_integrity",
                    table=child,
                    severity="critical",
                    n_affected=n_orphans,
                    detail=(
                        f"{n_orphans} rows in '{child}' reference a missing "
                        f"'{parent}.{pk}' via '{fk}'"
                    ),
                )
            )
    return issues


def _latest_observed_date(cdm: dict[str, pd.DataFrame]) -> pd.Timestamp:
    """Latest date seen anywhere in the cohort, used as the plausibility clock.

    EHR sources that are de-identified by date-shifting (MIMIC-IV among
    them) deliberately move all dates into an arbitrary future range, so
    "is this birth year in the future" must be judged relative to the
    dataset's own latest observed date, not wall-clock time.
    """
    candidates = []
    if "visit_occurrence" in cdm and len(cdm["visit_occurrence"]):
        candidates.append(cdm["visit_occurrence"]["visit_end_date"].max())
    if "observation_period" in cdm and len(cdm["observation_period"]):
        candidates.append(cdm["observation_period"]["observation_period_end_date"].max())
    if not candidates:
        return pd.Timestamp.now()
    return max(candidates)


def check_date_plausibility(
    cdm: dict[str, pd.DataFrame], as_of: pd.Timestamp | None = None
) -> list[QCIssue]:
    """Flag implausible dates: visits ending before starting, birth years in the future.

    Parameters
    ----------
    cdm : dict[str, pandas.DataFrame]
        OMOP CDM tables.
    as_of : pandas.Timestamp, optional
        Reference "current" date used for future/age plausibility checks.
        Defaults to the latest date observed anywhere in the cohort (see
        :func:`_latest_observed_date`) rather than wall-clock time, since
        de-identified EHR sources are commonly date-shifted into the
        future.
    """
    issues: list[QCIssue] = []
    as_of = as_of or _latest_observed_date(cdm)

    if "visit_occurrence" in cdm:
        vo = cdm["visit_occurrence"]
        bad = vo["visit_end_date"] < vo["visit_start_date"]
        if bad.any():
            issues.append(
                QCIssue(
                    "visit_end_before_start",
                    "visit_occurrence",
                    "critical",
                    int(bad.sum()),
                    f"{int(bad.sum())} visits end before they start",
                )
            )

    if "person" in cdm:
        person = cdm["person"]
        future_birth = person["year_of_birth"] > as_of.year
        if future_birth.any():
            issues.append(
                QCIssue(
                    "birth_year_in_future",
                    "person",
                    "critical",
                    int(future_birth.sum()),
                    f"{int(future_birth.sum())} persons have a year_of_birth after {as_of.year}",
                )
            )
        implausible_age = (as_of.year - person["year_of_birth"]) > 120
        if implausible_age.any():
            issues.append(
                QCIssue(
                    "implausible_age",
                    "person",
                    "warning",
                    int(implausible_age.sum()),
                    f"{int(implausible_age.sum())} persons would be over 120 years old",
                )
            )

    if "condition_occurrence" in cdm and "observation_period" in cdm:
        merged = cdm["condition_occurrence"].merge(
            cdm["observation_period"][
                ["person_id", "observation_period_start_date", "observation_period_end_date"]
            ],
            on="person_id",
            how="left",
        )
        outside = (merged["condition_start_date"] < merged["observation_period_start_date"]) | (
            merged["condition_start_date"] > merged["observation_period_end_date"]
        )
        if outside.any():
            issues.append(
                QCIssue(
                    "condition_outside_observation_period",
                    "condition_occurrence",
                    "warning",
                    int(outside.sum()),
                    (
                        f"{int(outside.sum())} conditions start outside the "
                        "person's observation period"
                    ),
                )
            )

    return issues


def check_unmapped_concepts(
    cdm: dict[str, pd.DataFrame], warn_threshold: float = 0.5
) -> list[QCIssue]:
    """Warn when a high fraction of a concept column is unmapped (concept_id 0)."""
    issues: list[QCIssue] = []
    for table_name, col in _PRIMARY_CONCEPT_COLUMN.items():
        df = cdm.get(table_name)
        if df is None or col not in df.columns or len(df) == 0:
            continue
        unmapped = df[col] == schema.UNMAPPED_CONCEPT_ID
        frac_unmapped = unmapped.mean()
        if frac_unmapped >= warn_threshold:
            issues.append(
                QCIssue(
                    "high_unmapped_concept_fraction",
                    table_name,
                    "warning",
                    int(unmapped.sum()),
                    f"{frac_unmapped:.0%} of '{table_name}.{col}' is unmapped "
                    "(concept_id 0) — supply a concept_map to build_omop_cdm",
                )
            )
    return issues


def run_qc(cdm: dict[str, pd.DataFrame]) -> QCReport:
    """Run the full Achilles-style QC suite against an OMOP CDM cohort.

    Parameters
    ----------
    cdm : dict[str, pandas.DataFrame]
        Output of :func:`equitable_rwe.cohort.omop_etl.build_omop_cdm`.

    Returns
    -------
    QCReport
        Table counts, concept frequencies, and any referential-integrity,
        date-plausibility, or vocabulary-coverage issues found.
    """
    issues = [
        *check_referential_integrity(cdm),
        *check_date_plausibility(cdm),
        *check_unmapped_concepts(cdm),
    ]
    return QCReport(
        table_counts=summarize_table_counts(cdm),
        concept_frequencies=summarize_concept_frequencies(cdm),
        issues=issues,
    )

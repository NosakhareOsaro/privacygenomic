"""OMOP CDM cohort extraction and Achilles-style quality control.

See :mod:`equitable_rwe.cohort.omop_etl` for the MIMIC-IV-to-OMOP
transform and :mod:`equitable_rwe.cohort.qc` for data quality checks.
"""

from equitable_rwe.cohort.omop_etl import build_omop_cdm, load_mimic_tables
from equitable_rwe.cohort.qc import QCReport, run_qc

__all__ = ["build_omop_cdm", "load_mimic_tables", "QCReport", "run_qc"]

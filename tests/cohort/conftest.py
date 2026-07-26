from pathlib import Path

import pytest

from equitable_rwe.cohort.omop_etl import build_omop_cdm, load_mimic_tables

FIXTURE_DIR = Path(__file__).parents[1] / "fixtures" / "mimic_demo"


@pytest.fixture(scope="module")
def raw_mimic_tables():
    return load_mimic_tables(FIXTURE_DIR)


@pytest.fixture(scope="module")
def omop_cdm(raw_mimic_tables):
    return build_omop_cdm(raw_mimic_tables)

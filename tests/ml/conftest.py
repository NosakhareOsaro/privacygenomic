import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[2] / "data" / "synthetic" / "generate_readmission_cohort.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_readmission_cohort", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def readmission_module():
    return _load_module()


@pytest.fixture(scope="session")
def readmission_cohort(readmission_module):
    return readmission_module.generate_readmission_cohort(n_patients=2000, seed=1)

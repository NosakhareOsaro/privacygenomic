import importlib.util
import sys
from pathlib import Path

import pytest

_DASHBOARDS_DIR = Path(__file__).parents[2] / "dashboards"


def _load_module():
    spec = importlib.util.spec_from_file_location("data_prep", _DASHBOARDS_DIR / "data_prep.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules["data_prep"] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def data_prep():
    return _load_module()


@pytest.fixture(scope="session")
def equity_cohort(data_prep):
    return data_prep.load_cohort()

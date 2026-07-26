"""Loads generate_equity_cohort.py directly, since data/synthetic is a
standalone script location, not part of the installable equitable_rwe
package (see data/synthetic/README.md)."""

import importlib.util
from pathlib import Path

import pytest

_MODULE_PATH = Path(__file__).parents[2] / "data" / "synthetic" / "generate_equity_cohort.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("generate_equity_cohort", _MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def equity_cohort_module():
    return _load_module()

"""Smoke test verifying the package imports and exposes a version."""

import equitable_rwe


def test_version_is_set() -> None:
    assert equitable_rwe.__version__ == "0.1.0"

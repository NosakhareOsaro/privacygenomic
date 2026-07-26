# PR: coverage floor, wider CI lint scope, pre-commit/CI drift fixes

**Branch:** `chore/ci-coverage` → `main`

## Summary

Enforces an 85% coverage floor in the test suite, widens CI's lint scope
to cover `data/`, `benchmark/`, and `dashboards/` (which had grown across
prior PRs without CI checking them), and fixes a real drift between
pre-commit's mypy hook and what CI actually runs.

## Why

This is the "make sure it actually holds together" pass before moving to
docs/paper work: verify the whole project passes on a genuinely
independent clean clone, not just my own cached working directory.

## What's included

- `--cov-fail-under=85` in `pyproject.toml`'s pytest config.
- CI's ruff/black steps now cover `data`, `benchmark`, `dashboards` in
  addition to `src`, `tests`.
- Two real typing bugs found by running `pre-commit run --all-files` for
  the first time in a while (its mypy hook had `pandas-stubs` that CI's
  `mypy src` doesn't use): an unqualified dict value type in
  `equity/choropleth.py`'s grid lookup, and an imprecise dtype-override
  type in `cohort/omop_etl.py`. Fixed both; removed the extra stub
  packages from the hook so it stays consistent with CI instead of being
  a stricter check that only sometimes runs.
- A real bug in `benchmark/run_benchmark.py`'s markdown generator: it
  relied on a trailing double-space for a Markdown hard line break,
  which the `trailing-whitespace` pre-commit hook strips on every
  commit, silently merging two lines into one paragraph. Switched to
  bullet points.
- Verified Python 3.11 *and* 3.12 both pass the full suite.
- Verified the complete lint/format/mypy/pytest sequence on a genuinely
  independent `git clone` of `main`, not the working directory.

## Testing

```
uv run pre-commit run --all-files
uv run pytest
```

173 passed, 96.87% coverage (floor: 85%). Both green on a fresh clone.

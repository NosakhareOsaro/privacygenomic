# Contributing to PrivacyGenomic / equitable_rwe

Thanks for your interest in contributing. This project follows a
research-software-engineering workflow intended to keep history reviewable
and results reproducible.

## Development setup

```bash
uv sync --extra dev
uv run pre-commit install
```

## Workflow

- Work in a feature branch named `feat/<topic>`, `fix/<topic>`, or
  `docs/<topic>`.
- Commit using [Conventional Commits](https://www.conventionalcommits.org/):
  a `type: summary` subject line under 72 characters, followed by a body
  explaining *why* the change was made where that isn't obvious from the
  diff. Common types: `feat`, `fix`, `docs`, `test`, `ci`, `refactor`,
  `chore`.
- Keep commits small and atomic — one logical change per commit.
- Run linting, type checks, and the test suite locally before opening a
  pull request:

  ```bash
  uv run ruff check src tests
  uv run black --check src tests
  uv run mypy src
  uv run pytest
  ```

- Every public function or class requires a NumPy-style docstring with a
  runnable example.
- Do not commit real patient data, credentials, or PhysioNet-derived raw
  extracts. Only synthetic or public demo data belongs in this repository —
  see [docs/data-governance/ocap-and-privacy.md](docs/data-governance/ocap-and-privacy.md).

## Pull requests

Describe the change, its motivation, and how it was tested. If you're
proposing a new statistical method, include a short note on validation
(e.g. comparison against a known-good implementation or a reference result).

## Reporting issues

Use the issue templates under `.github/ISSUE_TEMPLATE/`. Bug reports should
include a minimal reproduction against the synthetic cohort where possible.

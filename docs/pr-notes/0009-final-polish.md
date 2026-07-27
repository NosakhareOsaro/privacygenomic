# PR: final polish — README, badges, architecture diagram, results

**Branch:** `docs/final-readme-polish` → `main`

## Summary

Rewrites the top-level README as the project's actual front door: badges,
a Mermaid architecture diagram, a quickstart, the benchmark results table
stated in plain language, a repository-structure overview, and an
explicit "what's left to do manually" section.

## What's included

- Badges: CI status (real, resolves once pushed), Python version, MIT
  license (static), coverage (static, states clearly it reflects the
  last local run rather than a live Codecov integration), and a DOI
  placeholder that says explicitly it's pending the first release —
  no badge here claims something that isn't true yet.
- A Mermaid architecture diagram (renders natively on GitHub) showing
  data flow from the synthetic/schema-matched sources through the OMOP
  ETL, the six `equitable_rwe` submodules, the flagship combined API,
  and out to the benchmark, dashboard, and paper.
- A **Results** section with the four-estimator benchmark table restated
  in plain language — what the naive estimator's bias means, what the
  E-value column is actually asking, all traceable back to
  `benchmark/results/benchmark_summary.md`'s code-generated numbers.
- An explicit manual-steps list: PhysioNet credentials, GitHub push,
  Zenodo linking, ReadTheDocs import, JOSS submission, optional Codecov
  — matching exactly what this project cannot do without a human account.
- Cleaned up a `CHANGELOG.md` structural issue: two separate `### Added`
  sections had accumulated under `[Unreleased]` from consecutive PRs;
  merged into one.

## Testing

```
uv run pre-commit run --all-files
uv run pytest
```

173 passed, 96.87% coverage. All checks green.

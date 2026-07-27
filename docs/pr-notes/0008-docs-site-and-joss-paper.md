# PR: Sphinx API reference site, JOSS paper draft, Zenodo metadata

**Branch:** `docs/api-site-and-joss-paper` → `main`

## Summary

Adds a Sphinx API reference site (`docs/source/`), ReadTheDocs config, a
JOSS-format paper draft (`paper/`), and `.zenodo.json`.

## What's included

- `docs/source/` — Sphinx (Furo theme) with autodoc + napoleon (for the
  NumPy-style docstrings already throughout `src/equitable_rwe`) and
  MyST (Markdown support). One page per submodule
  (`api/{cohort,survival,causal,fairness,ml,equity,top_level}.rst`).
  Builds warning-free with `-W` (warnings-as-errors) — this took a real
  fix: napoleon's default `Attributes` rendering collided with autodoc's
  own dataclass field discovery, producing 58 "duplicate object
  description" warnings; `napoleon_use_ivar = True` resolves it (the
  documented fix for exactly this interaction, not a workaround).
  Loose Markdown docs elsewhere in the repo (ADRs, data governance,
  PR notes) are linked from the landing page rather than duplicated into
  the Sphinx tree.
- `.readthedocs.yaml` — deliberately uses ReadTheDocs' standard
  `pip install .[docs]` pattern rather than wiring `uv` into RTD's own
  environment management, since that integration isn't something I can
  verify without a live RTD account; CI's own `docs` job does use `uv`,
  since GitHub Actions is fully within my control to verify.
- `.github/workflows/ci.yml` — new `docs` job building the site with
  `-W`.
- `paper/paper.md`, `paper/paper.bib` — a JOSS-format draft. Every
  `@citekey` used has a matching bibliography entry and vice versa,
  checked programmatically. Not submitted to JOSS (would need a LaTeX
  toolchain and JOSS's own tooling this environment doesn't have); see
  `paper/README.md` for what's left before real submission.
- `.zenodo.json` — deposit metadata for when a GitHub release is cut.

## Testing

```
uv run sphinx-build -b html docs/source docs/_build/html -W
uv run pytest
uv run pre-commit run --all-files
```

Docs build clean with zero warnings. 173 tests pass, 96.87% coverage.

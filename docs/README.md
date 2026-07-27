# Documentation

`docs/source/` is the Sphinx source for the API reference site
(Furo theme, autodoc + napoleon for the NumPy-style docstrings throughout
`src/equitable_rwe`, MyST for Markdown support). It is separate from the
loose Markdown documentation elsewhere in `docs/` (`architecture/`,
`data-governance/`, `pr-notes/`), which is meant to be read directly on
GitHub and is linked from the Sphinx site's landing page rather than
duplicated into it.

## Build locally

```bash
uv sync --extra docs
uv run sphinx-build -b html docs/source docs/_build/html -W
open docs/_build/html/index.html
```

`-W` treats warnings as errors, matching what CI enforces (the `docs` job
in `.github/workflows/ci.yml`) and what ReadTheDocs enforces
(`fail_on_warning: true` in `.readthedocs.yaml`).

## ReadTheDocs

`.readthedocs.yaml` at the repo root configures the ReadTheDocs build via
a plain `pip install .[docs]`, deliberately not `uv`, since ReadTheDocs'
own environment/tooling integration with `uv` is not something this
project has been able to verify against a live ReadTheDocs build.
Importing the project (for `pip install .[docs]`) is unaffected by which core
package manager was used to develop it.

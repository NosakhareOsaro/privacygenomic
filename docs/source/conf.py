"""Sphinx configuration for PrivacyGenomic / equitable_rwe."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / "src"))

import equitable_rwe  # noqa: E402

project = "equitable_rwe"
copyright = "2026, Nosakhare Osaro"
author = "Nosakhare Osaro"
release = equitable_rwe.__version__

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx_copybutton",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"

napoleon_google_docstring = False
napoleon_numpy_docstring = True
napoleon_use_param = True
napoleon_use_rtype = False
# Render docstring "Attributes" sections as :ivar: field lists rather
# than `.. attribute::` directives: the latter collides with autodoc's
# own discovery of dataclass fields as class members, producing
# "duplicate object description" warnings for every dataclass result
# type in this codebase.
napoleon_use_ivar = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "pandas": ("https://pandas.pydata.org/docs", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "lifelines": ("https://lifelines.readthedocs.io/en/latest", None),
    "sklearn": ("https://scikit-learn.org/stable", None),
}

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

html_theme = "furo"
html_static_path = ["_static"]
html_title = "equitable_rwe"

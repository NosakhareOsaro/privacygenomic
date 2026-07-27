# JOSS paper draft

`paper.md` and `paper.bib` are a draft in
[JOSS](https://joss.theoj.org/) format — they have not been submitted to
JOSS. Verified so far:

- YAML frontmatter parses and includes every field JOSS requires
  (`title`, `tags`, `authors`, `affiliations`, `date`, `bibliography`).
- Every `@citekey` used in `paper.md` has a matching entry in
  `paper.bib`, and vice versa (no missing or unused references).

Not verified: actual PDF rendering via JOSS's `whedon`/Open Journals
build tooling, which requires a LaTeX toolchain this environment does
not have installed. Before real submission, render it with the [Open
Journals `inara` Docker image](https://github.com/openjournals/inara)
or JOSS's own submission preview and fix anything that surfaces.

Submitting to JOSS itself, and linking a GitHub release to Zenodo (which
`.zenodo.json` at the repo root configures the metadata for), both
require accounts this environment does not have — see the top-level
project README's "what's left to do manually" section.

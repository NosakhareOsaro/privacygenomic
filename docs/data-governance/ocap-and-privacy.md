# OCAP® and data governance

## What OCAP® is, and why it matters here

OCAP® — Ownership, Control, Access, and Possession — is a set of
principles asserting that First Nations peoples and communities have the
right to govern the collection, use, and disclosure of data about
themselves, their communities, and their lands. It is a registered
trademark of the First Nations Information Governance Centre (FNIGC),
which stewards it as both a political assertion of self-determination
over data and a practical framework for research and health data
governance:

- **Ownership** — a First Nation or Indigenous community owns information
  collected about it, in the same way an individual owns information
  about themselves.
- **Control** — First Nations have the right to control all aspects of
  research and information management processes that affect them,
  including data collection, use, and, importantly, interpretation.
- **Access** — First Nations must have access to information and data
  about themselves and their communities, regardless of where it is
  physically held.
- **Possession** — physical control (possession) of data is a mechanism
  by which ownership can be asserted and protected, distinct from
  ownership itself.

OCAP® applies specifically to First Nations data in Canada; Métis and
Inuit communities and organizations have articulated related but distinct
data governance frameworks. This project treats OCAP® as the minimum bar
for how any project touching Indigenous-identity data in a Canadian
health context should reason about governance, whether or not it directly
handles First Nations data.

## The decision this project makes

**This repository does not use, request, or process any real
Indigenous-identity patient-level data, at any point.** Every dataset in
this repository is either:

1. A small, deterministically generated fixture matching a public data
   schema (`tests/fixtures/mimic_demo/`), containing no data derived from
   real patients at all; or
2. A synthetic cohort (`data/synthetic/`) whose *aggregate* statistical
   properties are calibrated against **public, already-published**
   summary statistics from Statistics Canada and CIHI — never against any
   patient-level extract, and never against any dataset for which consent,
   governance, or a data-sharing agreement with an Indigenous community or
   governing body would be required.

This is a direct consequence of OCAP®, not merely a licensing
convenience: the right to control how data about a First Nation is
collected, used, and interpreted belongs to that Nation. A research
software portfolio project built by an individual, outside of any
relationship with a specific community or its governing data authority,
has no standing to hold, request, or make claims from real Indigenous
patient data — synthetic and public-aggregate data is the *only*
appropriate input for this kind of independent project, not a fallback
chosen for convenience.

### What "Indigenous-identity proxy" means in this codebase

The synthetic generator (`data/synthetic/generate_equity_cohort.py`)
includes a boolean field, `indigenous_identity_proxy`, used to model and
illustrate outcome and access disparities. Three things are true about it
simultaneously:

1. It is entirely synthetic — a Bernoulli draw calibrated to the national
   population share reported in the 2021 Census, with no linkage to any
   real individual.
2. It is explicitly named and documented as a *proxy*, not as
   "Indigenous status" — self-identification is the only legitimate way
   to determine Indigenous identity, and no synthetic or administrative
   proxy can substitute for it in real analytic work.
3. Its purpose is narrow: to give the survival, causal inference, and
   fairness-auditing modules in this project something realistic to
   detect and report on, so that this codebase's methodology — doubly
   robust causal estimation combined with subgroup fairness auditing —
   can be demonstrated end-to-end on a cohort with a real-shaped equity
   gap, without ever touching real data that would require it.

Any real-world application of this codebase to actual Indigenous-identity
data would need to be done in direct partnership with, and under the
governance of, the relevant Nation(s) or their designated data authority
(for example, via the FNIGC's First Nations Regional Health Survey (RHS)
governance model), following OCAP® in practice — not merely citing it in
a document.

### Non-Indigenous personal health information

The same reasoning applies more broadly: this repository processes no
real patient data of any kind. The OMOP CDM ETL
(`equitable_rwe.cohort`) is written against, and tested against, a
synthetic fixture matching the MIMIC-IV Clinical Database Demo schema
(see `docs/architecture/adr-0001-omop-etl.md`) rather than downloaded
demo data, specifically to avoid any question about redistributing
PhysioNet-governed data in a public repository. Credentialed use of real
MIMIC-IV data against this ETL is left to whoever holds that credential,
run entirely outside this repository.

## Practical checklist this project follows

- No `data/raw/`, `data/interim/`, or similar directory is ever committed
  (see `.gitignore`); only `data/synthetic/` is version-controlled.
- Every synthetic or fixture dataset's generator script is committed
  alongside its output, so provenance is always inspectable and
  reproducible from source, not just from a static file.
- Every calibration claim is backed by a cited public source and, where
  possible, a generated report showing the *actual* achieved statistic
  against that target (`data/synthetic/calibration_report.md`) — no
  disparity is asserted that the code did not measure.
- Any future contribution that introduces a real patient-level dataset —
  Indigenous or otherwise — into this repository should be rejected in
  code review unless it is accompanied by documented consent, governance,
  and legal authority to hold that data, consistent with this document.

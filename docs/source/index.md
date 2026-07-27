# equitable_rwe

Doubly-robust causal outcome estimation with integrated subgroup
fairness auditing for real-world clinical evidence.

`equitable_rwe` is the core Python library behind
[PrivacyGenomic](https://github.com/NosakhareOsaro/privacygenomic), a
reproducible pipeline turning OMOP-CDM-formatted EHR data into causally
valid, fairness-audited clinical outcome estimates and health-equity
intelligence.

```python
from equitable_rwe import fit_equitable_survival_estimate

result = fit_equitable_survival_estimate(
    df,
    duration_col="time_to_event_days",
    event_col="event_observed",
    treatment_col="treat",
    covariates=["age", "comorbidity_score"],
    subgroup_col="remoteness_level",
)
result.hazard_ratio      # the causal estimate
result.is_equitable      # is the evidence behind it adequate for every subgroup?
```

## Where to start

- **API reference** ({doc}`api/index`) — every public function and class,
  with a worked example.
- **Architecture decisions** — the non-obvious trade-offs behind this
  library's design (Fine-Gray via IPCW reweighting rather than a
  from-scratch subdistribution likelihood; DoWhy for backdoor
  identification and refutation rather than hand-rolled graph logic; a
  weight-and-adjust doubly robust estimator rather than a full AIPW
  estimator; two distinct notions of "fairness" for a classifier versus
  a causal estimate) are recorded as ADRs in the repository:
  [docs/architecture](https://github.com/NosakhareOsaro/privacygenomic/tree/main/docs/architecture).
- **Data governance** — this project uses only synthetic and
  public-aggregate data, never real patient data; see
  [OCAP® and data governance](https://github.com/NosakhareOsaro/privacygenomic/blob/main/docs/data-governance/ocap-and-privacy.md).
- **Benchmark results** — the four-estimator causal-adjustment comparison
  against a known ground truth:
  [benchmark/results](https://github.com/NosakhareOsaro/privacygenomic/tree/main/benchmark/results).
- **Changelog** —
  [CHANGELOG.md](https://github.com/NosakhareOsaro/privacygenomic/blob/main/CHANGELOG.md).

```{toctree}
:maxdepth: 2
:hidden:

api/index
```

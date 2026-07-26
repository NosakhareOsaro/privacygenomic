# Calibration notes: synthetic equity-calibrated cohort

`generate_equity_cohort.py` produces a fully synthetic patient-level
cohort used to exercise the survival, causal, and fairness modules in this
project against data with realistic, defensible health-equity gradients.
This document records exactly what is and isn't calibrated to a real,
citable public statistic, and why. See
[docs/data-governance/ocap-and-privacy.md](../../docs/data-governance/ocap-and-privacy.md)
for why synthetic data is used at all rather than any real Indigenous
patient data.

## Precisely calibrated

| Quantity | Target | Source |
|---|---|---|
| Indigenous-identity-proxy population share | ~5.0% | Statistics Canada, 2021 Census of Population (Indigenous peoples reported as approximately 5.0% of the total Canadian population). |
| Remote-vs-accessible hospitalization rate ratio | 8.74 / 6.75 ≈ 1.295 | CIHI, "Rural health care in Canada" — national rural hospitalization rate of 8.74 per 100 residents vs. urban rate of 6.75 per 100 residents, 2024–2025. `remote` and `very_remote` synthetic remoteness levels stand in for CIHI's "rural" designation; these are related but not identical classification systems (CIHI's rural/urban split vs. Statistics Canada's five-level Index of Remoteness), which is why this is described as a proxy substitution, not an exact reproduction of the same construct. |

The generator's `calibration_report()` function computes both quantities
from the *actually generated* sample and writes the observed values,
targets, and relative error to `calibration_report.md` every time the
generator runs — see that file for the current run's numbers. This
project does not state a calibration claim anywhere that this function
did not produce.

## Directionally calibrated (not point-precise — documented explicitly)

These reflect real, well-documented disparities in direction and rough
order of magnitude, but the specific numeric parameters (hazard ratios,
logit shifts) are illustrative choices, not fitted or reproduced from a
specific published statistic:

- **Survival hazard ratio for the Indigenous-identity proxy**
  (`INDIGENOUS_SURVIVAL_HAZARD_RATIO = 1.45`). Statistics Canada
  (catalogue 82-003-X, "Mortality among First Nations people, 2006 to
  2016") reports a First Nations life-expectancy shortfall of 8.9 years
  (males) and 9.6 years (females) relative to the non-Indigenous
  population, based on 2011 census-mortality-linked data. That is a
  *life-expectancy-at-birth* statistic; this generator instead produces
  *time from diagnosis to death* for a chronic-disease cohort — a
  different construct with a different baseline scale. Forcing a
  diagnosis-to-death Weibull model to reproduce an exact
  life-expectancy-at-birth gap would overstate the precision of this
  calibration, so instead the hazard ratio is set to a plausible,
  clearly-labeled illustrative value that produces a disparity in the
  correct direction and a defensible order of magnitude, not a specific
  target number.
- **Survival hazard ratio for remoteness** (`REMOTE_SURVIVAL_HAZARD_RATIO
  = 1.20`) and **later-stage-at-diagnosis probability** for both remote
  and Indigenous-proxy patients — motivated by the general, widely
  documented pattern that reduced geographic access to care is associated
  with later presentation and worse outcomes (consistent with, though not
  numerically derived from, the CIHI rural access findings above), but not
  tied to a specific published stage-shift or hazard-ratio statistic for
  this synthetic cohort's illustrative condition set.
- **Income quintile gradient** by remoteness and Indigenous-proxy status —
  reflects the general, well-documented association between geographic
  remoteness/marginalization and lower income, without claiming a specific
  quantile-shift statistic.

## Not calibrated at all (illustrative only)

- Relative prevalence of the four illustrative conditions
  (`colorectal_cancer`, `breast_cancer`, `lung_cancer`,
  `cardiovascular_event`) — sampled uniformly; no attempt is made to
  match real Canadian incidence rates for these conditions.
- Province population shares — approximate, order-of-magnitude only, used
  solely to make the equity dashboard's choropleth (see `dashboards/`)
  visually plausible.
- Remoteness-level population shares (68% / 16% / 8% / 5% / 3%) — a
  plausible illustrative distribution, not sourced from Statistics
  Canada's actual Index of Remoteness population counts.

## Reproducing

```bash
python data/synthetic/generate_equity_cohort.py
```

This overwrites `equity_cohort.parquet`, `equity_cohort_preview.csv`, and
`calibration_report.md` deterministically (fixed seed).

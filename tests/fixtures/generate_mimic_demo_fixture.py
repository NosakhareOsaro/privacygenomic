"""Generate a small, fully synthetic fixture matching the MIMIC-IV Clinical
Database Demo ``hosp`` module schema.

This does **not** contain any real patient data. It exists so the OMOP CDM
ETL in ``equitable_rwe.cohort`` can be exercised in CI without a network
dependency or a PhysioNet account, while still being written against the
real MIMIC-IV demo table and column names (see
``docs/architecture/adr-0001-omop-etl.md`` for the rationale). Anyone with
PhysioNet access can point the same ETL functions at the real
``mimic-iv-clinical-database-demo`` directory instead.

Run with: ``python tests/fixtures/generate_mimic_demo_fixture.py``
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT_DIR = Path(__file__).parent / "mimic_demo"
N_PATIENTS = 30
SEED = 20260726

_ICD9_CONDITIONS = [
    ("4019", "Unspecified essential hypertension"),
    ("4280", "Congestive heart failure, unspecified"),
    ("25000", "Diabetes mellitus without mention of complication, type II"),
    ("5849", "Acute kidney failure, unspecified"),
    ("41401", "Coronary atherosclerosis of native coronary artery"),
    ("2724", "Other and unspecified hyperlipidemia"),
    ("496", "Chronic airway obstruction, not elsewhere classified"),
]

_DRUGS = [
    ("Metoprolol Tartrate", "51079075920"),
    ("Furosemide", "00054829925"),
    ("Metformin", "00378211001"),
    ("Atorvastatin", "00071015523"),
    ("Insulin Human Regular", "00088221033"),
    ("Albuterol", "00591343060"),
]

_LAB_ITEMS = [
    (50912, "Creatinine", "Blood", "Chemistry"),
    (50971, "Potassium", "Blood", "Chemistry"),
    (51301, "White Blood Cells", "Blood", "Hematology"),
    (50931, "Glucose", "Blood", "Chemistry"),
    (50813, "Lactate", "Blood", "Blood Gas"),
]

_ADMISSION_TYPES = ["EW EMER.", "ELECTIVE", "URGENT", "OBSERVATION ADMIT"]
_RACES = ["WHITE", "BLACK/AFRICAN AMERICAN", "ASIAN", "HISPANIC/LATINO", "UNKNOWN"]


def generate() -> None:
    rng = np.random.default_rng(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    subject_ids = np.arange(10000000, 10000000 + N_PATIENTS)
    genders = rng.choice(["M", "F"], size=N_PATIENTS)
    anchor_ages = rng.integers(40, 90, size=N_PATIENTS)
    anchor_years = rng.integers(2110, 2120, size=N_PATIENTS)
    died_mask = rng.random(N_PATIENTS) < 0.15
    death_offsets = rng.integers(30, 700, size=N_PATIENTS)
    dod = np.where(
        died_mask,
        [
            pd.Timestamp(int(y), 1, 1) + pd.Timedelta(days=int(d))
            for y, d in zip(anchor_years, death_offsets, strict=True)
        ],
        pd.NaT,
    )

    patients = pd.DataFrame(
        {
            "subject_id": subject_ids,
            "gender": genders,
            "anchor_age": anchor_ages,
            "anchor_year": anchor_years,
            "anchor_year_group": [f"{y - (y % 3)} - {y - (y % 3) + 2}" for y in anchor_years],
            "dod": dod,
        }
    )
    patients.to_csv(OUT_DIR / "patients.csv", index=False)

    admission_rows = []
    diagnosis_rows = []
    prescription_rows = []
    labevent_rows = []
    hadm_counter = 20000000

    for subj, anchor_year in zip(subject_ids, anchor_years, strict=True):
        n_admissions = rng.integers(1, 4)
        for _ in range(n_admissions):
            hadm_id = hadm_counter
            hadm_counter += 1
            admit = pd.Timestamp(int(anchor_year), 1, 1) + pd.Timedelta(
                days=int(rng.integers(0, 300))
            )
            los_days = rng.integers(1, 14)
            discharge = admit + pd.Timedelta(days=int(los_days))
            race = rng.choice(_RACES)
            admission_rows.append(
                {
                    "subject_id": subj,
                    "hadm_id": hadm_id,
                    "admittime": admit,
                    "dischtime": discharge,
                    "deathtime": pd.NaT,
                    "admission_type": rng.choice(_ADMISSION_TYPES),
                    "admission_location": "EMERGENCY ROOM",
                    "discharge_location": "HOME",
                    "insurance": rng.choice(["Medicare", "Private", "Medicaid"]),
                    "language": "ENGLISH",
                    "marital_status": rng.choice(["MARRIED", "SINGLE", "WIDOWED"]),
                    "race": race,
                    "hospital_expire_flag": 0,
                }
            )

            n_dx = rng.integers(1, 4)
            dx_choices = rng.choice(len(_ICD9_CONDITIONS), size=n_dx, replace=False)
            for seq, idx in enumerate(dx_choices, start=1):
                code, _ = _ICD9_CONDITIONS[idx]
                diagnosis_rows.append(
                    {
                        "subject_id": subj,
                        "hadm_id": hadm_id,
                        "seq_num": seq,
                        "icd_code": code,
                        "icd_version": 9,
                    }
                )

            n_rx = rng.integers(1, 3)
            rx_choices = rng.choice(len(_DRUGS), size=n_rx, replace=False)
            for idx in rx_choices:
                drug, ndc = _DRUGS[idx]
                start = admit + pd.Timedelta(hours=int(rng.integers(1, 24)))
                stop = start + pd.Timedelta(days=int(rng.integers(1, los_days + 1)))
                prescription_rows.append(
                    {
                        "subject_id": subj,
                        "hadm_id": hadm_id,
                        "starttime": start,
                        "stoptime": stop,
                        "drug": drug,
                        "ndc": ndc,
                        "dose_val_rx": rng.integers(5, 100),
                        "dose_unit_rx": "mg",
                    }
                )

            n_labs = rng.integers(2, 6)
            lab_choices = rng.choice(len(_LAB_ITEMS), size=n_labs, replace=True)
            for idx in lab_choices:
                itemid, _, _, _ = _LAB_ITEMS[idx]
                charttime = admit + pd.Timedelta(hours=int(rng.integers(1, los_days * 24)))
                valuenum = float(rng.normal(loc=5.0, scale=2.0))
                labevent_rows.append(
                    {
                        "subject_id": subj,
                        "hadm_id": hadm_id,
                        "itemid": itemid,
                        "charttime": charttime,
                        "value": f"{valuenum:.1f}",
                        "valuenum": valuenum,
                        "valueuom": "mg/dL",
                        "flag": rng.choice([None, "abnormal"], p=[0.8, 0.2]),
                    }
                )

    pd.DataFrame(admission_rows).to_csv(OUT_DIR / "admissions.csv", index=False)
    pd.DataFrame(diagnosis_rows).to_csv(OUT_DIR / "diagnoses_icd.csv", index=False)
    pd.DataFrame(prescription_rows).to_csv(OUT_DIR / "prescriptions.csv", index=False)
    pd.DataFrame(labevent_rows).to_csv(OUT_DIR / "labevents.csv", index=False)

    pd.DataFrame(
        [{"icd_code": c, "icd_version": 9, "long_title": t} for c, t in _ICD9_CONDITIONS]
    ).to_csv(OUT_DIR / "d_icd_diagnoses.csv", index=False)

    pd.DataFrame(
        [
            {"itemid": i, "label": lbl, "fluid": fluid, "category": cat}
            for i, lbl, fluid, cat in _LAB_ITEMS
        ]
    ).to_csv(OUT_DIR / "d_labitems.csv", index=False)


if __name__ == "__main__":
    generate()
    print(f"Wrote synthetic MIMIC-IV-demo-shaped fixture to {OUT_DIR}")

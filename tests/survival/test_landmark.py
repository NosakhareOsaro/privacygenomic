import pandas as pd

from equitable_rwe.survival.landmark import fit_landmark_cox, landmark_dataset, landmark_table


def test_landmark_dataset_drops_subjects_not_at_risk():
    df = pd.DataFrame({"T": [5, 15, 30, 100], "E": [1, 0, 1, 1]})
    out = landmark_dataset(df, "T", "E", landmark_time=10)
    assert len(out) == 3
    assert (out["T"] > 0).all()


def test_landmark_dataset_resets_time_origin():
    df = pd.DataFrame({"T": [50], "E": [1]})
    out = landmark_dataset(df, "T", "E", landmark_time=20)
    assert out["T"].iloc[0] == 30


def test_fit_landmark_cox_runs(survival_df):
    cph = fit_landmark_cox(survival_df, "T", "E", landmark_time=5, covariates=["x1"])
    assert "x1" in cph.params_.index


def test_landmark_table_has_one_row_per_landmark_time(survival_df):
    table = landmark_table(survival_df, "T", "E", landmark_times=[0, 5, 10], covariate="x1")
    assert list(table["landmark_time"]) == [0, 5, 10]
    assert (table["n_at_risk"] > 0).all()
    assert (table["n_at_risk"].diff().dropna() <= 0).all()

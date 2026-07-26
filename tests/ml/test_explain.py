from equitable_rwe.ml.explain import explain_predictions
from equitable_rwe.ml.readmission import fit_readmission_model


def test_explain_predictions_shape_matches_input(readmission_cohort, readmission_module):
    result = fit_readmission_model(
        readmission_cohort,
        readmission_module.FEATURE_COLUMNS,
        readmission_module.LABEL_COLUMN,
        seed=0,
    )
    X_test = readmission_cohort.loc[result.test_index, result.feature_columns]
    explanation = explain_predictions(result.model, X_test)
    assert explanation.values.shape == (len(X_test), len(result.feature_columns))


def test_importance_covers_every_feature(readmission_cohort, readmission_module):
    result = fit_readmission_model(
        readmission_cohort,
        readmission_module.FEATURE_COLUMNS,
        readmission_module.LABEL_COLUMN,
        seed=0,
    )
    X_test = readmission_cohort.loc[result.test_index, result.feature_columns]
    explanation = explain_predictions(result.model, X_test)
    assert set(explanation.importance.index) == set(result.feature_columns)
    assert (explanation.importance >= 0).all()


def test_clinical_utilization_features_rank_above_noise(readmission_cohort, readmission_module):
    # n_prior_admissions_12mo and n_comorbidities are the strongest
    # drivers of the injected label; they should outrank income_quintile,
    # a much weaker signal in the DGP.
    result = fit_readmission_model(
        readmission_cohort,
        readmission_module.FEATURE_COLUMNS,
        readmission_module.LABEL_COLUMN,
        seed=0,
    )
    X_test = readmission_cohort.loc[result.test_index, result.feature_columns]
    explanation = explain_predictions(result.model, X_test)
    top_feature = explanation.importance.index[0]
    assert top_feature in {"n_prior_admissions_12mo", "n_comorbidities"}

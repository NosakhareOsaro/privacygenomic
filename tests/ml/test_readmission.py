from equitable_rwe.ml.readmission import fit_readmission_model


def test_fit_readmission_model_has_reasonable_auc(readmission_cohort, readmission_module):
    result = fit_readmission_model(
        readmission_cohort,
        readmission_module.FEATURE_COLUMNS,
        readmission_module.LABEL_COLUMN,
        seed=0,
    )
    assert 0.5 < result.auc <= 1.0


def test_fit_readmission_model_test_set_sizes_match(readmission_cohort, readmission_module):
    result = fit_readmission_model(
        readmission_cohort,
        readmission_module.FEATURE_COLUMNS,
        readmission_module.LABEL_COLUMN,
        test_size=0.3,
        seed=0,
    )
    expected_n = round(len(readmission_cohort) * 0.3)
    assert abs(len(result.test_index) - expected_n) <= 1
    assert len(result.y_test) == len(result.test_index)
    assert len(result.y_pred_proba) == len(result.test_index)


def test_predicted_probabilities_are_valid(readmission_cohort, readmission_module):
    result = fit_readmission_model(
        readmission_cohort,
        readmission_module.FEATURE_COLUMNS,
        readmission_module.LABEL_COLUMN,
        seed=0,
    )
    assert (result.y_pred_proba >= 0).all()
    assert (result.y_pred_proba <= 1).all()


def test_protected_attributes_are_not_model_features(readmission_module):
    assert "indigenous_identity_proxy" not in readmission_module.FEATURE_COLUMNS
    assert "remoteness_level" not in readmission_module.FEATURE_COLUMNS

"""XGBoost 30-day readmission model.

Deliberately excludes protected and proxy-for-protected attributes
(e.g. Indigenous-identity proxy, remoteness) from the feature set — see
``data/synthetic/generate_readmission_cohort.py`` for why that alone does
not guarantee a fair model, and :mod:`equitable_rwe.fairness` for the
post-hoc audit that checks whether it did.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


@dataclass
class ReadmissionModelResult:
    """A fitted readmission model plus held-out test-set predictions.

    Attributes
    ----------
    model : xgboost.XGBClassifier
        The fitted model.
    feature_columns : list[str]
        Feature columns used, in training order.
    test_index : pandas.Index
        Index of the held-out test rows (into the original data frame),
        so predictions can be joined back to subgroup columns for
        fairness auditing.
    y_test : numpy.ndarray
        True labels for the test set.
    y_pred_proba : numpy.ndarray
        Predicted probability of readmission for the test set.
    auc : float
        Test-set area under the ROC curve.
    """

    model: XGBClassifier
    feature_columns: list[str]
    test_index: pd.Index
    y_test: np.ndarray
    y_pred_proba: np.ndarray
    auc: float


def fit_readmission_model(
    df: pd.DataFrame,
    feature_columns: list[str],
    label_column: str,
    test_size: float = 0.25,
    seed: int = 0,
) -> ReadmissionModelResult:
    """Fit an XGBoost 30-day readmission classifier with a held-out test split.

    Parameters
    ----------
    df : pandas.DataFrame
        One row per patient.
    feature_columns : list[str]
        Columns to use as model features.
    label_column : str
        Binary (0/1) readmission label column.
    test_size : float, default 0.25
        Fraction of rows held out for evaluation and fairness auditing.
    seed : int, default 0
        Random seed for the train/test split and the model.

    Returns
    -------
    ReadmissionModelResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> x1 = rng.normal(size=n)
    >>> label = rng.binomial(1, 1 / (1 + np.exp(-x1)))
    >>> df = pd.DataFrame({"x1": x1, "label": label})
    >>> result = fit_readmission_model(df, ["x1"], "label", seed=0)
    >>> 0.5 <= result.auc <= 1.0
    True
    """
    train_df, test_df = train_test_split(
        df, test_size=test_size, random_state=seed, stratify=df[label_column]
    )

    model = XGBClassifier(
        n_estimators=200,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="logloss",
        random_state=seed,
    )
    model.fit(train_df[feature_columns], train_df[label_column])

    y_pred_proba = model.predict_proba(test_df[feature_columns])[:, 1]
    y_test = test_df[label_column].to_numpy()
    auc = float(roc_auc_score(y_test, y_pred_proba))

    return ReadmissionModelResult(
        model=model,
        feature_columns=list(feature_columns),
        test_index=test_df.index,
        y_test=y_test,
        y_pred_proba=y_pred_proba,
        auc=auc,
    )

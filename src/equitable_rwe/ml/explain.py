"""SHAP explainability for the readmission model."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import shap
from xgboost import XGBClassifier


@dataclass
class ShapExplanation:
    """SHAP values for a set of predictions, plus a feature-importance summary.

    Attributes
    ----------
    values : numpy.ndarray
        SHAP values, shape ``(n_rows, n_features)``.
    feature_columns : list[str]
        Feature names, matching the columns of ``values``.
    base_value : float
        The explainer's expected (base) model output.
    importance : pandas.Series
        Mean absolute SHAP value per feature, sorted descending — the
        standard global feature-importance summary derived from SHAP.
    """

    values: np.ndarray
    feature_columns: list[str]
    base_value: float
    importance: pd.Series


def explain_predictions(model: XGBClassifier, X: pd.DataFrame) -> ShapExplanation:
    """Compute SHAP values for a fitted XGBoost model's predictions.

    Parameters
    ----------
    model : xgboost.XGBClassifier
        A fitted model (see :func:`equitable_rwe.ml.readmission.fit_readmission_model`).
    X : pandas.DataFrame
        Rows to explain, with the same feature columns used to fit
        ``model``.

    Returns
    -------
    ShapExplanation

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from xgboost import XGBClassifier
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> X = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})
    >>> y = rng.binomial(1, 1 / (1 + np.exp(-X["x1"])))
    >>> model = XGBClassifier(n_estimators=50, max_depth=2).fit(X, y)
    >>> explanation = explain_predictions(model, X)
    >>> explanation.importance.index[0]
    'x1'
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer(X)

    importance = pd.Series(np.abs(shap_values.values).mean(axis=0), index=X.columns).sort_values(
        ascending=False
    )

    return ShapExplanation(
        values=shap_values.values,
        feature_columns=list(X.columns),
        base_value=float(np.asarray(shap_values.base_values).mean()),
        importance=importance,
    )

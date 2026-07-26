"""XGBoost 30-day readmission model with SHAP explainability."""

from equitable_rwe.ml.explain import ShapExplanation, explain_predictions
from equitable_rwe.ml.readmission import ReadmissionModelResult, fit_readmission_model

__all__ = [
    "ShapExplanation",
    "explain_predictions",
    "ReadmissionModelResult",
    "fit_readmission_model",
]

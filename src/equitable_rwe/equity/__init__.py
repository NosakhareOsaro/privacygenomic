"""Health-equity intelligence: age-standardized rates (direct
standardization and Poisson regression) and a simplified GIS choropleth.
"""

from equitable_rwe.equity.age_standardization import (
    StandardizedRateResult,
    direct_standardize,
    fit_poisson_rate_model,
)
from equitable_rwe.equity.choropleth import build_province_grid, render_choropleth_html

__all__ = [
    "StandardizedRateResult",
    "direct_standardize",
    "fit_poisson_rate_model",
    "build_province_grid",
    "render_choropleth_html",
]

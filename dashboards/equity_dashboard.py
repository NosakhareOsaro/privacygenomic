"""Health-equity dashboard (Streamlit).

An open-source alternative to a Power BI dashboard, chosen specifically
so reviewers without a Power BI license can run it. All figures are
computed from the synthetic, OCAP-governed cohort in
``data/synthetic/equity_cohort.parquet`` (see
``docs/data-governance/ocap-and-privacy.md``) — nothing here is a real
patient record.

Run with: ``streamlit run dashboards/equity_dashboard.py``
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).parent))
from data_prep import (  # noqa: E402
    age_standardized_hospitalization_rate_by_remoteness,
    income_quintile_by_remoteness,
    load_cohort,
    province_choropleth_values,
    stage_distribution_by_indigenous_proxy,
)

from equitable_rwe.equity.choropleth import (
    build_province_grid,
    render_choropleth_html,
)  # noqa: E402
from equitable_rwe.survival.kaplan_meier import fit_km, plot_km  # noqa: E402

st.set_page_config(page_title="PrivacyGenomic — Health Equity Dashboard", layout="wide")

st.title("Health-equity intelligence dashboard")
st.caption(
    "Synthetic, OCAP-governed cohort — see docs/data-governance/ocap-and-privacy.md. "
    "No real patient data is used anywhere in this app."
)

df = load_cohort()

col1, col2, col3 = st.columns(3)
col1.metric("Synthetic patients", f"{len(df):,}")
col2.metric("Indigenous-identity proxy share", f"{df['indigenous_identity_proxy'].mean():.1%}")
col3.metric(
    "Remote / very remote share",
    f"{df['remoteness_level'].isin(['remote', 'very_remote']).mean():.1%}",
)

st.header("Age-standardized hospitalization rate by remoteness")
st.caption(
    "Direct standardization against this cohort's own overall age distribution "
    "(see equitable_rwe.equity.age_standardization for why no external standard "
    "population is used)."
)
rates = age_standardized_hospitalization_rate_by_remoteness(df)
rate_df = pd.DataFrame(
    {
        "remoteness_level": list(rates.keys()),
        "standardized_rate": [r.rate for r in rates.values()],
        "ci_lower": [r.ci_lower for r in rates.values()],
        "ci_upper": [r.ci_upper for r in rates.values()],
    }
).sort_values("standardized_rate")
fig_rates = px.bar(
    rate_df,
    x="remoteness_level",
    y="standardized_rate",
    error_y=rate_df["ci_upper"] - rate_df["standardized_rate"],
    error_y_minus=rate_df["standardized_rate"] - rate_df["ci_lower"],
    title="Age-standardized hospitalizations/year by remoteness level",
)
st.plotly_chart(fig_rates, width="stretch")

st.header("Provincial choropleth")
st.caption(
    "Simplified non-overlapping tile-grid cartogram, not survey-accurate province "
    "outlines — see equitable_rwe.equity.choropleth for why."
)
province_values = province_choropleth_values(df)
gdf = build_province_grid(province_values, value_name="mean_hospitalizations_per_year")
html = render_choropleth_html(gdf, "mean_hospitalizations_per_year")
components.html(html, height=450)

st.header("Stage at diagnosis by Indigenous-identity proxy")
stage_table = stage_distribution_by_indigenous_proxy(df).reset_index()
stage_long = stage_table.melt(
    id_vars="indigenous_identity_proxy", var_name="stage", value_name="proportion"
)
fig_stage = px.bar(
    stage_long,
    x="stage",
    y="proportion",
    color="indigenous_identity_proxy",
    barmode="group",
    title="Stage-at-diagnosis distribution",
)
st.plotly_chart(fig_stage, width="stretch")

st.header("Income quintile by remoteness")
income_series = income_quintile_by_remoteness(df)
fig_income = px.line(
    x=income_series.index, y=income_series.values, markers=True, title="Mean income quintile"
)
fig_income.update_layout(xaxis_title="remoteness_level", yaxis_title="mean income quintile")
st.plotly_chart(fig_income, width="stretch")

st.header("Survival by Indigenous-identity proxy")
km_results = [
    fit_km(
        df.loc[df["indigenous_identity_proxy"] == flag, "time_to_event_days"],
        df.loc[df["indigenous_identity_proxy"] == flag, "event_observed"],
        label=f"indigenous_identity_proxy={flag}",
    )
    for flag in [False, True]
]
km_ax = plot_km(km_results)
st.pyplot(km_ax.figure)

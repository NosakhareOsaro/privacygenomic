import pandas as pd

from equitable_rwe.equity.choropleth import (
    GRID_POSITIONS,
    build_province_grid,
    render_choropleth_html,
)


def test_build_province_grid_includes_only_known_provinces():
    values = pd.Series({"ON": 10.0, "QC": 8.0, "NOT_A_PROVINCE": 5.0})
    gdf = build_province_grid(values)
    assert set(gdf["province"]) == {"ON", "QC"}


def test_build_province_grid_geometries_do_not_overlap():
    values = pd.Series({p: 1.0 for p in GRID_POSITIONS})
    gdf = build_province_grid(values)
    for i, geom_i in enumerate(gdf.geometry):
        for j, geom_j in enumerate(gdf.geometry):
            if i != j:
                assert geom_i.intersection(geom_j).area == 0


def test_all_thirteen_provinces_and_territories_have_grid_positions():
    assert len(GRID_POSITIONS) == 13


def test_render_choropleth_html_is_nonempty_html():
    values = pd.Series({"ON": 10.0, "QC": 8.0, "BC": 12.0})
    gdf = build_province_grid(values)
    html = render_choropleth_html(gdf, "value")
    assert "<html" in html.lower()
    assert len(html) > 500

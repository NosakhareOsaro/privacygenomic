"""A simplified provincial choropleth for the equity dashboard.

This deliberately renders Canadian provinces/territories as a
non-overlapping **tile-grid cartogram** — one unit-square grid cell per
region, arranged in its approximate compass position — rather than an
attempt at survey-accurate province outlines. Tile-grid maps are a
recognized cartographic technique for exactly this situation (e.g. NPR's
tile-grid maps of US states): they preserve relative geographic intuition
without asserting boundary precision this project has not sourced from a
verified GIS dataset. Real province boundaries are highly irregular
polygons; approximating them from memory as bounding boxes would produce
overlapping, misleading shapes, which would be worse than an honestly
simplified grid.
"""

from __future__ import annotations

import geopandas as gpd
import pandas as pd
from shapely.geometry import box

#: (column, row) grid position per province/territory, row 0 = north.
#: Layout approximates Canada's real relative compass geography (west to
#: east, north to south) without claiming boundary accuracy.
GRID_POSITIONS: dict[str, tuple[int, int]] = {
    "YT": (1, 0),
    "NT": (2, 0),
    "NU": (4, 0),
    "BC": (0, 1),
    "AB": (1, 1),
    "SK": (2, 1),
    "MB": (3, 1),
    "ON": (4, 1),
    "QC": (5, 1),
    "NL": (6, 1),
    "NB": (5, 2),
    "PE": (6, 2),
    "NS": (5, 3),
}


def build_province_grid(values: pd.Series, value_name: str = "value") -> gpd.GeoDataFrame:
    """Build a tile-grid GeoDataFrame of province values.

    Parameters
    ----------
    values : pandas.Series
        Value per province, indexed by province/territory abbreviation
        (a subset of :data:`GRID_POSITIONS`'s keys is fine).
    value_name : str, default "value"
        Column name for ``values`` in the returned GeoDataFrame.

    Returns
    -------
    geopandas.GeoDataFrame
        Columns: ``province``, ``value_name``, ``geometry`` (a unit
        square per province, in the tile-grid layout described in the
        module docstring).

    Examples
    --------
    >>> import pandas as pd
    >>> values = pd.Series({"ON": 10.0, "QC": 8.0, "BC": 12.0})
    >>> gdf = build_province_grid(values)
    >>> len(gdf)
    3
    >>> "geometry" in gdf.columns
    True
    """
    rows = []
    for province, value in values.items():
        if province not in GRID_POSITIONS:
            continue
        col, row = GRID_POSITIONS[province]
        # Flip row so north (row 0) renders at the top.
        geometry = box(col, -row - 1, col + 1, -row)
        rows.append({"province": province, value_name: value, "geometry": geometry})

    return gpd.GeoDataFrame(rows, geometry="geometry", crs="EPSG:4326")


def render_choropleth_html(gdf: gpd.GeoDataFrame, value_col: str, cmap: str = "YlOrRd") -> str:
    """Render the tile-grid choropleth to a self-contained HTML string.

    Parameters
    ----------
    gdf : geopandas.GeoDataFrame
        Output of :func:`build_province_grid`.
    value_col : str
        Column to color by.
    cmap : str, default "YlOrRd"
        Matplotlib colormap name.

    Returns
    -------
    str
        HTML for the rendered folium map (via ``GeoDataFrame.explore``).

    Examples
    --------
    >>> import pandas as pd
    >>> values = pd.Series({"ON": 10.0, "QC": 8.0})
    >>> gdf = build_province_grid(values)
    >>> html = render_choropleth_html(gdf, "value")
    >>> "<html" in html.lower() or "<div" in html.lower()
    True
    """
    m = gdf.explore(
        column=value_col,
        cmap=cmap,
        tooltip=["province", value_col],
        legend=True,
        style_kwds={"weight": 1, "color": "black"},
    )
    return m.get_root().render()

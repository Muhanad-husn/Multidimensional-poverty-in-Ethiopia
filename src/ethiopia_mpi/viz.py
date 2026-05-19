"""Visualization helpers — matplotlib + seaborn (static, publication look).

Style choice (declared in README): this is a survey-methods-and-ML project, and
the rigor signal benefits from a static publication look. The regional
choropleth, decomposition bars, and SHAP summary are static by nature.

Status: ``save_figure`` and the style setup are ready now. Session 4 adds the
regional MPI choropleth, the dimension x region decomposition bars, and the
dominant-deprivation overlay used by the hero figure.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parents[2]
FIGURES_DIR = ROOT / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams.update(
    {
        "figure.dpi": 100,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
        "font.family": "sans-serif",
    }
)


def save_figure(fig, name: str, dpi: int = 200) -> Path:
    """Save a matplotlib figure to ``figures/<name>.png``."""
    path = FIGURES_DIR / f"{name}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    return path


# ===========================================================================
# Palettes and markers (kept consistent across the three plots)
# ===========================================================================

# One color per OPHI dimension. The choropleth uses a sequential colormap for
# the MPI value; the dimension palette is used by the decomposition bars and
# the dominant-deprivation overlay markers.
DIM_PALETTE = {
    "health": "#d6604d",
    "education": "#4393c3",
    "living_standards": "#5aae61",
}
DIM_MARKERS = {
    "health": "o",
    "education": "^",
    "living_standards": "s",
}
DIM_LABELS = {
    "health": "health",
    "education": "education",
    "living_standards": "living standards",
}
CHOROPLETH_CMAP = "YlOrRd"


# ===========================================================================
# Regional MPI choropleth (+ dominant-deprivation overlay)
# ===========================================================================

def _merge_regions(reg_df, gadm, crosswalk):
    """Inner-join the per-region estimate frame onto the GADM admin-1 polygons.

    ``reg_df.domain`` is a string DHS region code (from ``svy``); the crosswalk
    maps it to the matching GADM ``NAME_1`` spelling.
    """
    xw = crosswalk.assign(domain=crosswalk["dhs_code"].astype(str))
    return (
        gadm.merge(xw, left_on="NAME_1", right_on="gadm_name", how="left")
        .merge(reg_df, on="domain", how="left")
    )


def regional_mpi_choropleth(
    reg_df: pd.DataFrame,
    gadm=None,
    crosswalk: pd.DataFrame | None = None,
    *,
    value: str = "MPI",
    overlay: pd.DataFrame | None = None,
    ax=None,
    title: str | None = "Ethiopia regional MPI",
    cmap: str = CHOROPLETH_CMAP,
):
    """Choropleth of an MPI-family statistic by Ethiopia admin-1 region.

    Parameters
    ----------
    reg_df
        Per-region estimate (e.g. from ``mpi.headcount_intensity_mpi(..., by=
        "hv024")``) carrying ``domain`` (DHS region code as string) and the
        ``value`` column to map.
    gadm, crosswalk
        Pre-loaded GADM admin-1 polygons and the DHS<->GADM crosswalk. Loaded
        lazily from :mod:`ethiopia_mpi.data` if ``None`` (the GADM call hits
        the network on first use).
    value
        Which column of ``reg_df`` to colour the map by (default ``"MPI"``).
    overlay
        Optional output of ``mpi.dominant_deprivation(..., by="hv024",
        level="dimension")`` — adds a per-region marker keyed by which
        dimension contributes most to that region's MPI.
    ax
        Existing axis to draw into; one is created if ``None``.

    Returns
    -------
    ``(fig, ax)`` so the caller can further customise or save the figure.
    """
    from . import data as _data

    if gadm is None:
        gadm = _data.load_gadm_admin1()
    if crosswalk is None:
        crosswalk = _data.region_crosswalk()

    merged = _merge_regions(reg_df, gadm, crosswalk)

    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()

    merged.plot(
        column=value,
        cmap=cmap,
        linewidth=0.5,
        edgecolor="0.4",
        ax=ax,
        legend=True,
        legend_kwds={"label": value, "shrink": 0.55},
        missing_kwds={"color": "lightgrey", "label": "no data"},
    )

    if overlay is not None:
        _draw_dominant_overlay(ax, overlay, merged)

    ax.set_axis_off()
    if title:
        ax.set_title(title, fontsize=13)
    return fig, ax


def _draw_dominant_overlay(ax, overlay: pd.DataFrame, merged) -> None:
    """Place one marker at each region centroid keyed by dominant dimension."""
    # Reproject to a metric CRS for the centroid calc (geopandas warns —
    # rightly — that centroids in lon/lat aren't true centroids). We
    # reproject back to the map's CRS afterwards so the marker coordinates
    # line up with the polygons being drawn.
    cent_gdf = merged.dropna(subset=["domain"]).copy()
    src_crs = cent_gdf.crs
    proj = cent_gdf.to_crs(epsg=32637)  # UTM 37N — Ethiopia
    cent_pts = proj.geometry.centroid.to_crs(src_crs)
    cent_gdf["x"] = cent_pts.x.values
    cent_gdf["y"] = cent_pts.y.values
    ov = overlay.merge(cent_gdf[["domain", "x", "y"]], on="domain", how="inner")

    handles = []
    for dim, marker in DIM_MARKERS.items():
        sub = ov[ov["dimension"] == dim]
        if len(sub):
            h = ax.scatter(
                sub["x"], sub["y"],
                marker=marker, s=90,
                edgecolor="black", linewidth=0.8,
                facecolor=DIM_PALETTE[dim],
                label=DIM_LABELS[dim],
                zorder=5,
            )
            handles.append(h)
    if handles:
        ax.legend(
            handles=handles, title="dominant dimension",
            loc="lower left", fontsize=8, frameon=True,
            facecolor="white", framealpha=0.9,
        )


# ===========================================================================
# Decomposition bars (dimension x region)
# ===========================================================================

def decomposition_bars(
    contrib_df: pd.DataFrame,
    *,
    value: str = "contribution",
    ax=None,
    order: str | None = "by_value",
    region_labels: pd.Series | None = None,
    title: str | None = "MPI decomposition by deprivation dimension",
):
    """Stacked horizontal bars: each region's MPI by deprivation dimension.

    Parameters
    ----------
    contrib_df
        Output of ``mpi.dimension_contributions(..., by="hv024")`` — one row
        per (region x dimension) with ``MPI`` and ``contribution`` columns.
    value
        ``"contribution"`` (default) plots the share per dimension — bars sum
        to 1 per region, surfaces compositional differences. ``"weighted_h"``
        plots absolute contributions (bars sum to MPI per region) — surfaces
        level differences as well.
    order
        ``"by_value"`` sorts regions by total bar height (MPI) descending;
        ``None`` preserves the input order.
    region_labels
        Optional mapping from DHS region code (string) to a display label
        (e.g. region name); falls back to the raw code.

    Returns
    -------
    ``(fig, ax)``.
    """
    if value not in {"contribution", "weighted_h"}:
        raise ValueError(f"value must be 'contribution' or 'weighted_h', got {value!r}")

    wide = contrib_df.pivot(index="domain", columns="dimension", values=value)
    dim_order = [d for d in DIM_PALETTE if d in wide.columns]
    wide = wide.reindex(columns=dim_order)

    if order == "by_value":
        # In contribution mode each bar sums to 1, so sort by MPI level
        # carried alongside the contributions (most-deprived region at the top).
        if value == "contribution" and "MPI" in contrib_df.columns:
            ranks = (
                contrib_df.drop_duplicates("domain")
                .set_index("domain")["MPI"]
                .reindex(wide.index)
            )
        else:
            ranks = wide.sum(axis=1)
        wide = wide.loc[ranks.sort_values(ascending=True).index]

    if region_labels is not None:
        # keep the raw code as fallback when a label is missing for a domain
        relabel = wide.index.to_series().map(region_labels)
        wide.index = relabel.fillna(wide.index.to_series()).values

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, max(3.5, 0.45 * len(wide) + 1.5)))
    else:
        fig = ax.get_figure()

    wide.plot(
        kind="barh", stacked=True, ax=ax,
        color=[DIM_PALETTE[c] for c in wide.columns],
        edgecolor="white", linewidth=0.6,
    )

    ax.set_xlabel(
        "MPI contribution share" if value == "contribution"
        else "weighted censored headcount (sums to MPI)"
    )
    ax.set_ylabel("region")
    ax.legend(
        [DIM_LABELS[c] for c in wide.columns],
        title="dimension", loc="lower right", fontsize=9, frameon=True,
    )
    if title:
        ax.set_title(title, fontsize=12)
    ax.grid(axis="y", visible=False)
    return fig, ax


# ===========================================================================
# Hero figure assembly (Session 6 will call this)
# ===========================================================================

def hero_choropleth(
    reg_df: pd.DataFrame,
    overlay: pd.DataFrame,
    *,
    gadm=None,
    crosswalk: pd.DataFrame | None = None,
    title: str = "Ethiopia: regional MPI and dominant deprivation, 2019",
):
    """Hero figure — regional MPI choropleth + dominant-deprivation overlay.

    Convenience wrapper around :func:`regional_mpi_choropleth` configured for
    the ~800x800 LinkedIn-thumbnail constraint declared in CLAUDE.md.
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    regional_mpi_choropleth(
        reg_df, gadm=gadm, crosswalk=crosswalk,
        overlay=overlay, ax=ax, title=title,
    )
    return fig, ax


# TODO Session 5: shap_summary_plot() — top 15 by mean |SHAP|

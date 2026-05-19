"""Smoke tests — package imports, the data layer, and the synthetic fixture.

Session 1 added the import/path smoke checks. Session 2 adds fixture-based
data-layer tests. Session 3 adds mpi.py coverage; Session 5 adds ml.py coverage.
"""

from pathlib import Path

import pandas as pd
import pytest

# The synthetic DHS-shaped fixture (see tests/fixtures/make_fixture.py).
FIXTURE_RAW = Path(__file__).parent / "fixtures" / "raw"


# ---------------------------------------------------------------------------
# Session 1 — imports & paths
# ---------------------------------------------------------------------------

def test_module_imports():
    """The project package imports without error."""
    import ethiopia_mpi  # noqa: F401


def test_submodule_imports():
    """All project submodules import without error."""
    from ethiopia_mpi import (  # noqa: F401
        data,
        diagnostics,
        ml,
        mpi,
        viz,
    )


def test_data_dirs_exist():
    """The data directories exist (auto-created on import of data.py)."""
    from ethiopia_mpi.data import EXTERNAL_DIR, PROCESSED_DIR, RAW_DIR

    assert RAW_DIR.exists()
    assert PROCESSED_DIR.exists()
    assert EXTERNAL_DIR.exists()


def test_diagnostics_import():
    """The diagnostics helpers import."""
    from ethiopia_mpi.diagnostics import (  # noqa: F401
        before_after,
        compare_alternatives,
        distribution_compare,
        distribution_summary,
        missingness_pattern,
        missingness_summary,
    )


def test_ophi_weights_sum_to_one():
    """The 10 OPHI indicator weights sum to 1 (3 equally weighted dimensions)."""
    from ethiopia_mpi.mpi import OPHI_WEIGHTS

    assert len(OPHI_WEIGHTS) == 10
    assert abs(sum(OPHI_WEIGHTS.values()) - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# Session 2 — synthetic fixture & data layer
# ---------------------------------------------------------------------------

def test_fixture_exists():
    """The synthetic fixture was generated and is committed."""
    from ethiopia_mpi.data import DHS_RECODES

    assert FIXTURE_RAW.is_dir(), (
        f"fixture missing — run `python tests/fixtures/make_fixture.py` ({FIXTURE_RAW})"
    )
    for rel in DHS_RECODES.values():
        assert (FIXTURE_RAW / rel).exists(), f"missing fixture recode: {rel}"


@pytest.mark.parametrize("recode", ["HR", "PR", "IR", "BR", "KR"])
def test_load_dhs_recode(recode):
    """Every recode loads from the fixture as a non-empty DataFrame."""
    from ethiopia_mpi.data import load_dhs_recode

    df = load_dhs_recode(recode, raw_dir=FIXTURE_RAW)
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0


def test_load_dhs_recode_survey_design_columns():
    """Recodes carry the survey-design columns (cluster / strata / weight).

    HR/PR use the hv* family; IR/BR/KR use the v* family.
    """
    from ethiopia_mpi.data import load_dhs_recode

    hr = load_dhs_recode("HR", raw_dir=FIXTURE_RAW)
    for col in ("hv001", "hv005", "hv021", "hv022", "hv023", "hv024"):
        assert col in hr.columns, f"HR missing survey-design column {col}"

    ir = load_dhs_recode("IR", raw_dir=FIXTURE_RAW)
    for col in ("v001", "v005", "v021", "v022", "v023", "v024"):
        assert col in ir.columns, f"IR missing survey-design column {col}"


def test_load_dhs_recode_columns_subset():
    """The `columns` argument loads only the requested columns."""
    from ethiopia_mpi.data import load_dhs_recode

    df = load_dhs_recode("HR", raw_dir=FIXTURE_RAW, columns=["hv001", "hv024"])
    assert list(df.columns) == ["hv001", "hv024"]


def test_load_dhs_recode_rejects_unknown():
    """An unknown recode name raises ValueError."""
    from ethiopia_mpi.data import load_dhs_recode

    with pytest.raises(ValueError):
        load_dhs_recode("ZZ", raw_dir=FIXTURE_RAW)


def test_load_dhs_recode_missing_file():
    """A missing recode file raises FileNotFoundError with the access hint."""
    from ethiopia_mpi.data import load_dhs_recode

    with pytest.raises(FileNotFoundError):
        load_dhs_recode("HR", raw_dir=Path(__file__).parent / "no_such_dir")


def test_region_crosswalk():
    """The crosswalk covers all 11 DHS Ethiopia regions with GADM names."""
    from ethiopia_mpi.data import region_crosswalk

    xw = region_crosswalk()
    assert list(xw.columns) == ["dhs_code", "dhs_region", "gadm_name"]
    assert len(xw) == 11
    assert set(xw["dhs_code"]) == set(range(1, 12))
    assert xw["gadm_name"].notna().all()


def test_fixture_recodes_share_household_keys():
    """Persons (PR) join back to households (HR) on (cluster, household)."""
    from ethiopia_mpi.data import load_dhs_recode

    hr = load_dhs_recode("HR", raw_dir=FIXTURE_RAW)
    pr = load_dhs_recode("PR", raw_dir=FIXTURE_RAW)
    hh_keys = set(zip(hr["hv001"], hr["hv002"]))
    pr_keys = set(zip(pr["hv001"], pr["hv002"]))
    assert pr_keys <= hh_keys, "PR households not a subset of HR households"


# ---------------------------------------------------------------------------
# Session 2 — diagnostics helpers exercised on real-shaped data
# ---------------------------------------------------------------------------

def test_diagnostics_on_fixture():
    """The diagnostic helpers run end to end on a loaded recode."""
    from ethiopia_mpi.data import load_dhs_recode
    from ethiopia_mpi.diagnostics import (
        distribution_summary,
        missingness_pattern,
        missingness_summary,
    )

    hr = load_dhs_recode("HR", raw_dir=FIXTURE_RAW)

    miss = missingness_summary(hr)
    assert {"column", "n_missing", "pct_missing"} <= set(miss.columns)
    # the fixture injects missingness into a few living-standards columns
    assert miss["n_missing"].sum() > 0

    pat = missingness_pattern(hr)
    assert "pattern" in pat.columns and len(pat) > 0

    summ = distribution_summary(hr["hv009"])
    assert "skew" in summ.index and "n_missing" in summ.index


def test_compare_alternatives_smoke():
    """compare_alternatives produces one row per candidate strategy."""
    from ethiopia_mpi.data import load_dhs_recode
    from ethiopia_mpi.diagnostics import compare_alternatives

    hr = load_dhs_recode("HR", raw_dir=FIXTURE_RAW)
    out = compare_alternatives(
        hr["hv201"],
        {
            "drop_missing": lambda s: s.dropna(),
            "mode_impute": lambda s: s.fillna(s.mode().iloc[0]),
        },
    )
    assert len(out) == 2


# ---------------------------------------------------------------------------
# Session 3 — MPI engine: indicators, score, survey-weighted H / A / MPI
# ---------------------------------------------------------------------------

def _load_mpi_recodes():
    """Load the three recodes the MPI engine needs from the fixture."""
    from ethiopia_mpi import mpi
    from ethiopia_mpi.data import load_dhs_recode

    hr = load_dhs_recode("HR", raw_dir=FIXTURE_RAW, columns=mpi.HR_COLS)
    pr = load_dhs_recode("PR", raw_dir=FIXTURE_RAW, columns=mpi.PR_COLS)
    br = load_dhs_recode("BR", raw_dir=FIXTURE_RAW, columns=mpi.BR_COLS)
    return hr, pr, br


def test_ophi_dimensions_partition_indicators():
    """The 3 OPHI dimensions partition the 10 indicators exactly."""
    from ethiopia_mpi.mpi import DIMENSIONS, INDICATORS

    flat = [ind for inds in DIMENSIONS.values() for ind in inds]
    assert sorted(flat) == sorted(INDICATORS)
    assert len(flat) == 10


def test_build_indicators_shape_and_values():
    """build_indicators yields the 10 indicator flags, one row per household."""
    from ethiopia_mpi.mpi import INDICATORS, build_indicators

    hr, pr, br = _load_mpi_recodes()
    ind = build_indicators(hr, pr, br)

    assert len(ind) == len(hr)
    assert list(ind.index.names) == ["hv001", "hv002"]
    for col in INDICATORS:
        assert col in ind.columns
        # every value is a deprivation flag (0 / 1) or NaN — nothing else
        vals = set(ind[col].dropna().unique())
        assert vals <= {0.0, 1.0}, f"{col} has non-flag values {vals}"


def test_no_eligible_member_rule_keeps_indicator_observed():
    """OPHI 'no eligible member' rule: child mortality is never NaN.

    A household with no births is non-deprived (0), not missing — so the
    child-mortality indicator has no NaN at all on the fixture.
    """
    from ethiopia_mpi.mpi import build_indicators

    hr, pr, br = _load_mpi_recodes()
    ind = build_indicators(hr, pr, br)
    assert ind["child_mortality"].notna().all()


def test_deprivation_score_missing_policies():
    """`drop` blanks the score when an indicator is missing; `available-case`
    renormalises and keeps it."""
    from ethiopia_mpi.mpi import build_indicators, deprivation_score

    hr, pr, br = _load_mpi_recodes()
    ind = build_indicators(hr, pr, br)

    dropped = deprivation_score(ind, missing_policy="drop")
    avail = deprivation_score(ind, missing_policy="available-case")

    # households with a missing indicator -> NaN score under `drop`
    incomplete = dropped["n_missing_ind"] > 0
    assert incomplete.any(), "fixture should inject some missingness"
    assert dropped.loc[incomplete, "score"].isna().all()
    assert dropped.loc[~incomplete, "score"].notna().all()

    # available-case keeps every household with at least one observed indicator
    assert avail["score"].notna().all()
    assert (avail["score"].between(0.0, 1.0)).all()


def test_deprivation_score_rejects_unknown_policy():
    from ethiopia_mpi.mpi import build_indicators, deprivation_score

    hr, pr, br = _load_mpi_recodes()
    with pytest.raises(ValueError):
        deprivation_score(build_indicators(hr, pr, br), missing_policy="bogus")


def test_assign_poverty_flag_and_censored_score():
    """poor is a {0,1,NaN} flag at cutoff k; censored score is score x poor."""
    import numpy as np

    from ethiopia_mpi.mpi import (
        K_DEFAULT,
        assign_poverty,
        build_indicators,
        deprivation_score,
    )

    hr, pr, br = _load_mpi_recodes()
    scored = deprivation_score(build_indicators(hr, pr, br), missing_policy="drop")
    out = assign_poverty(scored, k=K_DEFAULT)

    assert set(out["poor"].dropna().unique()) <= {0.0, 1.0}
    # poor exactly where score >= k
    ok = out["score"].notna()
    assert (out.loc[ok, "poor"] == (out.loc[ok, "score"] >= K_DEFAULT)).all()
    # censored score: equals score for the poor, 0 for the non-poor
    poor = ok & (out["poor"] == 1.0)
    assert np.allclose(out.loc[poor, "censored_score"], out.loc[poor, "score"])
    assert (out.loc[ok & (out["poor"] == 0.0), "censored_score"] == 0.0).all()


def test_headcount_intensity_mpi_national():
    """National survey-weighted estimate: one row, MPI = H x A, SEs positive."""
    from ethiopia_mpi.mpi import build_household_mpi, headcount_intensity_mpi

    hr, pr, br = _load_mpi_recodes()
    hh = build_household_mpi(hr, pr, br)
    nat = headcount_intensity_mpi(hh)

    assert len(nat) == 1
    row = nat.iloc[0]
    assert row["domain"] == "national"
    for col in ("H", "A", "MPI"):
        assert 0.0 <= row[col] <= 1.0
        assert row[f"{col}_se"] > 0
    # MPI is identically the headcount times the intensity
    assert abs(row["MPI"] - row["H"] * row["A"]) < 1e-6


def test_headcount_intensity_mpi_by_region():
    """Per-region estimation returns one row per region present in the data."""
    from ethiopia_mpi.mpi import build_household_mpi, headcount_intensity_mpi

    hr, pr, br = _load_mpi_recodes()
    hh = build_household_mpi(hr, pr, br)
    reg = headcount_intensity_mpi(hh, by="hv024")

    assert len(reg) == hh["hv024"].nunique()
    assert (reg["MPI"].between(0.0, 1.0)).all()
    # every domain's complete-case household count is carried through
    assert reg["n"].notna().all() and (reg["n"] > 0).all()


# ---------------------------------------------------------------------------
# Session 4 — decomposition (by indicator / dimension / region) and viz
# ---------------------------------------------------------------------------

def _build_hh():
    from ethiopia_mpi.mpi import build_household_mpi

    hr, pr, br = _load_mpi_recodes()
    return build_household_mpi(hr, pr, br)


def test_censored_headcounts_sum_to_mpi():
    """Sum of weighted censored headcounts (national) equals MPI."""
    from ethiopia_mpi.mpi import (
        censored_headcounts,
        headcount_intensity_mpi,
    )

    hh = _build_hh()
    ch = censored_headcounts(hh)
    assert len(ch) == 10  # one row per OPHI indicator
    assert set(ch["domain"]) == {"national"}
    # contributions sum to 1 by construction (per domain)
    assert abs(ch["contribution"].sum() - 1.0) < 1e-9
    # sum_j w_j * h_j ≡ MPI (numerically the same as the svy direct estimate)
    nat = headcount_intensity_mpi(hh).iloc[0]
    assert abs(ch["w_h"].sum() - nat["MPI"]) < 1e-6


def test_dimension_contributions_per_region():
    """Per-region dimension contributions sum to 1 within each region."""
    from ethiopia_mpi.mpi import DIMENSIONS, dimension_contributions

    hh = _build_hh()
    dc = dimension_contributions(hh, by="hv024")

    # three dimensions x N regions
    n_regions = hh["hv024"].nunique()
    assert len(dc) == 3 * n_regions
    assert set(dc["dimension"].astype(str)) == set(DIMENSIONS)

    # contributions sum to 1 per domain (within rounding)
    sums = dc.groupby("domain")["contribution"].sum()
    assert ((sums - 1.0).abs() < 1e-9).all()


def test_dominant_deprivation_one_row_per_region():
    """Dominant dimension picks one of the three dimensions per region."""
    from ethiopia_mpi.mpi import DIMENSIONS, dominant_deprivation

    hh = _build_hh()
    dom = dominant_deprivation(hh, by="hv024", level="dimension")
    assert len(dom) == hh["hv024"].nunique()
    assert set(dom["dimension"].astype(str)) <= set(DIMENSIONS)
    # the dominant share is always at least 1/3 (three dimensions, ties allowed)
    assert (dom["contribution"] >= 1 / 3 - 1e-9).all()


def test_dominant_deprivation_indicator_level():
    """level='indicator' drills down to the single most-contributing indicator."""
    from ethiopia_mpi.mpi import INDICATORS, dominant_deprivation

    hh = _build_hh()
    dom_i = dominant_deprivation(hh, by="hv024", level="indicator")
    assert set(dom_i["indicator"]) <= set(INDICATORS)
    assert len(dom_i) == hh["hv024"].nunique()


def test_decomposition_bars_renders():
    """decomposition_bars produces a figure with the expected legend entries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from ethiopia_mpi.mpi import dimension_contributions
    from ethiopia_mpi.viz import DIM_LABELS, decomposition_bars

    hh = _build_hh()
    dc = dimension_contributions(hh, by="hv024")
    fig, ax = decomposition_bars(dc)
    try:
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert set(labels) == set(DIM_LABELS.values())
        # bars sum to 1 per region in contribution mode (one container per dim)
        assert len(ax.containers) == 3
    finally:
        plt.close(fig)


def test_choropleth_renders_with_synthetic_gdf():
    """regional_mpi_choropleth runs end-to-end with a tiny synthetic GADM gdf.

    Avoids the live GADM download; this test isolates the merge + plotting
    logic from the network. The live GADM crosswalk is verified separately
    by the session smoke run (see IMPLEMENTATION_PLAN.md Session 4 handoff).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    import geopandas as gpd
    import pandas as pd
    from shapely.geometry import box

    from ethiopia_mpi.mpi import (
        dimension_contributions,
        dominant_deprivation,
        headcount_intensity_mpi,
    )
    from ethiopia_mpi.viz import regional_mpi_choropleth

    hh = _build_hh()
    reg = headcount_intensity_mpi(hh, by="hv024")
    dom = dominant_deprivation(hh, by="hv024")

    # one square polygon per region present in the fixture; NAME_1 spellings
    # must match the crosswalk's gadm_name column.
    from ethiopia_mpi.data import region_crosswalk

    xw = region_crosswalk()
    code_to_gadm = dict(zip(xw["dhs_code"].astype(str), xw["gadm_name"]))
    polys = []
    for i, code in enumerate(sorted(reg["domain"].unique())):
        # small adjacent unit squares — geometry doesn't need to be Ethiopia-shaped
        polys.append({"NAME_1": code_to_gadm[code], "geometry": box(i, 0, i + 1, 1)})
    gadm = gpd.GeoDataFrame(polys, crs="EPSG:4326")

    fig, ax = regional_mpi_choropleth(reg, gadm=gadm, crosswalk=xw, overlay=dom)
    try:
        # the choropleth's colorbar + the overlay legend should both be present
        assert ax.get_legend() is not None
        assert ax.collections, "no patches drawn"
    finally:
        plt.close(fig)

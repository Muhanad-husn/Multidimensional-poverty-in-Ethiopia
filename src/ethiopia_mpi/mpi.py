"""OPHI Multidimensional Poverty Index — indicator construction and estimation.

Implements the 10 OPHI Global MPI indicators (health / education / living
standards), the weighted deprivation score, the poverty cutoff at k = 1/3, and
survey-weighted headcount H, intensity A, and MPI = H x A — using DHS clusters,
strata, and sampling weights via ``svy``. Decomposition by region and dimension
is added in Session 4.

SURVEY DESIGN IS NON-NEGOTIABLE: every aggregate goes through the survey-weighted
path (weight hv005/1e6, strata hv022, clusters/PSU hv021). See CLAUDE.md.

Indicator construction (OPHI Global MPI methodology)
----------------------------------------------------
Each of the 10 indicators is built per household as a deprivation flag:
1.0 = deprived, 0.0 = not deprived, NaN = cannot be determined (missing item).

OPHI's "no eligible member" rule is applied: a household with no member
eligible for an indicator (e.g. no under-5 child for the nutrition indicator)
is counted as *non-deprived* (0.0) on it, not missing. NaN is reserved for the
case where an eligible member exists but the underlying item is missing.

Status: implemented in Session 3 (indicators + survey-weighted H/A/MPI).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# OPHI Global MPI: 3 dimensions, 10 indicators. Each dimension weighted equally
# (1/3); indicators within a dimension share their dimension's weight.
OPHI_WEIGHTS = {
    # Health (1/3) — 1/6 each
    "nutrition": 1 / 6,
    "child_mortality": 1 / 6,
    # Education (1/3) — 1/6 each
    "years_schooling": 1 / 6,
    "school_attendance": 1 / 6,
    # Living standards (1/3) — 1/18 each
    "cooking_fuel": 1 / 18,
    "sanitation": 1 / 18,
    "drinking_water": 1 / 18,
    "electricity": 1 / 18,
    "housing": 1 / 18,
    "assets": 1 / 18,
}

# The 10 indicator names, and their grouping into the 3 OPHI dimensions.
INDICATORS = list(OPHI_WEIGHTS)
DIMENSIONS = {
    "health": ["nutrition", "child_mortality"],
    "education": ["years_schooling", "school_attendance"],
    "living_standards": [
        "cooking_fuel",
        "sanitation",
        "drinking_water",
        "electricity",
        "housing",
        "assets",
    ],
}

# OPHI default multidimensional poverty cutoff.
K_DEFAULT = 1 / 3

# Survey-design columns (household recode `hv*` family — see Decision #2 in the
# implementation plan). hv001 = cluster, hv021 = PSU, hv022 = stratum,
# hv005 = sampling weight (x 1e6), hv024 = region, hv025 = urban/rural.
DESIGN = {
    "cluster": "hv001",
    "psu": "hv021",
    "stratum": "hv022",
    "weight": "hv005",
    "region": "hv024",
}
HH_KEYS = ["hv001", "hv002"]

# Columns each recode must supply for indicator construction. Passed to
# `data.load_dhs_recode(..., columns=...)` so the very wide real recodes
# (HR ~1,400 cols, PR ~3,000) load only what MPI needs.
HR_COLS = [
    "hv001", "hv002", "hv005", "hv021", "hv022", "hv023", "hv024", "hv025",
    "hv201", "hv205", "hv225", "hv206", "hv213", "hv214", "hv215", "hv226",
    "hv207", "hv208", "hv209", "hv210", "hv211", "hv212", "hv221", "hv243a",
]
PR_COLS = ["hv001", "hv002", "hv104", "hv105", "hv108", "hv121", "hc70", "hc72", "ha40"]
BR_COLS = ["v001", "v002", "b5"]

# --- DHS value-code sets (Ethiopia 2019; see CLAUDE.md "OPHI methodology") ----

# hv226 cooking fuel: solid fuels => deprived. 1-5 are clean (electricity, LPG,
# gas, biogas, kerosene); 95 = no food cooked, 96 = other => non-deprived.
_SOLID_FUEL = {6, 7, 8, 9, 10, 11}
_CLEAN_FUEL = {1, 2, 3, 4, 5, 95, 96}

# hv201 drinking water: improved sources (DHS standard classification).
_IMPROVED_WATER = {11, 12, 13, 14, 21, 31, 41, 51, 71, 72, 91, 92}

# hv205 toilet: improved sanitation facility types (deprived also if shared).
_IMPROVED_SANITATION = {11, 12, 13, 15, 21, 22, 41}


# ===========================================================================
# Indicator construction
# ===========================================================================

def _clean(s: pd.Series, *, lo: float | None = None, hi: float | None = None) -> pd.Series:
    """Blank out-of-range / special-code values to NaN (DHS uses 98/99, 9996+)."""
    s = s.astype("float64")
    if lo is not None:
        s = s.where(s >= lo)
    if hi is not None:
        s = s.where(s <= hi)
    return s


def _hh_flag(n_eligible, n_positive, n_valid) -> pd.Series:
    """Resolve a member-aggregated indicator to a household deprivation flag.

    OPHI logic: no eligible member -> 0 (non-deprived); any positive -> 1;
    at least one valid (non-missing) member observation, none positive -> 0;
    eligible members exist but all observations missing -> NaN.
    """
    flag = pd.Series(np.nan, index=n_eligible.index, dtype="float64")
    flag[n_eligible.eq(0)] = 0.0
    flag[n_valid.gt(0)] = 0.0
    flag[n_positive.gt(0)] = 1.0
    return flag


def _ind_nutrition(pr: pd.DataFrame) -> pd.Series:
    """Deprived if any under-5 child or woman 15-49 is undernourished.

    Child: height-for-age (hc70) or weight-for-age (hc72) z-score < -2 SD
    (DHS stores z-scores x 100). Woman: BMI (ha40, x 100) < 18.5.
    """
    p = pr[["hv001", "hv002", "hv104", "hv105"]].copy()
    haz = _clean(pr["hc70"], lo=-600, hi=600)
    waz = _clean(pr["hc72"], lo=-600, hi=600)
    bmi = _clean(pr["ha40"], lo=1200, hi=6000)

    is_child = pr["hv105"] < 5
    is_woman = (pr["hv104"] == 2) & pr["hv105"].between(15, 49)

    child_valid = is_child & (haz.notna() | waz.notna())
    child_under = (haz < -200) | (waz < -200)
    woman_valid = is_woman & bmi.notna()
    woman_under = bmi < 1850

    status = pd.Series(np.nan, index=pr.index, dtype="float64")
    status[child_valid] = child_under[child_valid].astype("float64")
    status[woman_valid] = woman_under[woman_valid].astype("float64")

    p["elig"] = is_child | is_woman
    p["pos"] = status.eq(1.0)
    p["valid"] = status.notna()
    g = p.groupby(HH_KEYS)
    return _hh_flag(g["elig"].sum(), g["pos"].sum(), g["valid"].sum())


def _ind_years_schooling(pr: pd.DataFrame) -> pd.Series:
    """Deprived if no household member has completed >= 6 years of schooling."""
    sch = _clean(pr["hv108"], hi=89)  # 98 = don't know, 99 = missing
    g = pd.DataFrame({"hv001": pr["hv001"], "hv002": pr["hv002"], "sch": sch})
    max_sch = g.groupby(HH_KEYS)["sch"].max()
    flag = (max_sch < 6).astype("float64")
    flag[max_sch.isna()] = np.nan  # no member with a valid schooling record
    return flag


def _ind_school_attendance(pr: pd.DataFrame) -> pd.Series:
    """Deprived if any child of school age (7-14) is not attending school.

    Ethiopia primary cycle is grades 1-8 starting at age 7 -> school age 7-14.
    hv121: 0 not attending, 1/2 attending; 8/9 special -> missing.
    """
    p = pr[["hv001", "hv002"]].copy()
    att = _clean(pr["hv121"], hi=7)  # keep 0/1/2; drop 8 (dk) / 9 (missing)
    school_age = pr["hv105"].between(7, 14)

    status = pd.Series(np.nan, index=pr.index, dtype="float64")
    seen = school_age & att.notna()
    status[seen] = att[seen].eq(0).astype("float64")  # 1 = not attending

    p["elig"] = school_age
    p["pos"] = status.eq(1.0)
    p["valid"] = status.notna()
    g = p.groupby(HH_KEYS)
    return _hh_flag(g["elig"].sum(), g["pos"].sum(), g["valid"].sum())


def _ind_child_mortality(br: pd.DataFrame, hh_index: pd.MultiIndex) -> pd.Series:
    """Deprived if any child born to a household member has died (b5 == 0)."""
    dead = br.assign(_dead=br["b5"].eq(0))
    by_hh = dead.groupby(["v001", "v002"])["_dead"].any()
    by_hh.index = by_hh.index.set_names(HH_KEYS)
    flag = by_hh.reindex(hh_index, fill_value=False).astype("float64")
    return flag  # households with no births reindex to 0 (non-deprived)


def _ind_cooking_fuel(hr: pd.DataFrame) -> pd.Series:
    fuel = hr["hv226"]
    flag = pd.Series(np.nan, index=hr.index, dtype="float64")
    flag[fuel.isin(_CLEAN_FUEL)] = 0.0
    flag[fuel.isin(_SOLID_FUEL)] = 1.0
    return flag


def _ind_sanitation(hr: pd.DataFrame) -> pd.Series:
    toilet = _clean(hr["hv205"], hi=96)
    shared = hr["hv225"]
    improved_unshared = toilet.isin(_IMPROVED_SANITATION) & shared.ne(1)
    flag = (~improved_unshared).astype("float64")
    flag[toilet.isna()] = np.nan
    return flag


def _ind_drinking_water(hr: pd.DataFrame) -> pd.Series:
    water = _clean(hr["hv201"], hi=96)
    flag = (~water.isin(_IMPROVED_WATER)).astype("float64")
    flag[water.isna()] = np.nan
    return flag


def _ind_electricity(hr: pd.DataFrame) -> pd.Series:
    elec = _clean(hr["hv206"], hi=1)  # 0/1; 9 = missing
    flag = elec.eq(0).astype("float64")
    flag[elec.isna()] = np.nan
    return flag


def _ind_housing(hr: pd.DataFrame) -> pd.Series:
    """Deprived if floor is natural, or roof/wall is natural or rudimentary.

    DHS material codes: 1x = natural, 2x = rudimentary, 3x+ = finished.
    """
    floor = _clean(hr["hv213"], hi=96) // 10
    wall = _clean(hr["hv214"], hi=96) // 10
    roof = _clean(hr["hv215"], hi=96) // 10
    bad = floor.eq(1) | roof.isin([1, 2]) | wall.isin([1, 2])
    all_valid = floor.notna() & roof.notna() & wall.notna()
    return pd.Series(
        np.where(bad, 1.0, np.where(all_valid, 0.0, np.nan)),
        index=hr.index, dtype="float64",
    )


def _ind_assets(hr: pd.DataFrame) -> pd.Series:
    """Deprived if the household owns <= 1 small asset and no car/truck.

    Small assets: radio, TV, refrigerator, bicycle, motorcycle, telephone
    (land-line or mobile). Missing asset items are treated as not-owned —
    DHS asset questions are answered for essentially all households.
    """
    small = hr[["hv207", "hv208", "hv209", "hv210", "hv211"]].fillna(0).gt(0)
    phone = hr["hv221"].fillna(0).gt(0) | hr["hv243a"].fillna(0).gt(0)
    count = small.sum(axis=1) + phone.astype(int)
    car = hr["hv212"].fillna(0).gt(0)
    return ((count <= 1) & ~car).astype("float64")


def build_indicators(
    hr: pd.DataFrame, pr: pd.DataFrame, br: pd.DataFrame
) -> pd.DataFrame:
    """Construct the 10 OPHI deprivation indicators, one row per household.

    Parameters
    ----------
    hr, pr, br
        DHS Household, Household-Member, and Births recodes (the columns in
        ``HR_COLS`` / ``PR_COLS`` / ``BR_COLS`` must be present).

    Returns
    -------
    DataFrame indexed by ``(hv001, hv002)`` carrying the survey-design columns
    and the 10 indicator flags (1.0 deprived / 0.0 not / NaN undetermined).
    """
    hh = hr.set_index(HH_KEYS).sort_index()
    design_cols = ["hv005", "hv021", "hv022", "hv023", "hv024", "hv025"]
    out = hh[[c for c in design_cols if c in hh.columns]].copy()

    # Living-standards indicators are household-level (HR).
    ls = hh  # aligned on the same index as `out`
    out["cooking_fuel"] = _ind_cooking_fuel(ls).values
    out["sanitation"] = _ind_sanitation(ls).values
    out["drinking_water"] = _ind_drinking_water(ls).values
    out["electricity"] = _ind_electricity(ls).values
    out["housing"] = _ind_housing(ls).values
    out["assets"] = _ind_assets(ls).values

    # Health / education indicators are aggregated up from members (PR) or
    # births (BR); reindex onto the household frame.
    out["nutrition"] = _ind_nutrition(pr).reindex(out.index)
    out["years_schooling"] = _ind_years_schooling(pr).reindex(out.index)
    out["school_attendance"] = _ind_school_attendance(pr).reindex(out.index)
    out["child_mortality"] = _ind_child_mortality(br, out.index)

    return out


# ===========================================================================
# Deprivation score, poverty flag
# ===========================================================================

def deprivation_score(
    indicators: pd.DataFrame, missing_policy: str = "drop"
) -> pd.DataFrame:
    """Add the weighted deprivation score to an indicators frame.

    Parameters
    ----------
    indicators
        Output of :func:`build_indicators` (carries the 10 indicator columns).
    missing_policy
        ``"drop"`` — the score is NaN if any indicator is missing
        (complete-case; OPHI's standard treatment). ``"available-case"`` —
        the score is the deprivation share over the *non-missing* indicators,
        with the OPHI weights renormalised to those indicators.

    Returns
    -------
    A copy of ``indicators`` with added columns ``score`` and
    ``n_missing_ind`` (count of missing indicators for the household).
    """
    if missing_policy not in ("drop", "available-case"):
        raise ValueError(f"unknown missing_policy {missing_policy!r}")

    out = indicators.copy()
    weights = pd.Series(OPHI_WEIGHTS)
    dep = out[INDICATORS]

    weighted = dep.mul(weights, axis=1)  # NaN where the indicator is NaN
    out["n_missing_ind"] = dep.isna().sum(axis=1).astype(int)

    if missing_policy == "drop":
        score = weighted.sum(axis=1)
        score[out["n_missing_ind"].gt(0)] = np.nan
    else:  # available-case: renormalise weights over observed indicators
        observed_weight = dep.notna().mul(weights, axis=1).sum(axis=1)
        score = weighted.sum(axis=1).div(observed_weight)
        score[observed_weight.eq(0)] = np.nan

    out["score"] = score
    return out


def assign_poverty(scored: pd.DataFrame, k: float = K_DEFAULT) -> pd.DataFrame:
    """Add the poverty flag and censored score at cutoff ``k``.

    A household is multidimensionally poor when its deprivation score is at
    least ``k`` (OPHI default k = 1/3). ``censored_score`` is the score for the
    poor and 0 for the non-poor — its survey-weighted mean is MPI itself.
    """
    out = scored.copy()
    score = out["score"]
    poor = (score >= k).astype("float64")
    poor[score.isna()] = np.nan
    out["poor"] = poor
    out["censored_score"] = (score * poor).astype("float64")
    return out


def build_household_mpi(
    hr: pd.DataFrame,
    pr: pd.DataFrame,
    br: pd.DataFrame,
    k: float = K_DEFAULT,
    missing_policy: str = "drop",
) -> pd.DataFrame:
    """Full household-level MPI pipeline: indicators -> score -> poverty flag.

    Returns one row per household with the design columns, the 10 indicators,
    ``score``, ``n_missing_ind``, ``poor`` and ``censored_score``. Ready to be
    handed to :func:`headcount_intensity_mpi` for survey-weighted estimation.
    """
    ind = build_indicators(hr, pr, br)
    scored = deprivation_score(ind, missing_policy=missing_policy)
    return assign_poverty(scored, k=k).reset_index()


# ===========================================================================
# Survey-weighted estimation (svy: strata + clusters + sampling weights)
# ===========================================================================

def _make_sample(df: pd.DataFrame, value_cols: list[str], by: str | None = None):
    """Build an ``svy.Sample`` with the DHS stratified-cluster design.

    The sampling weight is hv005 / 1e6 (DHS scales weights by 1e6). Strata and
    PSU identifiers are cast to strings — ``svy`` factorises them and rejects
    floating-point design columns.
    """
    import polars as pl
    import svy

    cols = ["hv005", DESIGN["psu"], DESIGN["stratum"], *value_cols]
    if by is not None and by not in cols:
        cols.append(by)

    pdf = df[cols].copy()
    pdf["_wgt"] = pdf["hv005"] / 1e6
    pdf = pdf.drop(columns="hv005")

    pl_df = pl.from_pandas(pdf)
    cast_cols = [DESIGN["psu"], DESIGN["stratum"]]
    if by is not None:
        cast_cols.append(by)
    pl_df = pl_df.with_columns(
        pl.col(cast_cols).cast(pl.Int64, strict=False).cast(pl.Utf8)
    )

    design = svy.Design(
        stratum=DESIGN["stratum"], wgt="_wgt", psu=DESIGN["psu"]
    )
    return svy.Sample(data=pl_df, design=design)


def _estimate_rows(estimate, label: str) -> pd.DataFrame:
    """Flatten an ``svy.Estimate`` to a tidy frame keyed by the `by` level."""
    rows = []
    for d in estimate.to_dicts():
        level = d["by_level"]
        rows.append(
            {
                "domain": level[0] if level is not None else "national",
                f"{label}": d["est"],
                f"{label}_se": d["se"],
                f"{label}_lci": d["lci"],
                f"{label}_uci": d["uci"],
            }
        )
    return pd.DataFrame(rows)


def headcount_intensity_mpi(
    df: pd.DataFrame, by: str | None = None
) -> pd.DataFrame:
    """Survey-weighted headcount H, intensity A, and MPI with standard errors.

    Parameters
    ----------
    df
        Household-level frame from :func:`build_household_mpi` (must carry
        ``poor``, ``score``, ``censored_score`` and the design columns).
    by
        Optional grouping column (e.g. ``"hv024"`` for per-region estimates).
        ``None`` returns the single national estimate.

    Returns
    -------
    DataFrame with one row per domain and columns ``n`` (unweighted
    complete-case households), ``H`` / ``A`` / ``MPI`` each with ``_se`` /
    ``_lci`` / ``_uci``. MPI is estimated directly as the survey-weighted mean
    of the censored score (identically H x A) so its SE reflects the design.
    """
    import polars as pl

    value_cols = ["poor", "score", "censored_score"]
    sample = _make_sample(df, value_cols, by=by)
    est = sample.estimation

    h = _estimate_rows(est.mean("poor", by=by, drop_nulls=True), "H")
    mpi = _estimate_rows(
        est.mean("censored_score", by=by, drop_nulls=True), "MPI"
    )
    a = _estimate_rows(
        est.mean("score", by=by, where=pl.col("poor") == 1.0, drop_nulls=True),
        "A",
    )

    out = h.merge(mpi, on="domain", how="outer").merge(a, on="domain", how="outer")

    # Unweighted complete-case household count per domain (denominator clarity).
    complete = df.dropna(subset=["poor"])
    if by is None:
        counts = pd.DataFrame({"domain": ["national"], "n": [len(complete)]})
    else:
        counts = (
            complete.groupby(by).size().rename("n").reset_index()
            .assign(domain=lambda d: d[by].astype("Int64").astype(str))
            [["domain", "n"]]
        )
    out = out.merge(counts, on="domain", how="left")

    order = ["domain", "n", "H", "H_se", "H_lci", "H_uci",
             "A", "A_se", "A_lci", "A_uci",
             "MPI", "MPI_se", "MPI_lci", "MPI_uci"]
    return out[[c for c in order if c in out.columns]]


# ===========================================================================
# Decomposition (Session 4): by deprivation indicator and dimension
# ===========================================================================

def _dim_of(indicator: str) -> str:
    """Return the OPHI dimension that contains ``indicator``."""
    for dim, items in DIMENSIONS.items():
        if indicator in items:
            return dim
    raise KeyError(indicator)


def censored_headcounts(
    df: pd.DataFrame, by: str | None = None
) -> pd.DataFrame:
    """Survey-weighted censored headcount h_j per OPHI indicator (and domain).

    For each indicator j the censored headcount is

        h_j = E[poor x deprived_j | design]

    — the survey-weighted share of households that are MPI-poor *and*
    deprived in indicator j. Summing the weighted censored headcounts
    recovers MPI: ``MPI = sum_j w_j * h_j``. The contribution of indicator j
    to MPI is therefore ``w_j * h_j / MPI``.

    Parameters
    ----------
    df
        Household-level frame from :func:`build_household_mpi`.
    by
        Optional grouping column (e.g. ``"hv024"`` for per-region decomposition).

    Returns
    -------
    Tidy DataFrame with columns ``domain, dimension, indicator, weight, h,
    h_se, h_lci, h_uci, w_h, contribution`` — one row per (domain x indicator).
    """
    work = df.copy()
    cens_cols = []
    for ind in INDICATORS:
        col = f"_c_{ind}"
        work[col] = (work["poor"] * work[ind]).astype("float64")
        cens_cols.append(col)

    sample = _make_sample(work, cens_cols, by=by)
    est = sample.estimation

    pieces = []
    for ind, col in zip(INDICATORS, cens_cols):
        e = _estimate_rows(est.mean(col, by=by, drop_nulls=True), "h")
        e["indicator"] = ind
        e["dimension"] = _dim_of(ind)
        e["weight"] = OPHI_WEIGHTS[ind]
        pieces.append(e)

    out = pd.concat(pieces, ignore_index=True)
    out["w_h"] = out["weight"] * out["h"]

    totals = out.groupby("domain")["w_h"].sum().rename("_mpi_total")
    out = out.merge(totals, on="domain")
    out["contribution"] = out["w_h"].div(out["_mpi_total"]).where(
        out["_mpi_total"] > 0, 0.0
    )

    cols = ["domain", "dimension", "indicator", "weight",
            "h", "h_se", "h_lci", "h_uci", "w_h", "contribution"]
    return out[cols]


def dimension_contributions(
    df: pd.DataFrame, by: str | None = None
) -> pd.DataFrame:
    """Aggregate censored headcounts to dimension-level MPI contributions.

    Returns one row per (domain x dimension) carrying ``weighted_h`` (the
    sum of ``w_j * h_j`` over the dimension's indicators), the domain's
    ``MPI`` (sum across all dimensions), and the dimension's ``contribution``
    share (sums to 1 per domain).
    """
    detail = censored_headcounts(df, by=by)
    agg = (
        detail.groupby(["domain", "dimension"], as_index=False)
        .agg(weighted_h=("w_h", "sum"))
    )
    mpi_per_domain = detail.groupby("domain")["w_h"].sum().rename("MPI")
    agg = agg.merge(mpi_per_domain, on="domain", how="left")
    agg["contribution"] = agg["weighted_h"].div(agg["MPI"]).where(
        agg["MPI"] > 0, 0.0
    )

    # canonical dimension order for downstream plotting
    dim_order = pd.Categorical(
        agg["dimension"], categories=list(DIMENSIONS), ordered=True
    )
    agg = agg.assign(dimension=dim_order).sort_values(["domain", "dimension"])
    return agg.reset_index(drop=True)


def dominant_deprivation(
    df: pd.DataFrame, by: str = "hv024", level: str = "dimension"
) -> pd.DataFrame:
    """Per-domain dominant deprivation (largest MPI contribution).

    ``level="dimension"`` returns the OPHI dimension with the largest share;
    ``level="indicator"`` drills down to the single largest of the 10
    indicators. Used by the hero figure to overlay a per-region symbol.
    """
    if level == "dimension":
        d = dimension_contributions(df, by=by)
        idx = d.groupby("domain")["contribution"].idxmax()
        return d.loc[idx, ["domain", "dimension", "contribution"]].reset_index(drop=True)
    if level == "indicator":
        d = censored_headcounts(df, by=by)
        idx = d.groupby("domain")["contribution"].idxmax()
        return d.loc[idx, ["domain", "dimension", "indicator", "contribution"]].reset_index(drop=True)
    raise ValueError(f"unknown level {level!r}; expected 'dimension' or 'indicator'")

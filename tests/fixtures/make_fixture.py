"""Generate a synthetic DHS-shaped fixture for testing the MPI pipeline.

The fixture mimics the structure of the DHS Ethiopia 2019 recodes — same column
names, dtypes, value-code conventions, and survey-design columns — but the values
are fully synthetic (random). It exists so the data layer, ``mpi.py`` and ``ml.py``
can be unit-tested without the licensed DHS microdata.

Run ``python tests/fixtures/make_fixture.py`` to regenerate. Output: five ``.DTA``
files under ``tests/fixtures/raw/ET<rec>81DT/``, matching ``data.DHS_RECODES`` so
``load_dhs_recode(recode, raw_dir=<fixtures>/raw)`` reads them like the real thing.

Scale: 24 clusters across 6 regions x urban/rural strata, 300 households,
~1,400 persons. Small enough to commit; large enough that survey-weighted
estimation and cluster-grouped CV are exercised meaningfully.

Survey design (faithful to DHS):
  - HR / PR recodes use the ``hv*`` household-recode column family.
  - IR / BR / KR recodes use the ``v*`` woman/birth column family.
  - cluster/PSU = hv021/v021, strata = hv022/v022, weight = hv005/v005 (x1e6).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pyreadstat

FIXTURE_RAW = Path(__file__).resolve().parent / "raw"

N_CLUSTERS = 24
N_HH = 300
SEED = 42

# Subset of the 11 DHS Ethiopia regions used by the fixture (codes match real data).
FIXTURE_REGIONS = [1, 2, 3, 4, 7, 10]  # tigray, afar, amhara, oromia, snnpr, addis ababa


def _write(df: pd.DataFrame, recode: str) -> Path:
    """Write one recode to tests/fixtures/raw/ET<rec>81DT/ET<rec>81FL.DTA."""
    folder = FIXTURE_RAW / f"ET{recode}81DT"
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"ET{recode}81FL.DTA"
    pyreadstat.write_dta(df, str(path))
    return path


def _inject_missing(df: pd.DataFrame, cols: list[str], rate: float, rng) -> None:
    """Set a random ``rate`` share of values to NaN in-place — gives the
    missing-indicator decision block (Session 3) something real to diagnose."""
    for c in cols:
        mask = rng.random(len(df)) < rate
        df.loc[mask, c] = np.nan


def build_fixture() -> dict[str, Path]:
    rng = np.random.default_rng(SEED)

    # --- clusters: each belongs to one region and one urban/rural class ------
    clu_region = rng.choice(FIXTURE_REGIONS, size=N_CLUSTERS)
    clu_urban = rng.integers(1, 3, size=N_CLUSTERS)  # 1 = urban, 2 = rural

    # --- households ----------------------------------------------------------
    hh_clu = rng.integers(0, N_CLUSTERS, size=N_HH)
    hv001 = hh_clu + 1  # cluster id, 1-indexed
    region = clu_region[hh_clu]
    urban = clu_urban[hh_clu]
    # household number, sequential within cluster
    hv002 = np.zeros(N_HH, dtype=int)
    for c in range(1, N_CLUSTERS + 1):
        idx = np.where(hv001 == c)[0]
        hv002[idx] = np.arange(1, len(idx) + 1)

    n_members = rng.integers(1, 11, size=N_HH)  # 1..10 members

    hr = pd.DataFrame(
        {
            "hhid": [f"{c:>8} {h:>3}" for c, h in zip(hv001, hv002)],
            "hv001": hv001,
            "hv002": hv002,
            "hv005": rng.integers(200_000, 3_000_000, size=N_HH),  # weight x1e6
            "hv021": hv001,                       # PSU == cluster
            "hv022": region * 10 + urban,         # strata: region x urban/rural
            "hv023": region * 10 + urban,
            "hv024": region,                      # region
            "hv025": urban,                       # type of residence
            "hv009": n_members,                   # number of household members
            # --- living standards (codes follow DHS conventions) -------------
            "hv201": rng.choice([11, 13, 21, 31, 32, 42, 43, 96], size=N_HH),  # water
            "hv205": rng.choice([11, 12, 14, 15, 21, 23, 31, 41], size=N_HH),  # toilet
            "hv225": rng.integers(0, 2, size=N_HH),                            # shared toilet
            "hv206": rng.integers(0, 2, size=N_HH),                            # electricity
            "hv213": rng.choice([11, 12, 21, 31, 33, 34, 35], size=N_HH),      # floor
            "hv214": rng.choice([11, 12, 21, 22, 31, 33, 34], size=N_HH),      # wall
            "hv215": rng.choice([11, 12, 21, 31, 33, 34, 36], size=N_HH),      # roof
            "hv226": rng.choice([1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 95], size=N_HH),  # cooking fuel
            "hv207": rng.integers(0, 2, size=N_HH),   # radio
            "hv208": rng.integers(0, 2, size=N_HH),   # television
            "hv209": rng.integers(0, 2, size=N_HH),   # refrigerator
            "hv210": rng.integers(0, 2, size=N_HH),   # bicycle
            "hv211": rng.integers(0, 2, size=N_HH),   # motorcycle/scooter
            "hv212": rng.integers(0, 2, size=N_HH),   # car/truck
            "hv221": rng.integers(0, 2, size=N_HH),   # telephone (land-line)
            "hv243a": rng.integers(0, 2, size=N_HH),  # mobile telephone
            "hv246": rng.integers(0, 2, size=N_HH),   # owns livestock
            "hv040": rng.integers(500, 3500, size=N_HH).astype(float),  # altitude (m)
        }
    )
    # Wealth index — correlated with asset/electricity ownership so the ML
    # feature-importance overlay has signal on the fixture. Real DHS supplies
    # hv270 (quintile 1-5) and hv271 (continuous factor score x 100_000).
    _asset_load = (
        hr["hv206"].astype(float)             # electricity
        + hr["hv208"].astype(float)           # tv
        + hr["hv209"].astype(float)           # fridge
        + hr["hv212"].astype(float)           # car
        + hr["hv243a"].astype(float)          # mobile
    )
    score = _asset_load + rng.normal(0, 0.5, size=N_HH)
    hr["hv271"] = ((score - score.mean()) / score.std() * 100_000).round().astype(float)
    hr["hv270"] = pd.qcut(hr["hv271"], 5, labels=False, duplicates="drop").astype(float) + 1
    hr = hr.astype({c: "float64" for c in hr.columns if c != "hhid"})
    _inject_missing(hr, ["hv201", "hv205", "hv226", "hv213"], rate=0.04, rng=rng)

    # --- persons (PR): n_members rows per household --------------------------
    rows = []
    for i in range(N_HH):
        n = n_members[i]
        for line in range(1, n + 1):
            age = int(rng.integers(0, 85))
            sex = int(rng.integers(1, 3))  # 1 male, 2 female
            rows.append(
                {
                    "hv001": int(hv001[i]),
                    "hv002": int(hv002[i]),
                    "hvidx": line,                       # person line number
                    "hv005": float(hr.loc[i, "hv005"]),
                    "hv021": int(hv001[i]),
                    "hv022": int(region[i] * 10 + urban[i]),
                    "hv023": int(region[i] * 10 + urban[i]),
                    "hv024": int(region[i]),
                    "hv025": int(urban[i]),
                    "hv101": 1 if line == 1 else int(rng.choice([2, 3, 4, 5, 6])),  # relationship
                    "hv104": sex,                        # sex
                    "hv105": age,                        # age in years
                    # years of education (single years); missing for under-5
                    "hv108": np.nan if age < 5 else float(rng.integers(0, 16)),
                    "hv109": np.nan if age < 5 else float(rng.integers(0, 6)),
                    # school attendance 0/1; only meaningful for school-age 6-18
                    "hv121": float(rng.integers(0, 2)) if 6 <= age <= 18 else 0.0,
                    # child anthropometry z-scores x100; only for under-5
                    "hc70": float(rng.integers(-400, 300)) if age < 5 else np.nan,
                    "hc72": float(rng.integers(-400, 300)) if age < 5 else np.nan,
                    # woman BMI x100; only for women 15-49
                    "ha40": float(rng.integers(1500, 3200))
                    if (sex == 2 and 15 <= age <= 49)
                    else np.nan,
                }
            )
    pr = pd.DataFrame(rows)

    # --- women (IR): female persons aged 15-49 -------------------------------
    women = pr[(pr["hv104"] == 2) & pr["hv105"].between(15, 49)].copy()
    ir = pd.DataFrame(
        {
            "v001": women["hv001"].astype(int).values,
            "v002": women["hv002"].astype(int).values,
            "v003": women["hvidx"].astype(int).values,   # woman's line number
            "v005": women["hv005"].values,
            "v021": women["hv021"].astype(int).values,
            "v022": women["hv022"].astype(int).values,
            "v023": women["hv023"].astype(int).values,
            "v024": women["hv024"].astype(int).values,
            "v013": pd.cut(women["hv105"], bins=[14, 19, 24, 29, 34, 39, 44, 49],
                           labels=False).astype(float).values + 1,  # 5-year age group
            "v206": rng.integers(0, 3, size=len(women)).astype(float),   # sons who have died
            "v207": rng.integers(0, 3, size=len(women)).astype(float),   # daughters who have died
            "v445": women["ha40"].values,                                # BMI x100
        }
    )

    # --- births (BR): 0-6 births per woman -----------------------------------
    brows = []
    for _, w in ir.iterrows():
        for bidx in range(1, int(rng.integers(0, 7)) + 1):
            alive = int(rng.random() > 0.12)  # ~12% of births not surviving
            brows.append(
                {
                    "v001": int(w["v001"]), "v002": int(w["v002"]), "v003": int(w["v003"]),
                    "v005": float(w["v005"]),
                    "v021": int(w["v021"]), "v022": int(w["v022"]),
                    "v023": int(w["v023"]), "v024": int(w["v024"]),
                    "bidx": bidx,
                    "b2": int(rng.integers(2000, 2020)),                # year of birth
                    "b5": float(alive),                                  # child is alive
                    "b7": np.nan if alive else float(rng.integers(0, 59)),  # age at death (months)
                }
            )
    br = pd.DataFrame(brows)

    # --- children under 5 (KR): living children with anthropometry -----------
    kids = pr[(pr["hv105"] < 5)].copy()
    kr = pd.DataFrame(
        {
            "v001": kids["hv001"].astype(int).values,
            "v002": kids["hv002"].astype(int).values,
            "v003": kids["hvidx"].astype(int).values,
            "v005": kids["hv005"].values,
            "v021": kids["hv021"].astype(int).values,
            "v022": kids["hv022"].astype(int).values,
            "v023": kids["hv023"].astype(int).values,
            "v024": kids["hv024"].astype(int).values,
            "b5": 1.0,                                   # KR holds living children
            "hw70": kids["hc70"].values,                 # height-for-age z x100
            "hw72": kids["hc72"].values,                 # weight-for-age z x100
        }
    )

    paths = {
        "HR": _write(hr, "HR"),
        "PR": _write(pr.astype({c: "float64" for c in pr.columns}), "PR"),
        "IR": _write(ir, "IR"),
        "BR": _write(br, "BR"),
        "KR": _write(kr, "KR"),
    }
    return paths


if __name__ == "__main__":
    paths = build_fixture()
    for rec, p in paths.items():
        df = pd.read_stata(p)
        print(f"{rec}: {len(df):>5} rows x {df.shape[1]:>2} cols  ->  {p.relative_to(FIXTURE_RAW.parent)}")

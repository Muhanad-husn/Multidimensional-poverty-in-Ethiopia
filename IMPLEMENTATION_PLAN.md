# Multidimensional Poverty in Ethiopia (MPI) — Multi-Session Implementation Plan

> **Created:** 2026-05-19
> **Source plan:** IMPLEMENTATION_PLAN.md (this file, restructured in place)
> **Total sessions:** 7
> **Estimated total effort:** 7 sessions (Sessions 1–2 runnable now; 3–7 gated on raw DHS data)

## Overview

Reproduce the OPHI Multidimensional Poverty Index for Ethiopia (DHS 2019) with correct
survey design (clusters + strata + sampling weights via `svy`), decompose MPI by region
and deprivation dimension, add a cluster-respecting ML feature-importance overlay
(RandomForest/XGBoost + SHAP), and produce a regional MPI choropleth hero figure. The
deliverable is a reproducible notebook with five-part decision blocks.

## Environment (verified 2026-05-19)

- conda env `portfolio`, Python 3.14.4.
- Verified installed: pandas 3.0.3, numpy 2.4.5, scikit-learn 1.8.0, geopandas 1.1.3,
  statsmodels 0.14.6, matplotlib 3.10.9, seaborn 0.13.2, jupyter, pyarrow,
  xgboost 3.2.0, shap 0.51.0, pyreadstat 1.3.5, numba 0.65.1.
- **Survey-design library: `svy` 0.18.1** — maintained successor to the archived
  `samplics` (same author). Documented deviation from CLAUDE.md decision #3 (which
  named `samplics`); record it in the README decisions table.

## Hard constraints (from CLAUDE.md — non-negotiable)

1. Every estimate carries sampling weights (`v005`/1e6), strata (`v022`/`v023`),
   clusters/PSUs (`v021`). Estimation goes through `svy`.
2. ML train/test split uses `GroupKFold` grouped on cluster ID (`v001`). Never
   `train_test_split` without grouping.
3. ML feature set excludes the 10 OPHI indicators themselves (target leakage).
4. SHAP aggregation across households respects sampling weights for national claims.
5. Every non-trivial data-processing decision documented in five-part form in
   `notebooks/02_main.ipynb` (problem / diagnostic / options / decision / sensitivity).
6. Raw DHS microdata is gitignored and never committed (license forbids redistribution).

## Data dependency gate

**DATA GATE CLEARED (2026-05-19).** Raw DHS 2019 files are present in `data/raw/`,
each recode in its own `ET<rec>81DT/` subfolder with the `.DTA` file inside:
PR (40,659 persons), HR (8,663 households), IR (8,885 women), BR (23,007 births),
KR (5,753 children — Children's Recode; preferred source for the child-nutrition
indicator). The Fieldworker recode (`ETFW81DT`) is also present but unused. All seven
sessions can now run. Module code is still unit-tested against a synthetic
DHS-shaped fixture so the pipeline is verifiable independent of the licensed data.

## Session Dependency Graph

```
Session 1 (Scaffold)
    └─> Session 2 (Synthetic fixture + data layer)
            └─> [DATA GATE: raw DHS files in data/raw/]
                    └─> Session 3 (MPI engine)
                            └─> Session 4 (Decomposition + viz)
                                    └─> Session 5 (ML + SHAP)
                                            └─> Session 6 (Notebook assembly + hero)
                                                    └─> Session 7 (Finalize + repro)
```

Strictly linear: each session depends only on its immediate predecessor.

---

## Session 1: Scaffold the repository

**Objective:** Create the full project skeleton so the package imports cleanly and all
directories exist.
**Inputs:** `CLAUDE.md`, `README.md`, this plan.
**Outputs:** `pyproject.toml`, `src/ethiopia_mpi/` package skeleton, directory tree,
`.gitignore`, `data/raw/README.md`, `notebooks/NOTEBOOK_STRUCTURE.md`.
**Depends on:** None.

### Context for resumption

The project folder currently contains only `CLAUDE.md`, `README.md`, and this plan.
The conda env `portfolio` (Python 3.14.4) already has every dependency installed and
verified — no installs needed. This session produces the empty skeleton only; no
analysis logic. Read `CLAUDE.md` "Files orientation" section first for the canonical
file layout.

### Steps

1. Write `pyproject.toml` with project metadata and optional-dependency groups
   `[viz]`, `[survey]` (svy), `[ml]` (xgboost, shap); install target
   `pip install -e ".[viz,survey,ml]"`. Pin to the verified versions.
2. Create `src/ethiopia_mpi/` with `__init__.py` and importable-but-empty modules:
   `data.py`, `mpi.py`, `ml.py`, `viz.py`, `diagnostics.py` (each with a module
   docstring and `TODO` markers).
3. Create directory tree: `notebooks/`, `notebooks/_scratch/`, `data/raw/`,
   `data/processed/`, `figures/`, `tests/`.
4. Write `.gitignore` — raw DHS files, `notebooks/_scratch/`, data artifacts that
   could carry PII, `__pycache__`, checkpoints.
5. Write `data/raw/README.md` — required DHS files (HR Household Recode, PR Household
   Member Recode, IR Individual Recode, BR Births Recode), the DHS registration/access
   workflow, and an access-date placeholder.
6. Write `notebooks/NOTEBOOK_STRUCTURE.md` — the canonical five-part decision-block
   template (problem / diagnostic / options / decision / sensitivity).
7. Run `pip install -e ".[viz,survey,ml]"` in env `portfolio` and confirm
   `import ethiopia_mpi` works.

### Completion criteria

- [ ] `pyproject.toml` exists and installs cleanly with all three extras.
- [ ] `import ethiopia_mpi` and submodule imports succeed under env `portfolio`.
- [ ] Full directory tree exists.
- [ ] `.gitignore`, `data/raw/README.md`, `notebooks/NOTEBOOK_STRUCTURE.md` written.

### Handoff notes

Completed 2026-05-19. Scaffold based on the monorepo `_template/`. Package
`ethiopia_mpi` installed editable into env `portfolio` with extras `viz,survey,ml,dev`;
all 5 smoke tests pass.
- `pyproject.toml`: extras are `viz` (matplotlib, seaborn, geopandas, shapely, pyogrio,
  mapclassify), `survey` (svy), `ml` (sklearn, xgboost, shap). `pyreadstat` is a core
  dependency. `requires-python>=3.11` (env is 3.14).
- `data.py` already has working path constants, `read/write_processed`, `download_file`
  (resolves relative paths against `EXTERNAL_DIR` — for the GADM download), and a
  `DHS_RECODES` dict mapping recode -> `data/raw/` path. `load_dhs_recode`,
  `load_gadm_admin1`, `region_crosswalk` are `NotImplementedError` stubs for Session 2.
- `mpi.py` defines `OPHI_WEIGHTS` (10 indicators summing to 1) and `K_DEFAULT=1/3`.
- `ml.py` defines `CLUSTER_COL="v001"`.
- `viz.py` activates the matplotlib+seaborn style and `save_figure`.
- `data/external/` was added (not in CLAUDE.md's file list) as the GADM boundary home,
  following the template convention; gitignored.
- `.gitignore` keeps `data/raw/README.md` tracked via an explicit `!` exception.

---

## Session 2: Synthetic fixture, data layer, diagnostics

**Objective:** Build a synthetic DHS-shaped fixture, the data-loading layer, the
diagnostic helpers, and a smoke test — all data-independent.
**Inputs:** Session 1 skeleton; `CLAUDE.md` survey-design and OPHI-methodology notes.
**Outputs:** `tests/fixtures/` synthetic dataset, `data.py`, `diagnostics.py`,
`tests/test_smoke.py`.
**Depends on:** Session 1.

### Context for resumption

The repo skeleton exists and the package imports. No real DHS data is present yet.
This session makes the pipeline testable without real data: a tiny synthetic dataset
shaped like DHS recodes, plus the loaders and diagnostic helpers. Read `CLAUDE.md`
"Survey design is non-negotiable" and "OPHI methodology reproduction" sections, and
`notebooks/NOTEBOOK_STRUCTURE.md`, before starting.

### Steps

1. Build `tests/fixtures/` — a synthetic DHS-shaped dataset (~300 households,
   ~20 clusters across multiple strata/regions) with the columns the 10 OPHI
   indicators need, plus `v001` (cluster), `v005` (weight), `v021` (PSU),
   `v022`/`v023` (strata). Save as small files in the recode shape.
2. Implement `data.py` — DHS recode loader (`pyreadstat` / `pandas.read_stata`),
   GADM admin-1 polygon loader, and a DHS-region ↔ GADM-admin1 crosswalk scaffold
   (~11 regions; values confirmed against real data later).
3. Implement `diagnostics.py` — `missingness_summary`, `missingness_pattern`,
   `distribution_summary`, `distribution_compare`, `before_after`,
   `compare_alternatives` (port from the `_template/` reference described in CLAUDE.md).
4. Write `tests/test_smoke.py` — loads the fixture and exercises the data layer end
   to end; will be extended in later sessions as `mpi.py`/`ml.py` are built.
5. Run the test suite under env `portfolio`.

### Completion criteria

- [ ] Synthetic fixture exists and loads via `data.py`.
- [ ] `data.py` and `diagnostics.py` implemented and importable.
- [ ] `tests/test_smoke.py` passes.
- [ ] GADM crosswalk scaffold present (placeholder values acceptable here).

### Handoff notes

_(filled in during execution)_

---

### Handoff notes

Completed 2026-05-19. The pipeline is now testable without licensed data: 19 smoke
tests pass under env `portfolio`.

- **Synthetic fixture.** `tests/fixtures/make_fixture.py` generates five synthetic
  `.DTA` recodes under `tests/fixtures/raw/ET<rec>81DT/` (committed — synthetic, no
  license issue). Deterministic (seed 42). Scale: 24 clusters, 6 regions (codes
  1,2,3,4,7,10), 300 households, 1,654 persons, 341 women, 1,050 births, 114 under-5
  children. Regenerate with `python tests/fixtures/make_fixture.py`. Missingness is
  injected into a few HR living-standards columns so the Session 3 missing-indicator
  decision block has real signal.
- **Real-data column reality (verified against the actual `.DTA` metadata).** HR/PR
  use the `hv*` column family; IR/BR/KR use `v*`. Region codes `hv024`/`v024`
  confirmed: 1 tigray, 2 afar, 3 amhara, 4 oromia, 5 somali, 6 benishangul-gumuz,
  7 snnpr, 8 gambela, 9 harari, 10 addis ababa, 11 dire dawa. All 10-indicator
  columns confirmed present — HR: `hv201/hv205/hv225/hv206/hv213/hv214/hv215/hv226`,
  assets `hv207-hv212/hv221/hv243a/hv246`; PR: `hv101/hv105/hv108/hv109/hv121` +
  anthropometry `hc70/hc72/ha40`; IR: `v206/v207/v445`; BR/KR: `b5/b7`, KR `hw70/hw72`.
- **`data.py`.** `load_dhs_recode(recode, raw_dir=None, columns=None)` — reads via
  `pyreadstat.read_dta`, returns numeric codes (no categorical→label conversion),
  `columns` subsets (real recodes are 1,400–4,400 cols wide — use it). Raises
  `ValueError` on unknown recode, `FileNotFoundError` (with access hint) on missing
  file. `load_gadm_admin1()` downloads the GADM 4.1 Ethiopia GeoPackage and reads
  layer `ADM_ADM_1` — **not exercised by tests (no network); verify in Session 4.**
  `region_crosswalk()` returns an 11-row df (`dhs_code`/`dhs_region`/`gadm_name`);
  `gadm_name` is a scaffold to verify against the real GADM layer in Session 4.
- **`diagnostics.py`** was already fully implemented in Session 1 — verified all six
  helpers import and run on the fixture; no changes needed.
- **Survey-design column naming (logged as Decision #2).** Household-level design
  columns are `hv001` (cluster), `hv021` (PSU), `hv022`/`hv023` (strata), `hv005`
  (weight). `ml.py` currently hard-codes `CLUSTER_COL = "v001"` — for the
  per-household MPI target this must become `"hv001"` in Session 5. `mpi.py`
  survey estimation (Session 3) must use the `hv*` family for HR/PR-based aggregates.
- Nutrition recode choice (deferred to Session 3): PR carries both child
  anthropometry (`hc70/hc72`) and woman BMI (`ha40`); KR carries `hw70/hw72`; IR
  carries `v445`. Session 3 picks the source per OPHI methodology.

---

## Session 3: MPI engine (OPHI indicators + survey-weighted estimation)

**Objective:** Construct the 10 OPHI indicators, the weighted deprivation score, the
poverty flag at k = 1/3, and survey-weighted H / A / MPI via `svy`.
**Inputs:** Sessions 1–2; **raw DHS files in `data/raw/`**; OPHI methodology docs.
**Outputs:** `mpi.py`; first three decision blocks drafted for `notebooks/02_main.ipynb`.
**Depends on:** Session 2 **and the data gate**.

### Context for resumption

Skeleton, data layer, fixture, and diagnostics exist. **Verify raw DHS files are
present in `data/raw/` before starting** — if not, stop and tell the user. This
session builds the MPI calculation core. Read `CLAUDE.md` "OPHI methodology
reproduction" and "Survey design is non-negotiable" sections, and `data/raw/README.md`.
Confirm real DHS column names against the OPHI definitions.

### Steps

1. Implement the 10 OPHI indicators in `mpi.py`: years of schooling, school
   attendance, child mortality, nutrition, cooking fuel, sanitation, drinking water,
   electricity, housing, assets — each with its OPHI threshold and a missing-data rule.
   Nutrition + child mortality need the BR recode joined to the household.
2. Apply OPHI dimension weights; compute the weighted deprivation score; flag
   multidimensionally poor at k = 1/3.
3. Implement survey-weighted headcount H, intensity A, MPI = H × A via `svy` (strata +
   clusters + weights), national and per-region, with standard errors.
4. Extend `tests/test_smoke.py` to cover `mpi.py` on the fixture.
5. Draft three five-part decision blocks for `02_main.ipynb`: missing-indicator-item
   treatment, poverty cutoff k, survey-design implementation (svy vs samplics note).

### Completion criteria

- [x] All 10 OPHI indicators constructed; thresholds + missing rules documented.
- [x] Survey-weighted H, A, MPI computed nationally and per region with SEs.
- [x] `mpi.py` tests pass on the fixture.
- [x] Three decision blocks drafted.

### Handoff notes

Completed 2026-05-19. `mpi.py` is the full MPI engine; 27 smoke tests pass under
env `portfolio` (19 prior + 8 new for `mpi.py`).

- **`svy` 0.18.1 API (verified by experiment).** Polars-based. Build a design
  with `svy.Design(stratum=, wgt=, psu=)`, a sample with
  `svy.Sample(data=<pl.DataFrame>, design=design)`, then estimate via the
  `sample.estimation` property — `.mean(y, by=, where=, drop_nulls=)`,
  `.prop(...)`, `.total(...)`. Results: `Estimate.to_dicts()` →
  `est/se/cv/lci/uci/by_level`. **Gotcha:** stratum/PSU/`by` columns must not be
  float — `svy` factorises them and rejects `f64`. `_make_sample` casts them
  `Int64 → Utf8`. Console-printing an `Estimate` on Windows fails on box-drawing
  glyphs (cp1252) — use `.to_dicts()` / `.to_polars()`, never bare `print`.
- **`mpi.py` public API.** `build_indicators(hr, pr, br)` → household-indexed
  frame with the 10 flags (1=deprived/0/NaN) + design cols. `deprivation_score(
  ind, missing_policy=)`, `assign_poverty(scored, k=)`, and the one-call
  pipeline `build_household_mpi(hr, pr, br, k=, missing_policy=)`. Estimation:
  `headcount_intensity_mpi(df, by=None)` → tidy frame with `n, H, A, MPI` each
  with `_se/_lci/_uci`; `by="hv024"` for per-region. MPI is estimated directly
  as the survey-weighted mean of `censored_score` (≡ H×A; verified to 1e-6).
- **Column lists** `HR_COLS / PR_COLS / BR_COLS` are exported — pass them to
  `load_dhs_recode(..., columns=)` so the wide real recodes load lean. `KR` and
  `IR` are **not used** by the MPI engine (see Decision #3).
- **`DIMENSIONS`** dict (health/education/living_standards → indicator lists)
  added for Session 4 decomposition; `INDICATORS` is the flat 10-name list.
- **Missing policy.** `"drop"` (default, OPHI complete-case) blanks the score
  if any indicator is NaN; `"available-case"` renormalises OPHI weights over
  observed indicators. On the fixture: drop keeps 254/300 households,
  available-case 300/300. Session 7 robustness compares them.
- **Three decision blocks drafted** in `notebooks/_drafts/session3_decision_blocks.md`
  (missing-indicator treatment / cutoff k / survey-design implementation), in
  five-part form, with runnable diagnostic code. Session 6 pastes them into
  `02_main.ipynb`. `notebooks/_drafts/` is tracked (not gitignored).
- **Doc reconciliation for Session 7:** `data/raw/README.md` still lists KR for
  "child nutrition" and IR for "woman's nutrition" — the engine sources both
  from PR (Decision #3). Update that "Used for" column during the Session 7
  README pass. KR/BR/IR stay required downloads (BR used; KR/IR reserved for ML).

---

## Session 4: Decomposition + visualization

**Objective:** Decompose MPI by region and by deprivation dimension; build the
choropleth and decomposition visualizations.
**Inputs:** Session 3 (`mpi.py`); GADM polygons via `data.py`.
**Outputs:** decomposition functions in `mpi.py`; `viz.py`.
**Depends on:** Session 3.

### Context for resumption

`mpi.py` computes survey-weighted MPI nationally and per region. This session adds
dimensional decomposition and the plotting layer. Read `CLAUDE.md` "Visual style" and
the hero-figure constraint (readable at ~800×800, ~11 regions).

### Steps

1. Implement survey-weighted decomposition of MPI by region and by deprivation
   dimension (censored headcounts / dimensional contribution shares) in `mpi.py`.
2. Implement `viz.py`: regional MPI choropleth (matplotlib + geopandas), dimension ×
   region decomposition stacked bars, dominant-deprivation overlay helper.
3. Verify the GADM ↔ DHS-region crosswalk against real region names; fix placeholders.
4. Sanity-check figures render and are legible at thumbnail size.

### Completion criteria

- [x] MPI decomposed by region and dimension, survey-weighted.
- [x] `viz.py` produces choropleth + decomposition bars.
- [x] Crosswalk verified against real DHS region codes.

### Handoff notes

Completed 2026-05-19. Decomposition + visualization layer added; 33 smoke tests
pass under env `portfolio` (27 prior + 6 new for Session 4).

- **`mpi.py` decomposition API.** Three new public functions, all
  survey-weighted via `_make_sample` (same `svy` design path as
  `headcount_intensity_mpi`):
  - `censored_headcounts(df, by=None)` — tidy frame, one row per
    `(domain x indicator)` with `weight`, `h`, `h_se`, `h_lci`, `h_uci`,
    `w_h = weight*h`, and `contribution = w_h / sum_k w_k h_k`. The OPHI
    identity `MPI = sum_j w_j * h_j` holds: on the fixture, sum of `w_h`
    nationally is 0.254355, identical to `headcount_intensity_mpi(hh)`'s MPI.
  - `dimension_contributions(df, by=None)` — aggregates to (domain x
    dimension), carries `weighted_h`, `MPI`, `contribution`. Contributions
    sum to 1 per domain; dimensions are stored as an ordered categorical
    in canonical order (health / education / living_standards).
  - `dominant_deprivation(df, by="hv024", level="dimension"|"indicator")`
    — picks the largest contributor per domain. Used by the hero overlay.
- **`viz.py` Session-4 additions** — all matplotlib + seaborn, geopandas
  imported lazily inside `regional_mpi_choropleth`. Module-level constants
  `DIM_PALETTE` / `DIM_MARKERS` / `DIM_LABELS` keep the three dimensions
  visually consistent across the choropleth overlay and the decomposition
  bars (health = red circle, education = blue triangle, living_standards =
  green square).
  - `regional_mpi_choropleth(reg_df, gadm=None, crosswalk=None, *, value=
    "MPI", overlay=None, ax=None, ...)` — joins the per-region estimate via
    the crosswalk onto GADM polygons; missing regions render light grey.
    `overlay` accepts the output of `dominant_deprivation(level="dimension")`
    and draws one marker per region at the polygon centroid.
  - `decomposition_bars(contrib_df, *, value="contribution", region_labels=
    None, order="by_value", ax=None, ...)` — stacked horizontal bars,
    dimension legend, region label column overridable. In contribution mode
    bars sum to 1 per region, so the sort uses the `MPI` column carried in
    `contrib_df` (highest-MPI region at the top); in `weighted_h` mode it
    falls back to row sums (= MPI per region) which is equivalent.
  - `hero_choropleth(reg_df, overlay, ...)` — convenience wrapper sized
    for the ~800x800 LinkedIn-thumbnail constraint from CLAUDE.md;
    Session 6 calls this to produce `figures/hero.png`.
- **Centroid placement gotcha.** GADM ships EPSG:4326 (lon/lat); computing
  centroids in a geographic CRS warns and is wrong. `_draw_dominant_overlay`
  reprojects to EPSG:32637 (UTM 37N — Ethiopia), takes the centroid, and
  projects the points back to the map's CRS so they align with the polygons.
- **Crosswalk verification (live GADM).** Downloaded the GADM 4.1 ETH
  GeoPackage, read layer `ADM_ADM_1` — 11 features. The Session-2 scaffold
  matched 10 of 11 spellings exactly; the only fix was SNNPR: GADM's
  `NAME_1` is `"Southern Nations, Nationalities"` (truncated, no trailing
  "and Peoples"). Applied in `data.py`; the inline scaffold-note comment is
  rewritten accordingly. The GADM file is cached under `data/external/`.
- **Test isolation.** `test_choropleth_renders_with_synthetic_gdf` builds a
  tiny synthetic `gpd.GeoDataFrame` (one unit square per fixture region with
  the correct `NAME_1` spelling) so the plotting path is unit-tested
  without the network. The live download is exercised once per session
  (regenerate crosswalk-fix only if GADM updates).
- **Smoke render (live data path) on the fixture** — choropleth + sorted
  decomposition bars produced the expected output (Tigray highest MPI on
  the fixture sample with codes 1,2,3,4,7,10; Afar lowest; living-standards
  the dominant dimension in five of six fixture regions, health dominant
  in Tigray and Oromia). Real DHS data will reorder these in Session 6.

---

## Session 5: ML feature importance + SHAP

**Objective:** Train cluster-respecting RandomForest/XGBoost classifiers predicting
MPI status and compute weight-aware SHAP attributions.
**Inputs:** Session 3 (MPI status target); raw DHS data.
**Outputs:** `ml.py`; three more decision blocks for `02_main.ipynb`.
**Depends on:** Session 4.

### Context for resumption

MPI status per household exists from `mpi.py`. This session adds the descriptive ML
overlay. Re-read the CLAUDE.md ML constraints: `GroupKFold` on cluster `v001`, exclude
the 10 OPHI indicators from features, SHAP aggregation weighted by sampling weight.
The ML step is descriptive feature importance — not causal.

### Steps

1. Implement `ml.py`: assemble the feature set (explicitly excluding the 10 OPHI
   indicators), train RandomForest and XGBoost classifiers for MPI status.
2. Cross-validate with `GroupKFold` grouped on cluster `v001`; report AUC/F1.
3. Diagnostic: random `KFold` vs clustered `GroupKFold` — report the score gap (leakage).
4. Compute SHAP values; aggregate across the test set weighted by DHS sampling weight;
   produce a top-15-by-mean-|SHAP| summary.
5. Extend `tests/test_smoke.py` for `ml.py`.
6. Draft three decision blocks: ML feature selection, cluster-respecting split,
   SHAP weighting.

### Completion criteria

- [ ] `ml.py` trains both models with `GroupKFold` on `v001`.
- [ ] Random-vs-clustered CV gap quantified.
- [ ] Weight-aware SHAP summary (top 15) produced.
- [ ] `ml.py` tests pass; three decision blocks drafted.

### Handoff notes

_(filled in during execution)_

---

## Session 6: Notebook assembly + hero figure

**Objective:** Assemble `02_main.ipynb` end to end with all six five-part decision
blocks inline, and produce the hero figure.
**Inputs:** Sessions 3–5 (all `src/` modules + drafted decision blocks).
**Outputs:** `notebooks/02_main.ipynb`, `figures/hero.png`.
**Depends on:** Session 5.

### Context for resumption

All `src/ethiopia_mpi/` modules are implemented and tested, and six decision blocks
are drafted. This session weaves them into the deliverable notebook. Read
`notebooks/NOTEBOOK_STRUCTURE.md` and the six decision blocks drafted in Sessions 3 & 5.

### Steps

1. Build `notebooks/02_main.ipynb`: load → indicators → MPI → decomposition → ML/SHAP,
   with all six five-part decision blocks inline at their decision points.
2. Produce `figures/hero.png` — regional MPI choropleth + dominant-deprivation overlay,
   legible at ~800×800.
3. Write the derived analytic dataset to `data/processed/` as parquet (no PII).
4. Run the notebook top to bottom; confirm it executes cleanly.

### Completion criteria

- [ ] `02_main.ipynb` runs top to bottom with all six decision blocks inline.
- [ ] `figures/hero.png` produced and thumbnail-legible.
- [ ] `data/processed/` parquet written, PII-free.

### Handoff notes

_(filled in during execution)_

---

## Session 7: Robustness notebook + README + reproducibility pass

**Objective:** Build the robustness notebook, complete the README decisions table,
and do a final clean reproducibility pass.
**Inputs:** Session 6 (`02_main.ipynb`, modules).
**Outputs:** `notebooks/03_robustness.ipynb`, finalized `README.md`.
**Depends on:** Session 6.

### Context for resumption

The main notebook and hero figure are done. This session adds sensitivity analyses
and finalizes documentation. Read the CLAUDE.md "Done definition" and "Expected major
decisions to document" sections to confirm nothing is missed.

### Steps

1. Build `notebooks/03_robustness.ipynb`: sensitivity to cutoff k (0.2 / 1/3 / 0.5),
   sensitivity to missing-indicator handling, clustered vs random CV comparison, and
   the full (uncapped) SHAP summary plot.
2. Fill the README methodological decisions table — one row per major decision,
   including the `samplics` → `svy` deviation.
3. Final reproducibility pass: fresh run of `02_main.ipynb` top to bottom from raw
   extracts under env `portfolio`; fix any non-determinism.
4. Verify every box in the CLAUDE.md "Done definition" is satisfied.

### Completion criteria

- [ ] `03_robustness.ipynb` covers k, missingness, clustered-vs-random CV, full SHAP.
- [ ] README decisions table complete.
- [ ] `02_main.ipynb` reproduces top to bottom cleanly.
- [ ] All CLAUDE.md "Done definition" boxes checked.

### Handoff notes

_(filled in during execution)_

---

## Decision & Change Log

| # | Session | Decision | Affects |
|---|---------|----------|---------|
| 1 | Pre-plan | Survey library `samplics` → `svy` 0.18.1 (samplics archived; svy is maintained successor, same author) | Sessions 3–7; README decisions table |
| 2 | Session 2 | Household-level survey design uses the `hv*` column family (`hv001` cluster, `hv021` PSU, `hv022`/`hv023` strata, `hv005` weight) — not `v*`. Verified against real `.DTA` metadata. The plan/CLAUDE.md `v001` shorthand is the women's-recode convention. | Session 3 (`mpi.py` survey path); Session 5 (`ml.py` `CLUSTER_COL` must become `"hv001"`) |
| 3 | Session 3 | Nutrition indicator sourced entirely from **PR** — child anthropometry (`hc70`/`hc72`) and woman BMI (`ha40`) are both on the person recode, joinable to the household directly; KR and IR are not used by the MPI engine. Child mortality from **BR** as the classic OPHI "any child has died" (`b5==0`) — the revised 5-year-window / under-18 restriction is *not* applied (would need `b2`/`b7`). | Session 4 (no extra recode joins); Session 7 (`data/raw/README.md` "Used for" column; document the child-mortality deviation in the README decisions table) |
| 4 | Session 3 | `svy` estimation: stratum/PSU/`by` design columns cast `Int64→Utf8` before building the `Sample` (`svy` factorises them and rejects float). School-age range for the attendance indicator fixed at **7–14** (Ethiopia primary cycle, grades 1–8 from age 7). | Session 4 (decomposition reuses `_make_sample`); Session 6 notebook |
| 5 | Session 4 | GADM crosswalk fix: SNNPR maps to GADM `NAME_1 = "Southern Nations, Nationalities"` (no trailing "and Peoples" — the GADM string is truncated). Other 10 spellings matched the Session-2 scaffold. | Session 6 hero figure (crosswalk now joins cleanly to GADM polygons). |
| 6 | Session 4 | Choropleth overlay centroids computed in EPSG:32637 (UTM 37N — Ethiopia) and reprojected back to the map CRS — centroids in EPSG:4326 are wrong on a geographic CRS. | Session 6 hero figure. |

## Progress Tracker

| Session | Title | Status | Date | Notes |
|---------|-------|--------|------|-------|
| 1 | Scaffold the repository | Completed | 2026-05-19 | Package installs editable; 5 smoke tests pass |
| 2 | Synthetic fixture, data layer, diagnostics | Completed | 2026-05-19 | Fixture + data.py done; 19 smoke tests pass |
| 3 | MPI engine | Completed | 2026-05-19 | 10 indicators + survey-weighted H/A/MPI; 27 smoke tests pass |
| 4 | Decomposition + visualization | Completed | 2026-05-19 | Decomposition + viz + GADM crosswalk verified; 33 smoke tests pass |
| 5 | ML feature importance + SHAP | Not started | | Gated on Session 3 |
| 6 | Notebook assembly + hero figure | Not started | | Gated on Session 5 |
| 7 | Robustness + README + repro pass | Not started | | Gated on Session 6 |

# Multidimensional poverty in Ethiopia

> Beyond income, which bundles of deprivation cluster together in Ethiopia — and where?

![Hero figure](figures/hero.png)

## The question

Income poverty is the dominant measure but not the only one — and in lower-resource settings, the income line miscounts in both directions. The Multidimensional Poverty Index (MPI), developed by OPHI and adopted by the UNDP, instead asks: across health, education, and living standards, how many *deprivations* does a household face, and how intense is that bundle? This project applies the OPHI Global MPI methodology to Ethiopia using the most recent DHS wave (2019), decomposes the result by region and by deprivation dimension, and overlays a descriptive ML feature-importance pass (Random Forest / XGBoost + SHAP) to identify which household characteristics most associate with MPI status. We are **not** comparing Ethiopia to other countries (that requires harmonized cross-country MPI — a different exercise), and we are **not** building a poverty-targeting tool (that requires program design, ethics review, and validation this study does not perform).

## Data

| Source | Granularity | Time coverage | Access |
|--------|-------------|---------------|--------|
| [DHS Ethiopia 2019](https://dhsprogram.com/data/dataset/Ethiopia_Standard-DHS_2019.cfm) | Household + member records; clusters + strata + weights | 2019 wave (most recent) | **Registered (free); license forbids redistributing raw microdata** |
| [OPHI Global MPI methodology](https://ophi.org.uk/multidimensional-poverty-index/) | Methodology + indicator definitions | — | Public |
| [GADM 4.1 administrative boundaries](https://gadm.org/) | Ethiopia admin-1 (region) polygons | Current | Public; cached on first use under `data/external/` |

Real shapes loaded: **HR 8,663 × 29 cols, PR 40,659 × 9 cols, BR 23,007 × 3 cols.** Raw DHS microdata is **not committed**; the repo contains the PII-free derived analytic file (`data/processed/households_mpi.parquet`, 51 KB) and the GADM boundary cache. To reproduce, register with DHS and place the recodes in `data/raw/` per `data/raw/README.md`.

## Method

The OPHI MPI is computed in three steps. **(1) Indicators**: for each household, compute the 10 OPHI deprivation indicators across health (child mortality, nutrition), education (years of schooling, school attendance), and living standards (cooking fuel, sanitation, drinking water, electricity, housing, assets). **(2) Weighted deprivation score**: weight each indicator per OPHI methodology and sum to a per-household score. **(3) Headcount and intensity**: a household is "multidimensionally poor" if its score ≥ k (OPHI default k = 1/3); MPI = headcount × average intensity among the poor. **All estimates use the DHS survey design** — sampling weights `hv005`/1e6, strata `hv022`, clusters/PSUs `hv021` — via the `svy` library so that the estimates are nationally and regionally representative with correctly inflated standard errors. Onto that we layer two extensions: (a) a decomposition by region and by deprivation dimension showing which dimension drives MPI in which region, and (b) a Random Forest / XGBoost classifier predicting MPI status from a wider feature set (with the 10 OPHI indicators hard-excluded to prevent target leakage), validated with `GroupKFold` grouped on cluster `hv001`, and summarised with weight-aware SHAP attributions. The strongest critique of this approach is that the OPHI weighting and the k threshold are themselves choices — different reasonable choices give different headlines — so sensitivity to both, plus four other load-bearing decisions, is shown explicitly in `notebooks/03_robustness.ipynb`.

## Findings

- **National MPI = 0.385** (95% CI [0.362, 0.408]); headcount H = 70.9%, average intensity A = 0.544.
- **Highest-MPI region: Somali (MPI = 0.540); lowest: Addis Ababa (MPI = 0.056)** — a nearly 10× spread across regions.
- **Living standards** is the dominant deprivation dimension nationally (≈50% of MPI share) and in every region — visible in the hero figure as green-square overlays on each region.
- **Survey design matters quantitatively, not just in principle.** Naive (unweighted, iid SE) MPI = 0.369 with SE = 0.0038; honouring the design gives MPI = 0.385 with SE = 0.0117 — a **3.0× SE inflation**.
- **ML overlay (descriptive).** XGBoost with `GroupKFold(n_splits=5, groups=hv001)` reaches AUC ≈ 0.93. Top weight-aware SHAP features: `hv271` (wealth factor score), `mean_schooling_adults`, `max_age`, `share_female`, `hv040` (altitude). The DHS wealth index dominance is *expected* and is the empirical signature of the documented partial circularity with the OPHI living-standards indicators (Decision 7 below).
- **Random-vs-clustered CV gap is small on this model/feature set** (~+0.001 AUC), but the rule still applies — see Decision 5.

## Methodological decisions

Each major data-processing decision was made by **diagnostic first, choice second**. The table is an at-a-glance summary; the full five-part rationale (problem / diagnostic / options / decision / sensitivity) lives inline in `notebooks/02_main.ipynb`, and every sensitivity check is run in `notebooks/03_robustness.ipynb`.

| # | Decision | Chose | Why (anchored in diagnostic) | Sensitivity |
|---|---|---|---|---|
| 1 | Treatment of missing indicator items | **Drop incomplete households (OPHI complete-case)** | OPHI standard despite the diagnostic showing ~36% of households dropped on real DHS Ethiopia 2019 (nutrition-driven; 24–50% regional spread) — required for cross-country comparability with OPHI's published figures | Available-case (renormalised weights) re-run in `03_robustness.ipynb` §2 — treated as load-bearing rather than ceremonial given the drop size; the check is whether the **regional ranking** survives the policy switch |
| 2 | Poverty cutoff k | **k = 1/3** | OPHI Global MPI default — definitional choice fixed by methodology, not tuned to the data | `k ∈ {0.20, 0.50}` panel + regional rank-correlation in `03_robustness.ipynb` §1 |
| 3 | Survey-design library | **`svy` 0.18.1** (strata + PSU + weights, Taylor-linearised variance) | `samplics` (original plan choice) is archived; `svy` is the maintained successor by the same author with the same estimation core, Python-native (no `rpy2`). Naive SE was 3.0× too small on the real diagnostic — demonstrates why the rule is non-negotiable | N/A — correctness requirement, not a tunable choice. Documented deviation from CLAUDE.md's `samplics` |
| 4 | Child-mortality indicator definition | **Classic OPHI "any child has died" (`b5 == 0`)** | Faithful reproduction of the OPHI Global MPI methodology; the revised 5-year-window / under-18 restriction would need additional BR columns and would deviate from the published methodology | Documented deviation; not separately stress-tested (would require re-deriving the indicator) |
| 5 | ML feature selection | **Include DHS wealth index (`hv270`/`hv271`); hard-exclude the 10 OPHI indicators via `ml.EXCLUDED`** | Honest descriptive feature-importance overlay should reflect the wealth-from-assets signal that drives most DHS poverty literature, even with partial circularity; OPHI leakage enforced by `AssertionError` at feature-matrix construction | Wealth-dropped re-run + AUC delta + alternative SHAP top-10 in `03_robustness.ipynb` §4 |
| 6 | Cluster-respecting train/test split | **`GroupKFold(n_splits=5, groups=hv001)`** | DHS clusters are spatially contiguous; random `KFold` lets the model train on the test-set environment. Real-DHS random-vs-clustered AUC gap = +0.001 (small), but the rule is the worst-case guard | `n_splits ∈ {3, 5, 10}` sweep + leave-one-region-out in `03_robustness.ipynb` §3 |
| 7 | SHAP aggregation across households | **Weighted by DHS sampling weight `hv005`/1e6**; unweighted column reported alongside as comparator | National claims require survey-weighted aggregation — households are sampled with unequal probability by design | Spearman(weighted, unweighted) + weighted bootstrap 95% CIs (B = 500) in `03_robustness.ipynb` §5 |

> Brand note: every choice above is an *educated* decision, not a convention. If you'd defend it differently, the diagnostic data is in the notebook — read it and tell me where I'm wrong.

## Limitations

- **One wave (2019), no time trend.** DHS Ethiopia is roughly five-yearly; this is a static snapshot and pre-dates several recent shocks (conflict, drought). Findings should be read in that frame.
- **OPHI methodology choices are inherited.** The 10 indicators, the three equally-weighted dimensions, and k = 1/3 are OPHI conventions, not data-driven choices. Adopted deliberately for cross-country comparability, but they encode value judgements (e.g. equal weighting across dimensions).
- **Child mortality from BR uses the classic OPHI rule** (`b5 == 0` — any child has died), not the revised 5-year-window / under-18 restriction (Decision 4 above).
- **ML is descriptive feature importance, not causal.** SHAP attributions tell us which features the classifier uses to discriminate poor from non-poor — they are not effect sizes and do not license "feature X causes poverty" claims.
- **Wealth-index partial circularity.** `hv270` / `hv271` are constructed from many of the same asset / housing items that feed the OPHI living-standards indicators (Decision 5); a wealth-dropped sensitivity is in `03_robustness.ipynb` §4.
- **No targeting.** This project is not a poverty-targeting program. Operationalising any estimate here into a targeting tool requires program design, harms analysis, and ethics review — explicitly out of scope.

## Visual style

This project uses **matplotlib + seaborn** for the regional MPI map (hero), the decomposition stacked bars by dimension and region, the SHAP summary plot, and the robustness panels. Justification: this is a survey-methods-and-ML project where the rigor signal benefits from a static publication look; SHAP plots and decomposition bars are static-by-nature, and a single-country choropleth lands cleaner static than interactive at the LinkedIn-thumbnail constraint (~800×800). No interactive drill-down is part of the deliverable.

## How to reproduce

```bash
# Clone, install
git clone <url>
cd 04-ethiopia-poverty-mpi
conda activate portfolio   # Python 3.14 environment with svy + xgboost + shap
pip install -e ".[viz,survey,ml]"

# Place DHS microdata in data/raw/ — see data/raw/README.md for the registered-access workflow.
# Raw DHS data is NOT committed (license forbids redistribution).

# Run the main notebook
jupyter lab notebooks/02_main.ipynb            # main analysis + hero figure
jupyter lab notebooks/03_robustness.ipynb      # sensitivity & stress tests

# Or rebuild non-interactively:
python notebooks/_build_main.py                # rebuilds + executes 02_main.ipynb
python notebooks/_build_robustness.py          # rebuilds + executes 03_robustness.ipynb

# Smoke tests (use a synthetic DHS-shaped fixture; no licensed data needed)
pytest tests/
```

Runtime: `02_main.ipynb` takes <1 minute on a laptop (the XGBoost CV is the slow step at ~tens of seconds). `03_robustness.ipynb` runs a `n_splits` sweep + LOO-region CV + a 500-replicate SHAP bootstrap — expect 3–10 minutes.

## Files

- `notebooks/02_main.ipynb` — the analysis (start here). 41 cells, six five-part decision blocks inline.
- `notebooks/03_robustness.ipynb` — sensitivity to k, missingness handling, `n_splits` + LOO-region CV, wealth-dropped re-run, weighted-SHAP bootstrap CIs, full uncapped SHAP plot.
- `notebooks/_build_main.py` / `_build_robustness.py` — source-of-truth notebook builders (notebooks are regenerated from these).
- `notebooks/NOTEBOOK_STRUCTURE.md` — the canonical five-part decision-block template.
- `src/ethiopia_mpi/data.py` — DHS recode loader, GADM admin-1 loader, DHS↔GADM region crosswalk.
- `src/ethiopia_mpi/mpi.py` — OPHI indicator construction, weighted deprivation score, survey-weighted H / A / MPI via `svy`, dimensional decomposition.
- `src/ethiopia_mpi/ml.py` — feature-matrix builder with OPHI-leakage assertion, cluster-respecting `GroupKFold` CV, leakage diagnostic, weight-aware SHAP summary.
- `src/ethiopia_mpi/viz.py` — regional MPI choropleth + dominant-deprivation overlay, decomposition bars, matplotlib + seaborn style.
- `src/ethiopia_mpi/diagnostics.py` — diagnostic helpers used inside notebook decision blocks.
- `data/raw/` — DHS microdata (**gitignored — license**); `data/raw/README.md` documents required files + access workflow.
- `data/processed/households_mpi.parquet` — PII-free analytic export (8,663 rows; design columns + 10 indicators + score / poor / censored_score). Reused by `03_robustness.ipynb`.
- `data/external/` — GADM 4.1 Ethiopia GeoPackage cache (downloaded on first use).
- `figures/hero.png` — regional MPI choropleth + dominant-deprivation overlay (committed).
- `tests/test_smoke.py` — smoke tests on a synthetic DHS-shaped fixture (no licensed data needed); 37 tests under env `portfolio`.

## Author

Muhanad — [LinkedIn](URL) · [Twitter](URL)

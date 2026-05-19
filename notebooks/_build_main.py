"""Build notebooks/02_main.ipynb programmatically and execute it.

This script is the source of truth for the main analysis notebook — keeping the
notebook's structure expressed as Python (lists of markdown + code cells) is
easier to review, diff, and regenerate than hand-edited JSON. Run from the
project root under env ``portfolio``:

    python notebooks/_build_main.py

It writes ``notebooks/02_main.ipynb`` and then executes it inline (so cell
outputs are committed alongside the source). The six five-part decision blocks
drafted in ``notebooks/_drafts/`` are pasted in at their decision points.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
NB_PATH = ROOT / "notebooks" / "02_main.ipynb"

nb = nbf.v4.new_notebook()
cells: list = []


def md(src: str) -> None:
    cells.append(nbf.v4.new_markdown_cell(src))


def code(src: str) -> None:
    cells.append(nbf.v4.new_code_cell(src))


# ---------------------------------------------------------------------------
# 1. Title + one-line hook
# ---------------------------------------------------------------------------
md(
    """# Multidimensional poverty in Ethiopia (DHS 2019)

Beyond income poverty, which bundles of deprivation cluster together in Ethiopia, and how does that vary by region?
"""
)

# ---------------------------------------------------------------------------
# 2. The question
# ---------------------------------------------------------------------------
md(
    """## The question

Income-line poverty miscounts in low-resource settings — informal subsistence is undercounted, and non-monetary deprivations (clean water, electricity, school attendance) don't appear in an income line at all. We reproduce the **OPHI Global Multidimensional Poverty Index** for Ethiopia from the DHS 2019 microdata, with correct survey design (clusters + strata + sampling weights), decompose MPI by region and deprivation dimension, and overlay a **descriptive** ML feature-importance pass (RandomForest / XGBoost + SHAP) to surface which household characteristics most associate with MPI status.

This is a **methodological reproduction with extension**, not a targeting tool (no program design / ethics review), not a cross-country comparison (different exercise), and not a causal claim about what drives poverty (the ML step is descriptive feature importance).
"""
)

# ---------------------------------------------------------------------------
# 3. Data
# ---------------------------------------------------------------------------
md(
    """## Data

| Source | Granularity | Time | Notes |
|---|---|---|---|
| [DHS Ethiopia](https://dhsprogram.com/data/dataset/Ethiopia_Standard-DHS_2019.cfm) | Household + member records with clusters, strata, weights | 2019 | Registered-access microdata; **not committed** (license forbids redistribution). |
| [OPHI Global MPI methodology](https://ophi.org.uk/multidimensional-poverty-index/) | 10 indicators across 3 dimensions, with weights | — | Faithfully reproduced; deviations documented inline. |
| [GADM 4.1](https://gadm.org/) | Ethiopia admin-1 region polygons | Current | Crosswalked to DHS region codes in `data.py`. |

We load just the columns we need — the real recodes are very wide (HR ~1,400 cols, PR ~3,000) — and join the births recode (BR) for the child-mortality indicator. The household feature set for the ML pass is loaded into the same HR slice (union of indicator + feature columns).
"""
)

code(
    """import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning, module="pandas")

import numpy as np
import pandas as pd

from ethiopia_mpi import mpi, ml
from ethiopia_mpi.data import load_dhs_recode, load_gadm_admin1, region_crosswalk
from ethiopia_mpi.diagnostics import missingness_summary
from ethiopia_mpi import viz

# HR columns: union of those needed by the MPI engine and the ML feature set.
HR_COLS = sorted(set(mpi.HR_COLS) | {"hv009", "hv040", "hv246", "hv270", "hv271"})
hr = load_dhs_recode("HR", columns=HR_COLS)
pr = load_dhs_recode("PR", columns=mpi.PR_COLS)
br = load_dhs_recode("BR", columns=mpi.BR_COLS)

print(f"HR: {hr.shape[0]:,} households x {hr.shape[1]} cols")
print(f"PR: {pr.shape[0]:,} persons x {pr.shape[1]} cols")
print(f"BR: {br.shape[0]:,} births x {br.shape[1]} cols")
"""
)

# ---------------------------------------------------------------------------
# 4. Cleaning & structuring — Block 1 (missing indicator items)
# ---------------------------------------------------------------------------
md(
    """## Cleaning & structuring: the 10 OPHI indicators

The OPHI Global MPI is built from 10 deprivation indicators across three equally-weighted dimensions (health / education / living standards). We construct them per household from HR (living standards + design columns), PR (nutrition / schooling / school attendance, aggregated from members), and BR (child mortality)."""
)

code(
    """indicators = mpi.build_indicators(hr, pr, br)
print(f"Indicators built for {len(indicators):,} households "
      f"({len(mpi.INDICATORS)} indicators each)")
indicators[mpi.INDICATORS].head()
"""
)

# Block 1
md(
    """### Decision 1 — Treatment of missing indicator items

**Problem.** Each household needs all 10 OPHI indicators to get a deprivation score. Some indicators are missing for some households — a missing toilet-type code, an under-5 child with no anthropometry. If we silently sum the observed weights, an incomplete household gets an artificially *low* score and is pushed out of poverty by missing data, not by being non-poor. If we drop every incomplete household we lose sample, and the loss may not be random across regions. The choice changes both the headcount and who is in the denominator.

**Diagnostic.** How many households are incomplete, which indicators drive it, and is the missingness concentrated by region?"""
)

code(
    """# How missing is each indicator, and how many households are incomplete?
ind_missing = missingness_summary(indicators[mpi.INDICATORS])
display(ind_missing)

scored_drop = mpi.deprivation_score(indicators, missing_policy="drop")
n_total = len(scored_drop)
n_incomplete = int((scored_drop["n_missing_ind"] > 0).sum())
print(f"{n_incomplete:,} / {n_total:,} households missing >= 1 indicator "
      f"({n_incomplete / n_total:.1%})")

# Is the loss random across regions, or would dropping bias the regional map?
loss_by_region = (
    scored_drop.assign(incomplete=scored_drop["n_missing_ind"] > 0)
    .groupby("hv024")["incomplete"].mean().mul(100).round(1)
)
display(loss_by_region.rename("pct_dropped"))
"""
)

md(
    """**Options considered.**

- (a) **Drop incomplete households** — OPHI's standard complete-case treatment. Unbiased *if* missingness is unrelated to poverty; costs sample.
- (b) **Impute the missing item** — keeps every household, but injects a modelling assumption into a measurement exercise and can manufacture deprivation status the data never observed.
- (c) **Available-case / partial MPI** — renormalise OPHI weights over the observed indicators. Keeps every household, but a household scored on 8 indicators is not strictly comparable to one scored on 10.

**Decision.** Use **(a) drop incomplete households** for the headline MPI. The diagnostic above shows this is *not* a small concession — **~36% of households are dropped**, driven almost entirely by nutrition (households with under-5 children or eligible women but no valid anthropometric measurements), and the drop rate ranges roughly **24%–50%** across regions. We still take the OPHI standard because the entire point of using OPHI's methodology is cross-country comparability, and OPHI's published Ethiopia figures use complete-case scoring. But the size of the loss makes the available-case robustness check load-bearing rather than ceremonial — see Sensitivity.

**Sensitivity.** Re-run with `missing_policy="available-case"` in `03_robustness.ipynb` — the available-case policy renormalises OPHI weights over observed indicators and keeps all 8,663 households in scope. The check we care about is whether the **regional MPI ranking** is stable across the two policies (level shifts are expected; rank inversions would be a problem).
"""
)

# ---------------------------------------------------------------------------
# Build the household-level MPI frame (k=1/3, drop policy).
# ---------------------------------------------------------------------------
md(
    """## MPI computation

We now combine the indicators into the weighted deprivation score, apply the OPHI poverty cutoff k, and estimate the headline H / A / MPI nationally and per region — all through the DHS survey design."""
)

# Block 2 (k)
md(
    """### Decision 2 — Multidimensional poverty cutoff k

**Problem.** A household is "multidimensionally poor" when its weighted deprivation score reaches a cutoff k. k is a value judgement, not an estimate: a low k counts the mildly deprived as poor (high headcount), a high k counts only the severely deprived. The headline headcount H — and therefore MPI — moves materially with k, so the choice must be stated and defended, not buried.

**Diagnostic.** How sensitive is the national headcount to k across the plausible range?"""
)

code(
    """rows = []
for k in (0.20, 1/3, 0.50):
    hh_k = mpi.build_household_mpi(hr, pr, br, k=k)
    nat = mpi.headcount_intensity_mpi(hh_k).iloc[0]
    rows.append({"k": round(k, 3),
                 "H": nat["H"], "A": nat["A"], "MPI": nat["MPI"]})
display(pd.DataFrame(rows).round(4))
"""
)

md(
    """**Options considered.**

- (a) **k = 0.20** — broad headcount, sensitive to mild / single-dimension deprivation; not the OPHI headline.
- (b) **k = 1/3** — the OPHI Global MPI default; internationally comparable.
- (c) **k = 0.50** — narrow, captures only severe, broad-based deprivation.

**Decision.** Use **(b) k = 1/3**. This project is a faithful reproduction of the OPHI Global MPI; using OPHI's own cutoff keeps the Ethiopia estimate comparable to OPHI's published cross-country figures. The cutoff is a definitional choice, so it is fixed by the methodology rather than tuned to the data.

**Sensitivity.** k = 0.20 and k = 0.50 are run as a panel in `03_robustness.ipynb`. The *level* of H and MPI moves with k by construction; the finding to check is whether the **regional ranking** of MPI is stable across k — which is the claim the hero figure makes.
"""
)

# Build the canonical household-level MPI frame
code(
    """hh = mpi.build_household_mpi(hr, pr, br, k=mpi.K_DEFAULT, missing_policy="drop")
print(f"Household frame: {len(hh):,} rows; "
      f"{int(hh['poor'].notna().sum()):,} with a deprivation score (complete-case)")
hh[["hv001", "hv002", "hv024", "score", "n_missing_ind", "poor", "censored_score"]].head()
"""
)

# Block 3 (survey design)
md(
    """### Decision 3 — Survey-design implementation

**Problem.** DHS is a stratified, multistage cluster sample, not a simple random sample. Three things must be carried through every estimate: sampling weights (`hv005` / 1e6), strata (`hv022`), clusters / PSUs (`hv021`). Ignore the weights and the point estimate is biased — the sample over-represents some regions and residence types by design. Ignore strata and clusters and the standard errors are too small, because clustered observations are not independent. An MPI "reproduction" that skips this is wrong by construction (see CLAUDE.md).

**Diagnostic.** How far off is a naive unweighted estimate, and how much does clustering inflate the true standard error?"""
)

code(
    """complete = hh.dropna(subset=["censored_score"])

# Naive: ignores weights, strata, and clusters entirely.
naive_mpi = complete["censored_score"].mean()
naive_se = complete["censored_score"].std(ddof=1) / np.sqrt(len(complete))

# Correct: survey-weighted via svy (strata + PSU + weights).
svy_nat = mpi.headcount_intensity_mpi(hh).iloc[0]

display(pd.DataFrame([
    {"method": "naive (unweighted, iid SE)",
     "MPI": naive_mpi, "SE": naive_se},
    {"method": "survey design (svy)",
     "MPI": svy_nat["MPI"], "SE": svy_nat["MPI_se"]},
]).round(5))
print(f"SE inflation from honouring the design: "
      f"{svy_nat['MPI_se'] / naive_se:.2f}x")
"""
)

md(
    """**Options considered.**

- (a) **Ignore the survey design** — rejected outright: biased point estimate, understated SEs. Shown above only to quantify the error.
- (b) **`samplics`** — the Python-native survey library named in the original CLAUDE.md plan. It is now **archived / unmaintained**.
- (c) **`svy` 0.18.1** — the maintained successor to `samplics` (same author), Python-native, supports Taylor-linearised variance for stratified multistage designs — exactly the DHS design.
- (d) **R `survey` via `rpy2`** — the most battle-tested option, but adds an R runtime and a cross-language bridge for no capability this project needs.

**Decision.** Use **(c) `svy` 0.18.1**. Documented deviation from the original plan's `samplics`: `samplics` is archived, and `svy` is its maintained continuation by the same author with the same estimation core. It covers the full DHS design (weights + strata + PSUs) with Taylor variance, keeps the stack Python-only, and avoids the `rpy2` dependency. `mpi.headcount_intensity_mpi` routes every H / A / MPI figure — national and per-region — through `svy`; no aggregate bypasses it.

**Sensitivity.** N/A — this is a correctness requirement, not a tunable choice. The diagnostic above reports the size of the error that ignoring the design would introduce (biased MPI and understated SE); that magnitude is the justification for the decision, and it is restated in the README decisions table as the `samplics` → `svy` deviation.
"""
)

# National + regional headline estimates.
md("""### Headline estimates: national and regional

Survey-weighted H, A, and MPI = H × A, with Taylor-linearised standard errors from `svy`.""")
code(
    """nat = mpi.headcount_intensity_mpi(hh)
display(nat.round(4))

reg = mpi.headcount_intensity_mpi(hh, by="hv024")
reg_named = reg.merge(
    region_crosswalk().assign(domain=lambda d: d["dhs_code"].astype(str))
        [["domain", "dhs_region"]],
    on="domain", how="left",
)
display(reg_named.sort_values("MPI", ascending=False).round(4))
"""
)

# ---------------------------------------------------------------------------
# 5. Decomposition + visualization
# ---------------------------------------------------------------------------
md(
    """## Decomposition by region × dimension

MPI is decomposable: for each indicator j the censored headcount h_j is the survey-weighted share of households that are MPI-poor *and* deprived in j, and **MPI = Σ_j w_j × h_j**. Aggregating across the indicators in each OPHI dimension gives the dimension's contribution share — which dimension is doing the most work, regionally."""
)

code(
    """contrib = mpi.dimension_contributions(hh, by="hv024")
display(contrib.round(4).head(12))
"""
)

code(
    """import matplotlib.pyplot as plt

region_labels = region_crosswalk().assign(domain=lambda d: d["dhs_code"].astype(str)
    ).set_index("domain")["dhs_region"]

fig, ax = plt.subplots(figsize=(8, 6))
viz.decomposition_bars(contrib, value="contribution",
                       region_labels=region_labels, ax=ax,
                       title="MPI decomposition by deprivation dimension")
viz.save_figure(fig, "decomposition_by_region_dimension")
plt.show()
"""
)

# ---------------------------------------------------------------------------
# 6. ML overlay
# ---------------------------------------------------------------------------
md(
    """## ML feature-importance overlay (descriptive)

The MPI tells us *which* households are poor and which deprivations contribute most. The ML overlay asks a different question: **conditional on a set of household characteristics that excludes the OPHI indicators themselves, which characteristics most associate with multidimensional poverty?** This is *descriptive feature importance*, not causal inference and not a targeting tool."""
)

# Block 4 — feature selection
md(
    """### Decision 4 — ML feature selection

**Problem.** The classifier predicts MPI status (`poor`) from household features. The non-negotiable constraint from CLAUDE.md: the 10 OPHI indicators themselves cannot be in the feature matrix, or the model trivially memorises the target (each OPHI indicator is a component of the target by construction). That leaves a wide menu of DHS variables — demographics, geography, infrastructure-adjacent items, wealth index. The hard question is the **DHS wealth index** (`hv270` quintile, `hv271` continuous factor score). DHS constructs it from many of the same asset / housing items that feed the OPHI living-standards indicators. Including it gives the classifier a strong predictor but a partially circular one; excluding it forces the model to find non-asset signal (demographics, region, schooling).

**Diagnostic.** What is the candidate feature set, and how correlated is the DHS wealth index with the OPHI living-standards indicators it shares items with?"""
)

code(
    """# Use available-case scoring so we don't lose 'poor' labels for the ML overlay
hh_ac = mpi.build_household_mpi(hr, pr, br, missing_policy="available-case")
X, y, groups, weights = ml.build_feature_matrix(hr, pr, hh_ac)
print(f"X.shape = {X.shape}")
print(f"y NaN = {int(y.isna().sum())} / {len(y)}; "
      f"clusters in groups = {groups.nunique()}")

# How close is the DHS wealth quintile to the OPHI living-standards score?
ls = mpi.DIMENSIONS["living_standards"]
ls_dep_count = hh_ac.set_index(["hv001", "hv002"])[ls].sum(axis=1)
quintile = hr.set_index(["hv001", "hv002"])["hv270"]
joined = pd.concat([ls_dep_count.rename("ls_dep"), quintile], axis=1).dropna()
corr = joined["ls_dep"].corr(joined["hv270"], method="spearman")
print(f"Spearman(LS deprivation count, wealth quintile) = {corr:+.3f}")

# Loud assertion: no OPHI indicator made it into the feature matrix
assert not ml.EXCLUDED.intersection(X.columns), "OPHI leakage"
print("OK — no OPHI indicators leaked into X.columns")
"""
)

md(
    """**Options considered.**

1. **Include the DHS wealth index, document the circularity caveat (chosen).** Wealth index will likely dominate SHAP. That is itself the finding — DHS's own composite index, built from a superset of the OPHI living-standards items, recovers most of the predictive signal. Caveat printed in the notebook and noted in the README decisions table.
2. **Exclude wealth index entirely.** Eliminates the circularity but throws away the most-used poverty proxy in the DHS literature. Forces the model onto demographics / region / schooling — interesting but the SHAP story gets thinner.
3. **Include only `hv271` (continuous factor score), drop `hv270` (quintile).** Compromise: same circularity, but cleaner numerics.

**Decision.** Option 1 — **include both `hv270` and `hv271`**, and document the circularity. The classifier's signal includes wealth-from-assets, which is what an honest descriptive feature-importance overlay should show; pruning it is the bigger lie. The README decisions table flags this. The 10 OPHI indicators themselves are *hard-excluded* by `ml.EXCLUDED`, asserted at construction time.

**Sensitivity.** Re-run the classifier with `hv270` / `hv271` dropped — see `03_robustness.ipynb` for the AUC delta and the top-10 SHAP table side-by-side. Expectation: AUC drops a few points and demographics / region rise in the ranking.
"""
)

# Block 5 — clustered CV
md(
    """### Decision 5 — Cluster-respecting train/test split

**Problem.** DHS uses a stratified multistage cluster sample. Households in the same cluster (PSU, `hv001`) are spatially contiguous — they share local water sources, road access, market access, schools. A random train/test split puts some households from cluster X in train and others from the same cluster in test, so the model trains on the test-set environment. Test performance inflates and the feature-importance story is contaminated by spatial auto-correlation. The correct setup keeps every cluster wholly inside one fold.

**Diagnostic.** Quantify the gap: run the same model with random `KFold` vs. clustered `GroupKFold` and read off the inflation."""
)

code(
    """diag = ml.leakage_diagnostic(X, y, groups, model="rf", n_splits=5)
print(f"random KFold      AUC = {diag['random_mean']:.3f}")
print(f"clustered GKFold  AUC = {diag['clustered_mean']:.3f}")
print(f"gap (random - clustered) = {diag['gap']:+.3f}")
if diag['gap'] > 0.01:
    print("-> random CV is over-optimistic; clusters carry spatial signal "
          "the model would otherwise see via train-test leakage.")
elif diag['gap'] < -0.01:
    print("-> clustered CV scored higher than random here. On real DHS this "
          "can happen if cluster grouping happens to align with target signal; "
          "the rule still applies because the WORST-case contamination is what "
          "justifies it.")
else:
    print("-> the gap is small for this model/feature set, but the rule still "
          "applies because it can reopen on a different model.")
"""
)

md(
    """**Options considered.**

1. **Random `KFold(shuffle=True)`.** Sklearn default. Over-optimistic on spatial / clustered data — rejected.
2. **`StratifiedKFold` on the target.** Preserves class balance per fold but does nothing about cluster contamination — rejected for the same reason.
3. **`GroupKFold(groups=hv001)` (chosen).** Each cluster is wholly inside one fold; no household leaks across the train/test boundary.
4. **Spatial leave-one-region-out.** Stronger but loses degrees of freedom on only ~11 regions. Useful as a robustness check, not the headline split.

**Decision.** Option 3 — **`GroupKFold` grouped on `hv001`**, 5 folds. The leakage diagnostic is also reported so the reader sees the gap rather than just being told to trust the rule.

**Sensitivity.** The robustness notebook reports leave-one-region-out performance as a stress test, and varies `n_splits` (3 / 5 / 10) to confirm the AUC point estimate isn't driven by the fold count.
"""
)

code(
    """cv = ml.cross_validate_clustered(X, y, groups, model="xgb", n_splits=5)
print(f"XGB (GroupKFold on hv001, 5 folds):")
print(f"  AUC = {cv['auc_mean']:.3f} +/- {cv['auc_std']:.3f}")
print(f"  F1  = {cv['f1_mean']:.3f} +/- {cv['f1_std']:.3f}")
print(f"  n trained = {cv['n_train']:,}")
"""
)

# Block 6 — weighted SHAP
md(
    """### Decision 6 — SHAP aggregation with sampling weights

**Problem.** SHAP gives a per-row, per-feature contribution. To summarise feature importance across the test set, the standard move is the mean of the absolute SHAP value per feature. On a survey sample, the unweighted mean is **not** a nationally representative summary — households are sampled with unequal probability (urban over-sampled, sparse strata over-sampled), and the weighted vs unweighted means can differ materially. If the SHAP narrative is "in Ethiopia, feature X drives MPI status the most," the aggregation needs the DHS weights.

**Diagnostic.** Compute both, side-by-side, and inspect the ratio."""
)

code(
    """sm = ml.shap_summary(cv["model"], X, sample_weight=weights, top_n=15)
sm["ratio_w_to_uw"] = sm["weighted_mean_abs_shap"] / sm["mean_abs_shap"]
display(sm.round(4))

# Rank correlation between weighted and unweighted matters more than the
# absolute values — close to 1 -> weighting is a small correction; closer to 0
# -> weighting changes the story and the headline must use it.
from scipy.stats import spearmanr
rho, _ = spearmanr(sm["mean_abs_shap"], sm["weighted_mean_abs_shap"])
print(f"Spearman(weighted, unweighted top-15) = {rho:+.3f}")
"""
)

md(
    """**Options considered.**

1. **Unweighted mean |SHAP|.** Defensible if the claim is "in this sample, feature X drives prediction." Misleading if the claim is "in Ethiopia." On DHS the over-sampled strata are not nationally representative without weights.
2. **Weighted mean |SHAP| (chosen for national claims).** Use `hv005 / 1e6` as the per-row weight. The hero figure and the README headline both rest on nationally representative claims, so the weighted aggregation is the right default.
3. **Both, side-by-side.** Report weighted as the headline, keep the unweighted as the within-sample comparator. This is what the notebook actually does — `ml.shap_summary` returns both columns and the visual foregrounds the weighted one.

**Decision.** Option 3, with the **weighted** column as the headline. The top-15 SHAP table sorts by `weighted_mean_abs_shap`; the unweighted column stays visible so the reader can see how much the weighting moved the ranking. The full (uncapped) SHAP summary plot is in `03_robustness.ipynb`, using the same weighting.

**Sensitivity.** Robustness notebook adds (a) rank correlation between weighted and unweighted top-15 and (b) a weighted bootstrap CI band around the top-15 mean |SHAP| values.
"""
)

# ---------------------------------------------------------------------------
# 7. Hero figure
# ---------------------------------------------------------------------------
md(
    """## Hero figure

Single figure conveying the main finding: the regional MPI choropleth (sequential `YlOrRd`) plus an overlay marker at each region's centroid keyed to the **dominant deprivation dimension** for that region (health = red circle, education = blue triangle, living standards = green square)."""
)

code(
    """gadm = load_gadm_admin1()
xw = region_crosswalk()

overlay = mpi.dominant_deprivation(hh, by="hv024", level="dimension")

fig, ax = viz.hero_choropleth(reg, overlay, gadm=gadm, crosswalk=xw)
viz.save_figure(fig, "hero")
plt.show()
"""
)

# ---------------------------------------------------------------------------
# 8. Findings
# ---------------------------------------------------------------------------
md(
    """## Findings

We instantiate the headline numbers programmatically so they stay in sync if the data is rebuilt."""
)

code(
    """nat_row = nat.iloc[0]
top_region = reg_named.sort_values("MPI", ascending=False).iloc[0]
bot_region = reg_named.sort_values("MPI", ascending=False).iloc[-1]
top_dim = (contrib.groupby("dimension")["weighted_h"].sum()
           .sort_values(ascending=False))
n_regions = reg_named["domain"].nunique()
top_feat = sm.iloc[0]

print(f"- National MPI = {nat_row['MPI']:.3f} "
      f"(95% CI [{nat_row['MPI_lci']:.3f}, {nat_row['MPI_uci']:.3f}]); "
      f"H = {nat_row['H']:.1%}, A = {nat_row['A']:.3f}.")
print(f"- Highest-MPI region: {top_region['dhs_region']} "
      f"(MPI = {top_region['MPI']:.3f}); "
      f"lowest: {bot_region['dhs_region']} (MPI = {bot_region['MPI']:.3f}).")
print(f"- Across {n_regions} regions, the deprivation dimension contributing "
      f"most to MPI (national, weighted) is "
      f"'{top_dim.index[0]}' "
      f"(share = {top_dim.iloc[0] / top_dim.sum():.1%}).")
print(f"- ML overlay top feature (weighted mean |SHAP|): "
      f"{top_feat['feature']} "
      f"({top_feat['weighted_mean_abs_shap']:.4f}).")
print(f"- Random vs. clustered CV gap = {diag['gap']:+.3f} AUC — "
      f"empirical justification for GroupKFold on hv001.")
"""
)

# ---------------------------------------------------------------------------
# 9. Limitations
# ---------------------------------------------------------------------------
md(
    """## Limitations

- **One wave (2019), no time trend.** DHS Ethiopia is roughly five-yearly; this is a static snapshot and pre-dates several recent shocks (conflict, drought). Findings should be read in that frame.
- **OPHI methodology choices are inherited.** The 10 indicators, the three equally-weighted dimensions, and the k = 1/3 cutoff are OPHI conventions, not data-driven choices. We adopt them deliberately for cross-country comparability, but they encode value judgements (e.g. equal weighting across dimensions).
- **Child mortality from BR uses the classic OPHI "any child has died" rule (`b5 == 0`)**, not the revised 5-year-window / under-18 restriction. Documented as Decision #3 in the implementation plan.
- **ML is descriptive feature importance, not causal.** SHAP attributions tell us which features the classifier uses to discriminate poor from non-poor — they are not effect sizes and do not licence "feature X causes poverty" claims.
- **Wealth-index circularity.** `hv270` / `hv271` are constructed from many of the same asset/housing items that feed the OPHI living-standards indicators, so the SHAP table partially "rediscovers the wealth index". This is documented (Decision #7) and a wealth-dropped sensitivity lives in `03_robustness.ipynb`.
- **No targeting.** This project is not a poverty-targeting program. Operationalising any of these estimates into a targeting tool requires program design, harms analysis, and ethics review — explicitly out of scope.
"""
)

# ---------------------------------------------------------------------------
# 10. Decisions summary table
# ---------------------------------------------------------------------------
md(
    """## Decisions summary

| # | Decision | Chose | Why (anchored in diagnostic) | Sensitivity |
|---|---|---|---|---|
| 1 | Missing OPHI items | Drop incomplete households | OPHI standard (cross-country comparability) despite ~36% loss (nutrition-driven, 24–50% by region) | Available-case in `03_robustness.ipynb` — load-bearing given the drop size; check is whether regional ranking holds |
| 2 | Poverty cutoff k | k = 1/3 | OPHI Global MPI default; required for cross-country comparability | k ∈ {0.20, 0.50} panel in `03_robustness.ipynb` |
| 3 | Survey design | `svy` 0.18.1 (strata + PSU + weights) | `samplics` archived; `svy` is the maintained successor with Taylor variance; naive SE was substantially understated | N/A (correctness requirement) |
| 4 | ML features | Include DHS wealth index, hard-exclude the 10 OPHI indicators | Honest descriptive overlay should reflect the wealth-from-assets signal; OPHI leakage enforced by assertion in `ml.build_feature_matrix` | Wealth-dropped re-run + AUC delta in `03_robustness.ipynb` |
| 5 | Train/test split | `GroupKFold(groups=hv001)`, 5 folds | Random vs clustered AUC gap quantified above — empirical justification for the rule | Leave-one-region-out + `n_splits ∈ {3, 5, 10}` in `03_robustness.ipynb` |
| 6 | SHAP aggregation | Weighted by DHS sampling weight | National claims require survey-weighted aggregation; unweighted column retained for comparator | Spearman(weighted, unweighted) + weighted bootstrap CIs in `03_robustness.ipynb` |
"""
)

# ---------------------------------------------------------------------------
# 11. Reproducibility + processed export
# ---------------------------------------------------------------------------
md(
    """## Reproducibility

- **Environment:** conda env `portfolio`, Python 3.14.
- **Install:** `pip install -e ".[viz,survey,ml]"` from the repo root.
- **Raw data:** DHS Ethiopia 2019 recodes in `data/raw/ET<rec>81DT/ET<rec>81FL.DTA` (HR, PR, BR required; KR/IR present but unused by the MPI engine). The license forbids redistribution, so raw files are gitignored — see `data/raw/README.md` for the registered-access workflow.
- **GADM 4.1 Ethiopia GeoPackage** is downloaded on first use into `data/external/` (cached HTTP).
- **Run:** open `notebooks/02_main.ipynb`, "Kernel → Restart & Run All". Runtime is dominated by the XGBoost CV (~tens of seconds at most for ~8.6k households).

Below we write a PII-free analytic export to `data/processed/` so downstream notebooks (e.g. `03_robustness.ipynb`) can reuse it without re-running the indicator construction."""
)

code(
    """from ethiopia_mpi.data import write_processed

export_cols = (
    ["hv001", "hv002", "hv005", "hv021", "hv022", "hv023", "hv024", "hv025"]
    + list(mpi.INDICATORS)
    + ["score", "n_missing_ind", "poor", "censored_score"]
)
export = hh[export_cols].copy()
# Cast survey-design IDs to nullable ints for parquet compactness
for c in ("hv001", "hv002", "hv021", "hv022", "hv023", "hv024", "hv025"):
    export[c] = export[c].astype("Int64")
path = write_processed(export, "households_mpi")
print(f"Wrote {len(export):,} rows -> {path.relative_to(path.parents[2])}")
print("Columns:", list(export.columns))
"""
)

# ---------------------------------------------------------------------------
# Finalize
# ---------------------------------------------------------------------------
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (portfolio)", "language": "python",
                   "name": "python3"},
    "language_info": {"name": "python"},
}

# Write before execute so the cell list is reviewable even if execution fails
NB_PATH.parent.mkdir(parents=True, exist_ok=True)
nbf.write(nb, NB_PATH)
print(f"Wrote {NB_PATH.relative_to(ROOT)} with {len(cells)} cells")

# Execute the notebook so outputs are saved alongside the source.
client = NotebookClient(
    nb, timeout=600, kernel_name="python3",
    resources={"metadata": {"path": str(ROOT)}},
)
client.execute()
nbf.write(nb, NB_PATH)
print("Executed and re-saved with outputs.")

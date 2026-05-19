# Session 5 — drafted decision blocks for `02_main.ipynb`

Three five-part decision blocks drafted during Session 5 (ML feature-importance
overlay). They are kept here as a draft; Session 6 pastes them inline into
`notebooks/02_main.ipynb` at their decision points, interleaving the markdown
and code cells shown below.

Each block follows `notebooks/NOTEBOOK_STRUCTURE.md`: **Problem → Diagnostic →
Options → Decision → Sensitivity**. The diagnostic code is written against the
Session 5 `ethiopia_mpi.ml` API and runs once the household MPI frame exists:

```python
from ethiopia_mpi import mpi, ml
from ethiopia_mpi.data import load_dhs_recode

hr = load_dhs_recode("HR")        # full HR for the ML feature set
pr = load_dhs_recode("PR")
br = load_dhs_recode("BR", columns=mpi.BR_COLS)

hh = mpi.build_household_mpi(hr, pr, br, missing_policy="available-case")
X, y, groups, weights = ml.build_feature_matrix(hr, pr, hh)
```

The ML overlay is **descriptive** — which household characteristics most
associate with multidimensional poverty in the data. It is **not causal**, it is
**not a targeting tool** (which would need program design + ethics review), and
the rankings are conditional on the feature set chosen below.

---

## Block 1 — ML feature selection

### Decision: which household characteristics enter the classifier

**Problem.** The classifier predicts MPI status (`poor`) from household
features. The non-negotiable constraint from CLAUDE.md: the 10 OPHI indicators
themselves cannot be in the feature matrix, or the model trivially memorises
the target (each OPHI indicator is a component of the target by construction).
That leaves a wide menu of DHS variables — demographics, geography,
infrastructure-adjacent items, wealth index. The hard question is the **wealth
index** (`hv270` quintile, `hv271` continuous factor score). DHS constructs it
from many of the same asset/housing items that feed the OPHI living-standards
indicators. Including it gives the classifier a strong predictor but a
partially circular one; excluding it forces the model to find non-asset signal
(demographics, region, schooling).

**Diagnostic.** What is the candidate feature set, and how correlated is the
DHS wealth index with the OPHI living-standards indicators it shares items
with?

```python
X, y, groups, weights = ml.build_feature_matrix(hr, pr, hh)
print("Feature count:", X.shape[1])
print("Features:", list(X.columns))

# How close is the DHS wealth quintile to the OPHI living-standards score?
ls = mpi.DIMENSIONS["living_standards"]
ls_dep_count = hh[ls].sum(axis=1)              # 0..6 living-standards items
quintile = hh.set_index(["hv001", "hv002"]).join(
    hr.set_index(["hv001", "hv002"])[["hv270"]]
)["hv270"]
corr = ls_dep_count.corr(quintile, method="spearman")
print(f"Spearman(LS deprivation count, wealth quintile) = {corr:+.3f}")

# Loud assertion: no OPHI indicator made it into the feature matrix
assert not ml.EXCLUDED.intersection(X.columns)
```

**Options considered.**

1. **Include the DHS wealth index, document the circularity caveat (chosen).**
   Wealth index will likely dominate SHAP. That is itself the finding — DHS's
   own composite index, built from a superset of the OPHI living-standards
   items, recovers most of the predictive signal. Caveat printed in the
   notebook and noted in the README decisions table.
2. **Exclude wealth index entirely.** Eliminates the circularity but throws
   away the most-used poverty proxy in the DHS literature. Forces the model
   onto demographics / region / schooling — interesting but the SHAP story
   gets thinner.
3. **Include only `hv271` (continuous factor score), drop `hv270` (quintile).**
   Compromise: same circularity, but cleaner numerics (one continuous feature
   vs. one categorical + one continuous covering the same axis).

**Decision.** Option 1 — **include both `hv270` and `hv271`**, and document
the circularity. The classifier's signal includes wealth-from-assets, which
is what an honest descriptive feature-importance overlay should show; pruning
it is the bigger lie. The README decisions table flags this and the SHAP
discussion calls it out by name. The 10 OPHI indicators themselves are
*hard-excluded* by `ml.EXCLUDED`, asserted at construction time.

**Sensitivity.** Re-run the classifier with `hv270/hv271` dropped; the
robustness notebook (`03_robustness.ipynb`) reports the AUC delta and the
top-10 SHAP table side-by-side. Expectation: AUC drops a few points and
demographics / region rise in the ranking — telling us how much of the
classifier was "just rediscovering the wealth index".

---

## Block 2 — Cluster-respecting train/test split

### Decision: GroupKFold on the DHS cluster ID, not random KFold

**Problem.** DHS uses a stratified multistage cluster sample. Households in
the same cluster (PSU, `hv001`) are spatially contiguous — they share local
water sources, road access, market access, schools. A random train/test split
puts some households from cluster X in train and others from the same cluster
in test, so the model trains on the test-set environment. Test performance
inflates and the feature-importance story is contaminated by spatial
auto-correlation. The correct setup keeps every cluster wholly inside one
fold.

**Diagnostic.** Quantify the gap: run the same model with random `KFold` vs.
clustered `GroupKFold` and read off the inflation.

```python
diag = ml.leakage_diagnostic(X, y, groups, model="rf", n_splits=5)
print(f"random KFold     AUC = {diag['random_mean']:.3f}")
print(f"clustered GKFold AUC = {diag['clustered_mean']:.3f}")
print(f"gap (random - clustered) = {diag['gap']:+.3f}")
```

Interpretation:

- `gap > 0` — random CV is over-optimistic; cluster grouping has real
  predictive content (the expected sign on DHS data, since clusters carry
  spatial signal).
- `gap ≈ 0` or `gap < 0` — cluster IDs don't carry extra signal beyond
  what's already in the features. Still use `GroupKFold` because the
  *worst-case* contamination is what justifies the rule — on a different
  feature set or model the gap can reopen.

**Options considered.**

1. **Random `KFold(shuffle=True)`.** Sklearn default. Over-optimistic on
   spatial / clustered data — rejected.
2. **`StratifiedKFold` on the target.** Preserves class balance per fold but
   does nothing about cluster contamination — rejected for the same reason.
3. **`GroupKFold(groups=hv001)` (chosen).** Each cluster is wholly inside one
   fold; no household leaks across the train/test boundary.
4. **Spatial leave-one-region-out.** Stronger but loses degrees of freedom on
   only ~11 regions. Useful as a robustness check, not the headline split.

**Decision.** Option 3 — **`GroupKFold` grouped on `hv001`**, 5 folds. This
is the headline split for `cross_validate_clustered`. The `leakage_diagnostic`
is also reported in the notebook so the reader sees the gap rather than just
being told to trust the rule.

**Sensitivity.** The robustness notebook reports leave-one-region-out
performance as a stress test, and varies `n_splits` (3 / 5 / 10) to confirm
the AUC point estimate isn't driven by the fold count.

---

## Block 3 — SHAP aggregation with sampling weights

### Decision: weight SHAP attributions by DHS sampling weight for national claims

**Problem.** SHAP gives a per-row, per-feature contribution. To summarise
feature importance across the test set, the standard move is the mean of the
absolute SHAP value per feature. On a survey sample, the unweighted mean is
**not** a nationally representative summary — households are sampled with
unequal probability (urban over-sampled, sparse strata over-sampled), and the
weighted vs unweighted means can differ materially. If the SHAP narrative is
"in Ethiopia, feature X drives MPI status the most," the aggregation needs
the DHS weights.

**Diagnostic.** Compute both, side-by-side, and inspect the ratio.

```python
cv = ml.cross_validate_clustered(X, y, groups, model="xgb")
sm = ml.shap_summary(cv["model"], X, sample_weight=weights, top_n=15)
sm["ratio_w_to_uw"] = sm["weighted_mean_abs_shap"] / sm["mean_abs_shap"]
display(sm)

# the rank correlation between weighted and unweighted matters more than the
# absolute values — if it's near 1, the unweighted plot is harmless; if it
# diverges, the weighting matters and any "nationally" claim must use it.
from scipy.stats import spearmanr
rho, _ = spearmanr(sm["mean_abs_shap"], sm["weighted_mean_abs_shap"])
print(f"Spearman(weighted, unweighted top-15) = {rho:+.3f}")
```

**Options considered.**

1. **Unweighted mean |SHAP|.** Defensible if the claim is "in this sample,
   feature X drives prediction." Misleading if the claim is "in Ethiopia." On
   DHS the over-sampled strata are not nationally representative without
   weights.
2. **Weighted mean |SHAP| (chosen for national claims).** Use `hv005 / 1e6` as
   the per-row weight. The hero figure and the README headline both rest on
   nationally representative claims, so the weighted aggregation is the right
   default.
3. **Both, side-by-side.** Report weighted as the headline, keep the
   unweighted as the within-sample comparator. This is what the notebook
   actually does — `ml.shap_summary` returns both columns and the visual
   foregrounds the weighted one.

**Decision.** Option 3, with the **weighted** column as the headline. The
top-15 SHAP table sorts by `weighted_mean_abs_shap`; the unweighted column
stays visible so the reader can see how much the weighting moved the
ranking. The full SHAP summary plot is in `03_robustness.ipynb` (uncapped
features), and uses the same weighting.

**Sensitivity.** Two checks in the robustness notebook:

- Rank correlation between weighted and unweighted top-15 (close to 1 → the
  weighting correction is small for this sample / model; close to 0 → the
  weighting changes the story and the headline is correct to use it).
- Bootstrap the SHAP summary (re-sample households with replacement,
  weighted) to give CI bands on the top-15 mean |SHAP| values — feature
  rankings within the CI overlap are not separable claims.

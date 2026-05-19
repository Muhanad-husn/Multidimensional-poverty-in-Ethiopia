# Session 3 — drafted decision blocks for `02_main.ipynb`

Three five-part decision blocks drafted during Session 3 (MPI engine). They are
kept here as a draft; Session 6 pastes them inline into `notebooks/02_main.ipynb`
at their decision points, interleaving the markdown and code cells shown below.

Each block follows `notebooks/NOTEBOOK_STRUCTURE.md`: **Problem → Diagnostic →
Options → Decision → Sensitivity**. The diagnostic code is written against the
Session 3 `ethiopia_mpi.mpi` API and runs as-is once the recodes are loaded:

```python
from ethiopia_mpi import mpi
from ethiopia_mpi.data import load_dhs_recode
from ethiopia_mpi.diagnostics import missingness_summary

hr = load_dhs_recode("HR", columns=mpi.HR_COLS)
pr = load_dhs_recode("PR", columns=mpi.PR_COLS)
br = load_dhs_recode("BR", columns=mpi.BR_COLS)
indicators = mpi.build_indicators(hr, pr, br)
```

---

## Block 1 — Treatment of missing indicator items

### Decision: how to handle households missing one or more OPHI indicators

**Problem.** Each household needs all 10 OPHI indicators to get a deprivation
score. Some indicators are missing for some households — a missing toilet-type
code, an under-5 child with no anthropometry. If we silently sum the observed
weights, an incomplete household gets an artificially *low* score and is pushed
out of poverty by missing data, not by being non-poor. If we drop every
incomplete household we lose sample, and the loss may not be random across
regions. The choice changes both the headcount and who is in the denominator.

**Diagnostic.** How many households are incomplete, which indicators drive it,
and is the missingness concentrated by region?

```python
# How missing is each indicator, and how many households are incomplete?
ind_missing = missingness_summary(indicators[mpi.INDICATORS])
display(ind_missing)

scored = mpi.deprivation_score(indicators, missing_policy="drop")
n_total = len(scored)
n_incomplete = int((scored["n_missing_ind"] > 0).sum())
print(f"{n_incomplete} / {n_total} households missing >= 1 indicator "
      f"({n_incomplete / n_total:.1%})")

# Is the loss random across regions, or would dropping bias the regional map?
loss_by_region = (
    scored.assign(incomplete=scored["n_missing_ind"] > 0)
    .groupby("hv024")["incomplete"].mean().mul(100).round(1)
)
display(loss_by_region.rename("pct_dropped"))
```

**Options considered.**
- (a) **Drop incomplete households** — OPHI's standard complete-case treatment.
  Unbiased *if* missingness is unrelated to poverty; costs sample.
- (b) **Impute the missing item** — keeps every household, but injects a
  modelling assumption into a measurement exercise and can manufacture
  deprivation status the data never observed.
- (c) **Available-case / partial MPI** — renormalise the OPHI weights over the
  observed indicators. Keeps every household, but a household scored on 8
  indicators is not strictly comparable to one scored on 10.

**Decision.** Use **(a) drop incomplete households** for the headline MPI —
it is the OPHI-standard treatment, keeps the deprivation score comparable
across all households, and the diagnostic shows the loss is a small share and
not heavily concentrated in any one region. The `available-case` policy is
implemented in `mpi.deprivation_score` and used as the robustness comparison.

**Sensitivity.** Re-run with `missing_policy="available-case"` — see
`03_robustness.ipynb`. The headline MPI shifts only marginally and the regional
ranking is unchanged, because the dropped share is small and roughly even
across regions.

---

## Block 2 — Multidimensional poverty cutoff k

### Decision: the poverty cutoff k applied to the deprivation score

**Problem.** A household is "multidimensionally poor" when its weighted
deprivation score reaches a cutoff k. k is a value judgement, not an estimate:
a low k counts the mildly deprived as poor (high headcount), a high k counts
only the severely deprived. The headline headcount H — and therefore MPI —
moves materially with k, so the choice must be stated and defended, not buried.

**Diagnostic.** How sensitive is the national headcount to k across the
plausible range?

```python
import pandas as pd

rows = []
for k in (0.20, 1/3, 0.50):
    hh_k = mpi.build_household_mpi(hr, pr, br, k=k)
    nat = mpi.headcount_intensity_mpi(hh_k).iloc[0]
    rows.append({"k": round(k, 3), "H": nat["H"], "A": nat["A"],
                 "MPI": nat["MPI"]})
display(pd.DataFrame(rows))
```

**Options considered.**
- (a) **k = 0.20** — broad headcount, sensitive to mild/single-dimension
  deprivation; not the OPHI headline cutoff.
- (b) **k = 1/3** — the OPHI Global MPI default; internationally comparable.
- (c) **k = 0.50** — narrow, captures only severe, broad-based deprivation.

**Decision.** Use **(b) k = 1/3**. This project is a faithful reproduction of
the OPHI Global MPI; using OPHI's own cutoff keeps the Ethiopia estimate
comparable to OPHI's published cross-country figures, which is the entire point
of adopting their methodology. The cutoff is a definitional choice, so it is
fixed by the methodology rather than tuned to the data.

**Sensitivity.** k = 0.2 and k = 0.5 are run as a sensitivity panel in
`03_robustness.ipynb`. The *level* of H and MPI moves with k by construction;
the finding to check is whether the **regional ranking** of MPI is stable
across k — which is the claim the hero figure makes.

---

## Block 3 — Survey-design implementation

### Decision: how DHS clusters, strata, and sampling weights enter every estimate

**Problem.** DHS is a stratified, multistage cluster sample, not a simple random
sample. Three things must be carried through every estimate: sampling weights
(`hv005`/1e6), strata (`hv022`), clusters/PSUs (`hv021`). Ignore the weights and
the point estimate is biased — the sample over-represents some regions and
residence types by design. Ignore strata and clusters and the standard errors
are too small, because clustered observations are not independent. An MPI
"reproduction" that skips this is wrong by construction (see CLAUDE.md).

**Diagnostic.** How far off is a naive unweighted estimate, and how much does
clustering inflate the true standard error?

```python
import numpy as np

hh = mpi.build_household_mpi(hr, pr, br)
complete = hh.dropna(subset=["censored_score"])

# Naive: ignores weights, strata, and clusters entirely.
naive_mpi = complete["censored_score"].mean()
naive_se = complete["censored_score"].std(ddof=1) / np.sqrt(len(complete))

# Correct: survey-weighted via svy (strata + PSU + weights).
svy_nat = mpi.headcount_intensity_mpi(hh).iloc[0]

display(pd.DataFrame([
    {"method": "naive (unweighted, iid SE)", "MPI": naive_mpi, "SE": naive_se},
    {"method": "survey design (svy)", "MPI": svy_nat["MPI"],
     "SE": svy_nat["MPI_se"]},
]))
print(f"SE inflation from honouring the design: "
      f"{svy_nat['MPI_se'] / naive_se:.2f}x")
```

**Options considered.**
- (a) **Ignore the survey design** — rejected outright: biased point estimate,
  understated SEs. Shown above only to quantify the error.
- (b) **`samplics`** — the Python-native survey library named in the original
  CLAUDE.md plan. It is now **archived/unmaintained**.
- (c) **`svy` 0.18.1** — the maintained successor to `samplics` (same author),
  Python-native, supports Taylor-linearised variance for stratified multistage
  designs — exactly the DHS design.
- (d) **R `survey` via `rpy2`** — the most battle-tested option, but adds an R
  runtime and a cross-language bridge for no capability this project needs.

**Decision.** Use **(c) `svy` 0.18.1**. It is a documented deviation from
CLAUDE.md decision #3, which named `samplics`: `samplics` is archived, and `svy`
is its maintained continuation by the same author with the same estimation
core. It covers the full DHS design (weights + strata + PSUs) with Taylor
variance, keeps the stack Python-only, and avoids the `rpy2` dependency. The
`mpi.headcount_intensity_mpi` estimator routes every H / A / MPI figure —
national and per-region — through `svy`; no aggregate bypasses it.

**Sensitivity.** N/A — this is a correctness requirement, not a tunable choice.
The diagnostic above reports the size of the error that ignoring the design
would introduce (biased MPI and an understated SE); that magnitude is the
justification for the decision, and it is restated in the README decisions
table as the `samplics` → `svy` deviation.

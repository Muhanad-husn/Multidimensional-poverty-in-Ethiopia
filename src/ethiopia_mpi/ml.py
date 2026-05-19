"""ML feature-importance overlay — descriptive, not causal.

Trains RandomForest / XGBoost classifiers predicting MPI status (``poor``)
from a household feature set, then computes SHAP attributions. The output is
a *descriptive* picture of which household characteristics most associate
with multidimensional poverty in the data — not a causal model.

TWO NON-NEGOTIABLE CONSTRAINTS (see CLAUDE.md):

1. **Train/test splits use** :class:`sklearn.model_selection.GroupKFold`
   **grouped on the DHS cluster ID** (Decision #2 in the implementation
   plan: ``hv001``, not the women's-recode shorthand ``v001``). Households
   in the same cluster never split across folds — otherwise the model
   trains on the test-set environment and test performance inflates.
2. **The feature set EXCLUDES the 10 OPHI indicators themselves**
   (and the derived ``score`` / ``poor`` / ``censored_score`` columns),
   or the model trivially memorises the target.

SHAP aggregation across households respects DHS sampling weights for any
nationally representative feature-importance claim — see
:func:`shap_summary` (the ``weighted_mean_abs_shap`` column).
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

# DHS cluster ID — the grouping key for cluster-respecting cross-validation.
# Decision #2: the household recode uses the `hv*` family, not `v*`.
CLUSTER_COL = "hv001"
WEIGHT_COL = "hv005"

# The 10 OPHI indicators and the derived MPI columns — *never* enter the
# feature matrix. Building X by exclusion (rather than inclusion) means a
# future contributor who adds a new MPI artefact gets a loud assertion if
# they forget to add it here, instead of a silent leak.
_OPHI_INDICATORS = (
    "nutrition", "child_mortality",
    "years_schooling", "school_attendance",
    "cooking_fuel", "sanitation", "drinking_water", "electricity",
    "housing", "assets",
)
_MPI_DERIVED = ("score", "n_missing_ind", "poor", "censored_score")
EXCLUDED = frozenset(_OPHI_INDICATORS + _MPI_DERIVED)

# Household-recode features kept for the model. Wealth index columns
# (hv270/hv271) are included with an acknowledged caveat — they are
# constructed from many of the same asset/housing items as the
# living-standards OPHI indicators, so they are partially circular. We keep
# them in and flag the caveat in the decision block; expect them to
# dominate SHAP rankings on real DHS data.
_HR_FEATURE_COLS = (
    "hv009",   # number of household members
    "hv025",   # urban (1) / rural (2)
    "hv024",   # region — one-hot encoded below
    "hv040",   # altitude (m)
    "hv246",   # owns livestock
    "hv270",   # wealth quintile (1-5)
    "hv271",   # wealth factor score
)


def build_feature_matrix(
    hr: pd.DataFrame,
    pr: pd.DataFrame,
    mpi_hh: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Assemble household-level features + target, *excluding* OPHI indicators.

    Parameters
    ----------
    hr, pr
        DHS Household and Household-Member recodes.
    mpi_hh
        Household-level MPI frame from :func:`mpi.build_household_mpi` —
        carries the ``poor`` target plus ``hv001`` / ``hv005``.

    Returns
    -------
    X
        Feature matrix indexed by ``(hv001, hv002)``. Region one-hot,
        PR aggregates joined.
    y
        Target ``poor`` ({0.0, 1.0, NaN}) aligned to ``X``.
    groups
        DHS cluster ID (``hv001``) per row — the ``groups`` argument for
        :class:`GroupKFold`.
    weights
        Sampling weight (``hv005 / 1e6``) per row — for survey-aware SHAP
        aggregation downstream.

    Raises
    ------
    AssertionError
        If any OPHI indicator or MPI-derived column leaks into ``X``. The
        check is loud on purpose; this is the non-negotiable rule from
        CLAUDE.md.
    """
    keys = ["hv001", "hv002"]

    # ---- household-level features (HR) -----------------------------------
    have = [c for c in _HR_FEATURE_COLS if c in hr.columns]
    hr_feats = hr.set_index(keys)[have].copy()
    # one-hot the region; preserve a "region" prefix so the SHAP table is readable
    if "hv024" in hr_feats.columns:
        region_oh = pd.get_dummies(
            hr_feats["hv024"].astype("Int64"), prefix="region", dtype="float64"
        )
        hr_feats = hr_feats.drop(columns=["hv024"]).join(region_oh)

    # ---- person-level aggregates (PR) ------------------------------------
    pr_g = pr.groupby(keys)
    sch_adult = pr["hv108"].where(pr["hv105"].ge(15) & pr["hv108"].lt(90))
    pr_agg = pd.DataFrame(
        {
            "n_members_pr": pr_g.size(),
            "mean_age": pr_g["hv105"].mean(),
            "max_age": pr_g["hv105"].max(),
            "share_female": pr_g["hv104"].apply(lambda s: float((s == 2).mean())),
            "share_child": pr_g["hv105"].apply(lambda s: float((s < 15).mean())),
            "mean_schooling_adults": sch_adult.groupby(
                [pr["hv001"], pr["hv002"]]
            ).mean(),
        }
    )

    X = hr_feats.join(pr_agg)

    # ---- align to the MPI household frame --------------------------------
    target = mpi_hh.set_index(keys)
    common = X.index.intersection(target.index)
    X = X.loc[common].copy()
    y = target.loc[common, "poor"].astype("float64")
    # cluster ID lives on the index after set_index; pull from there
    groups = pd.Series(
        common.get_level_values(CLUSTER_COL).astype("int64"),
        index=common, name=CLUSTER_COL,
    )
    weights = (target.loc[common, WEIGHT_COL].astype("float64") / 1e6)

    # ---- the non-negotiable check ----------------------------------------
    leaked = EXCLUDED.intersection(X.columns)
    assert not leaked, (
        f"OPHI indicators or MPI-derived columns leaked into the feature "
        f"matrix: {sorted(leaked)} — the model would trivially memorise the "
        f"target. See CLAUDE.md, 'ML feature selection'."
    )

    return X, y, groups, weights


# ==========================================================================
# Models
# ==========================================================================

ModelKind = Literal["rf", "xgb"]


def _make_model(kind: ModelKind, random_state: int = 42):
    """Construct an untrained classifier. Imports xgboost lazily."""
    if kind == "rf":
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_leaf=5,
            n_jobs=-1,
            random_state=random_state,
            class_weight="balanced",
        )
    if kind == "xgb":
        from xgboost import XGBClassifier

        return XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.9,
            colsample_bytree=0.9,
            n_jobs=-1,
            random_state=random_state,
            eval_metric="logloss",
            tree_method="hist",
        )
    raise ValueError(f"unknown model kind {kind!r}; expected 'rf' or 'xgb'")


def _filter_valid(X, y, groups, weights=None):
    """Drop rows with NaN target; cast y to int."""
    mask = y.notna()
    Xm = X.loc[mask].copy()
    ym = y.loc[mask].astype(int)
    gm = groups.loc[mask]
    wm = weights.loc[mask] if weights is not None else None
    return Xm, ym, gm, wm


# ==========================================================================
# Cluster-respecting cross-validation
# ==========================================================================

def cross_validate_clustered(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model: ModelKind = "rf",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Cluster-respecting CV via :class:`GroupKFold` on ``groups``.

    The non-negotiable rule from CLAUDE.md: households in the same DHS
    cluster (``hv001``) never split across train/test folds, or the model
    is implicitly trained on the test-set environment.

    Returns a dict with per-fold AUC/F1, their mean/std, and the final
    classifier fit on all valid rows.
    """
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import GroupKFold

    Xm, ym, gm, _ = _filter_valid(X, y, groups)
    gkf = GroupKFold(n_splits=n_splits)
    aucs, f1s = [], []
    for tr, te in gkf.split(Xm, ym, groups=gm):
        clf = _make_model(model, random_state)
        clf.fit(Xm.iloc[tr], ym.iloc[tr])
        proba = clf.predict_proba(Xm.iloc[te])[:, 1]
        aucs.append(float(roc_auc_score(ym.iloc[te], proba)))
        f1s.append(float(f1_score(ym.iloc[te], (proba >= 0.5).astype(int))))

    final = _make_model(model, random_state).fit(Xm, ym)
    return {
        "model": final,
        "fold_auc": aucs,
        "fold_f1": f1s,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
        "f1_mean": float(np.mean(f1s)),
        "f1_std": float(np.std(f1s)),
        "n_train": int(len(Xm)),
    }


def leakage_diagnostic(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    model: ModelKind = "rf",
    n_splits: int = 5,
    random_state: int = 42,
) -> dict:
    """Quantify the inflation from ignoring cluster grouping.

    Runs the same classifier under both random :class:`KFold` and clustered
    :class:`GroupKFold`. The gap (random - clustered) is the leakage
    magnitude — it's the empirical justification for the rule.

    Returns
    -------
    dict
        ``random_auc`` / ``clustered_auc`` (per-fold lists), their means,
        and the ``gap``. A positive gap means random-CV is over-optimistic
        — the expected direction.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import GroupKFold, KFold

    Xm, ym, gm, _ = _filter_valid(X, y, groups)

    def _cv(splitter, **split_kw) -> list[float]:
        scores = []
        for tr, te in splitter.split(Xm, ym, **split_kw):
            clf = _make_model(model, random_state)
            clf.fit(Xm.iloc[tr], ym.iloc[tr])
            proba = clf.predict_proba(Xm.iloc[te])[:, 1]
            scores.append(float(roc_auc_score(ym.iloc[te], proba)))
        return scores

    rkf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    gkf = GroupKFold(n_splits=n_splits)
    rand = _cv(rkf)
    grp = _cv(gkf, groups=gm)
    return {
        "random_auc": rand,
        "clustered_auc": grp,
        "random_mean": float(np.mean(rand)),
        "clustered_mean": float(np.mean(grp)),
        "gap": float(np.mean(rand) - np.mean(grp)),
        "n_train": int(len(Xm)),
    }


# ==========================================================================
# SHAP — weight-aware aggregation
# ==========================================================================

def shap_summary(
    model,
    X: pd.DataFrame,
    sample_weight: pd.Series | None = None,
    top_n: int | None = 15,
) -> pd.DataFrame:
    """Top-N SHAP feature importance with optional survey-weight aggregation.

    For each feature j, the mean absolute SHAP value across the rows in
    ``X`` is the standard descriptive feature-importance summary. When the
    rows represent a survey sample, the unweighted mean does not give a
    nationally representative answer — sampling weights matter for the
    summary statistic, not just the point estimates. ``weighted_mean_abs_shap``
    aggregates with the supplied weights; ``mean_abs_shap`` is the
    unweighted comparator (their ratio surfaces the size of the weighting
    correction).

    Parameters
    ----------
    model
        Fitted tree model — Random Forest or XGBoost (anything supported by
        :class:`shap.TreeExplainer`).
    X
        Feature matrix the SHAP attributions are computed on.
    sample_weight
        Per-row DHS sampling weight (e.g. ``hv005 / 1e6``). If ``None``, the
        weighted column equals the unweighted one.
    top_n
        Truncate to the top-N features by ``weighted_mean_abs_shap``.
        Pass ``None`` to keep all features.

    Returns
    -------
    DataFrame with columns ``feature``, ``mean_abs_shap``,
    ``weighted_mean_abs_shap``, ``rank`` (1-indexed by weighted importance).
    """
    import shap

    explainer = shap.TreeExplainer(model)
    raw = explainer.shap_values(X)

    # Normalise across model APIs:
    #  - XGBClassifier (binary): ndarray (n, n_features)
    #  - RandomForestClassifier (binary, shap >= 0.45): ndarray (n, n_features, n_classes)
    #  - older shap RF: list[ndarray] of length n_classes
    if isinstance(raw, list):
        vals = np.asarray(raw[1])
    else:
        arr = np.asarray(raw)
        vals = arr[:, :, 1] if arr.ndim == 3 else arr

    abs_vals = np.abs(vals)
    n = len(X)
    w = (
        np.asarray(sample_weight, dtype="float64")
        if sample_weight is not None
        else np.ones(n)
    )
    if len(w) != n:
        raise ValueError(f"sample_weight length {len(w)} != X rows {n}")

    unweighted = abs_vals.mean(axis=0)
    weighted = (abs_vals * w[:, None]).sum(axis=0) / w.sum()

    out = (
        pd.DataFrame(
            {
                "feature": list(X.columns),
                "mean_abs_shap": unweighted,
                "weighted_mean_abs_shap": weighted,
            }
        )
        .sort_values("weighted_mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    out["rank"] = np.arange(1, len(out) + 1)
    if top_n is not None:
        out = out.head(top_n).reset_index(drop=True)
    return out

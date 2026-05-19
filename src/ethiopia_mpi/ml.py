"""ML feature-importance overlay — descriptive, not causal.

Trains RandomForest / XGBoost classifiers predicting MPI status from a wider
household feature set, then computes SHAP attributions.

TWO NON-NEGOTIABLE CONSTRAINTS (see CLAUDE.md):
  1. Train/test splits use ``GroupKFold`` grouped on the DHS cluster ID (v001).
     Households in the same cluster never split across folds — otherwise the
     model trains on the test-set environment and test performance inflates.
  2. The feature set EXCLUDES the 10 OPHI indicators themselves, or the model
     trivially memorizes the target.

SHAP aggregation across households respects DHS sampling weights for any
nationally representative feature-importance claim.

Status: implemented in Session 5 (see IMPLEMENTATION_PLAN.md).
"""

from __future__ import annotations

# DHS cluster ID — the grouping key for cluster-respecting cross-validation.
CLUSTER_COL = "v001"

# TODO Session 5: build_feature_matrix() — excludes the 10 OPHI indicators
# TODO Session 5: train_classifier() with GroupKFold(groups=df[CLUSTER_COL])
# TODO Session 5: random-vs-clustered CV diagnostic (the leakage gap)
# TODO Session 5: shap_summary() — weighted by DHS sampling weight

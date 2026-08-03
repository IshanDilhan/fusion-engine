"""
ONE-OFF DIAGNOSTIC EXPERIMENT -- not part of the production pipeline, does
NOT modify clip_features.parquet, splits.csv, or any shipped dataset file,
and does NOT replace the registered "fusion-gbt" model.

Question: is GBT's catastrophic F05 recall (0.175) really just a training-
data coverage gap, or a deeper fusion-capability limitation? scenarios.csv
has exactly one both_hands_up+Neutral scenario (the only training signal
for F05's "both_hands_up" rule branch) -- S26_F05 (v3_row 26) -- and it's
test-only. Every both_hands_up clip GBT has ever trained on is F02 or F07
(verified: 205 train+dev clips, 0 of them F05). This script adds S26_F05's
clips to the training pool for ONE experimental fit, in memory only, and
checks whether that's enough for GBT to learn the pattern -- and, more
importantly, whether it GENERALIZES to F05's OTHER test scenario (S57_F05,
28 clips, still genuinely held out), not just memorizes S26_F05 itself.

Reports three numbers, not one, because a single "test accuracy" here would
be misleading:
  1. Whole-test-set accuracy (as usual) -- CAVEAT: no longer a blind measure
     for the S26_F05 clips specifically, since they're now in training too.
  2. Test accuracy EXCLUDING S26_F05 -- a fair like-for-like comparison
     against the production model's normal test accuracy.
  3. F05 recall on S57_F05 ONLY -- the genuinely still-unseen scenario; this
     is the number that actually answers the coverage-vs-capability
     question, since nothing about S57_F05 was touched.

Logged to MLflow under a clearly-tagged experimental run name, separate
from the "gbt" run fusion/gbt.py produces -- does not register a model.
"""
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from aggregate import FEATURE_NAMES  # noqa: E402
sys.path.insert(0, os.path.join(REPO_ROOT, "fusion"))
from rule_based import predict_intent  # noqa: E402

FEATURES_PATH = os.path.join(REPO_ROOT, "data", "features", "clip_features.parquet")
RANDOM_SEED = 42
COVERAGE_GAP_SCENARIO = "S26_F05"  # v3_row 26 -- the only both_hands_up+Neutral (F05) scenario, shipped test-only

# Matches fusion/gbt.py's current tuned production config (see tune_gbt.py).
MODEL_PARAMS = dict(
    objective="multiclass", class_weight="balanced",
    n_estimators=50, max_depth=3, learning_rate=0.05,
    subsample=0.9, colsample_bytree=0.7,
    random_state=RANDOM_SEED, verbosity=-1,
)


def apply_modality_dropout_simple(X, y, rng, dropout_p=0.15, max_dropped=2):
    """Same shape as fusion/gbt.py's apply_modality_dropout, duplicated here
    (not imported) to keep this throwaway script fully self-contained and
    not create a dependency from a permanent module onto a one-off script."""
    from rule_based import predict_intent as _predict_intent
    cue_blocks = {
        "emotion": [c for c in FEATURE_NAMES if c.startswith("emotion_")],
        "gesture": [c for c in FEATURE_NAMES if c.startswith("gesture_")],
        "motion": [c for c in FEATURE_NAMES if c.startswith("motion_")],
        "context": [c for c in FEATURE_NAMES if c.startswith("context_")],
    }
    X = X.copy()
    y = y.copy()
    for idx in X.index:
        drawn = [cue for cue in cue_blocks if rng.random() < dropout_p]
        if len(drawn) > max_dropped:
            drawn = list(rng.choice(drawn, size=max_dropped, replace=False))
        if not drawn:
            continue
        for cue in drawn:
            X.loc[idx, cue_blocks[cue]] = 0.0
            X.loc[idx, f"missing_{cue}"] = 1.0
        if len(drawn) >= 2 and _predict_intent(X.loc[idx], fallback_intent="__NO_RULE_MATCH__") != y.loc[idx]:
            y.loc[idx] = "F05"
    return X, y


def main():
    import mlflow

    from tracking.mlflow_setup import init_tracking

    df = pd.read_parquet(FEATURES_PATH)
    train_df = df[df["split_design_v2"].isin(["train", "dev"])].copy()
    test_df = df[df["split_design_v2"] == "test"].reset_index(drop=True)

    gap_clips = df[df["scenario_dir"] == COVERAGE_GAP_SCENARIO]
    print(f"[experiment] {COVERAGE_GAP_SCENARIO}: {len(gap_clips)} clips, "
          f"currently split_design_v2={gap_clips['split_design_v2'].unique().tolist()}")
    assert (gap_clips["split_design_v2"] == "test").all(), \
        f"{COVERAGE_GAP_SCENARIO} is expected to be test-only in the shipped split"

    experimental_train_df = pd.concat([train_df, gap_clips], ignore_index=True)
    print(f"[experiment] experimental train pool: {len(train_df)} -> {len(experimental_train_df)} clips "
          f"(+{len(gap_clips)} from {COVERAGE_GAP_SCENARIO})")

    rng = np.random.default_rng(RANDOM_SEED)
    X_train, y_train = apply_modality_dropout_simple(
        experimental_train_df[FEATURE_NAMES], experimental_train_df["intent"], rng)

    model = LGBMClassifier(**MODEL_PARAMS)
    model.fit(X_train, y_train)

    preds_all = model.predict(test_df[FEATURE_NAMES])
    acc_all = (preds_all == test_df["intent"].values).mean()

    not_gap_mask = test_df["scenario_dir"].values != COVERAGE_GAP_SCENARIO
    acc_excl_gap = (preds_all[not_gap_mask] == test_df["intent"].values[not_gap_mask]).mean()

    s57_mask = test_df["scenario_dir"].values == "S57_F05"
    f05_s57_recall = (preds_all[s57_mask] == "F05").mean() if s57_mask.any() else float("nan")

    f05_mask_all = test_df["intent"].values == "F05"
    f05_recall_all = (preds_all[f05_mask_all] == "F05").mean()

    print(f"\n[experiment] whole-test accuracy (n={len(test_df)}, includes now-non-blind "
          f"{COVERAGE_GAP_SCENARIO} clips): {acc_all:.3f}")
    print(f"[experiment] test accuracy EXCLUDING {COVERAGE_GAP_SCENARIO} "
          f"(n={not_gap_mask.sum()}, fair comparison to production): {acc_excl_gap:.3f}")
    print(f"[experiment] F05 recall, whole test set (n={f05_mask_all.sum()}): {f05_recall_all:.3f}")
    print(f"[experiment] F05 recall on S57_F05 ONLY (n={s57_mask.sum()}, genuinely still unseen "
          f"-- the number that actually answers the coverage question): {f05_s57_recall:.3f}")

    init_tracking()
    with mlflow.start_run(run_name="gbt_f05_coverage_experiment"):
        mlflow.set_tag("model_type", "gbt_experimental_diagnostic")
        mlflow.set_tag("experimental", "true")
        mlflow.set_tag("not_for_production", "true")
        mlflow.set_tag("note", f"{COVERAGE_GAP_SCENARIO} added to training pool as a one-off diagnostic; "
                                "no dataset files modified; not registered as a model")
        mlflow.log_param("coverage_gap_scenario", COVERAGE_GAP_SCENARIO)
        mlflow.log_params(MODEL_PARAMS)
        mlflow.log_metric("test_accuracy_whole_nonblind", acc_all)
        mlflow.log_metric("test_accuracy_excl_gap_scenario", acc_excl_gap)
        mlflow.log_metric("f05_recall_whole_test", f05_recall_all)
        mlflow.log_metric("f05_recall_s57_only_held_out", f05_s57_recall)

    print(f"\n[experiment] verdict: {'coverage WAS the bottleneck -- generalizes to the still-unseen scenario' if f05_s57_recall > 0.3 else 'does NOT generalize to the still-unseen scenario -- coverage alone is not sufficient'}")


if __name__ == "__main__":
    main()

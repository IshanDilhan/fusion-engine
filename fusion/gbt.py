"""
Phase 4 GBT fusion -- PROVISIONAL (emotion + gesture cue models are expected
to be replaced next, same as motion already was; this model and its
accuracy number are a moving baseline, not the locked deliverable yet).

LightGBM multiclass classifier over the Phase-2 feature vector
(pipeline/aggregate.py's 33 cue-derived columns), trained/evaluated on
pipeline/build_features.py's split_design_v2 partition (train/dev/test,
grouped by scenario and by video-take so no clip-level near-duplicate
straddles a split boundary -- see build_features.py's assign_dev_split()).

Implements, per the dataset's own design doc (v2.0.0's shipped
final_dataset_merged.docx):
  - Class weighting (`class_weight="balanced"`) -- F02 is never down-weighted
    by construction (balanced weighting up-weights rarer classes, and F02 is
    one of the more common ones here, so this does not suppress it either).
  - Modality-dropout augmentation during training: for each training row,
    with probability DROPOUT_P per cue (capped at MAX_DROPPED_CUES), zero
    that cue's block and set its missing bit, relabeling to RELABEL_INTENT
    when 2+ cues drop and the survivors no longer identify the true intent
    -- teaches the model to redistribute weight onto the remaining cues
    instead of only ever seeing this dataset's real (but per-clip, not
    per-row) missingness (see build_features.py's apply_cue_mask()).
  - Safety override: if the model's predicted F02 probability exceeds
    F02_SAFETY_THRESHOLD, classify as F02 regardless of argmax.

NOT yet implemented in this provisional pass (explicitly deferred, not
silently skipped): isotonic/Platt calibration, SHAP per-prediction
attribution. Both are meaningful follow-up work once the emotion/gesture
cue models stop moving.
"""
import os
import sys

import mlflow.pyfunc
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from aggregate import FEATURE_NAMES  # noqa: E402
sys.path.insert(0, os.path.join(REPO_ROOT, "fusion"))
from rule_based import predict_all, fit_fallback, predict_intent  # noqa: E402

FEATURES_PATH = os.path.join(REPO_ROOT, "data", "features", "clip_features.parquet")

CUE_BLOCKS = {
    "emotion": [c for c in FEATURE_NAMES if c.startswith("emotion_")],
    "gesture": [c for c in FEATURE_NAMES if c.startswith("gesture_")],
    "motion": [c for c in FEATURE_NAMES if c.startswith("motion_")],
    "context": [c for c in FEATURE_NAMES if c.startswith("context_")],
}
DROPOUT_P = 0.15  # per-cue, per-row probability of simulated dropout during training
MAX_DROPPED_CUES = 2  # dataset docx spec: "randomly mask each cue with p~0.15, max 1-2 cues per sample"
RELABEL_INTENT = "F05"  # safe default when a heavily-degraded row no longer identifies its true intent
F02_SAFETY_THRESHOLD = 0.15
RANDOM_SEED = 42


def apply_modality_dropout(X: pd.DataFrame, y: pd.Series, rng: np.random.Generator):
    """Per training row, independently rolls DROPOUT_P per cue block, capped
    at MAX_DROPPED_CUES per row. When 2+ cues get dropped and the surviving
    cues no longer identify the row's true intent (checked via rule_based's
    own predict_intent on the post-dropout row, with a sentinel fallback so
    "no rule matched" can't spuriously look like a match), the row's label
    is relabeled to RELABEL_INTENT -- teaches the model to fall back safely
    on heavily-degraded input rather than confidently guess wrong.

    NOT applied: the dataset docx's "unless direction still identifies
    F02/F06" carve-out -- gesture direction/point_target were dropped from
    FEATURE_NAMES entirely (pipeline/aggregate.py's own docstring: they are
    hardcoded constants upstream, carrying no real signal), so there is no
    operable "direction" cue here to check against. Known simplification,
    not a silent omission.
    """
    X = X.copy()
    y = y.copy()
    cue_names = list(CUE_BLOCKS.keys())
    n_relabeled = 0

    for idx in X.index:
        drawn = [cue for cue in cue_names if rng.random() < DROPOUT_P]
        if len(drawn) > MAX_DROPPED_CUES:
            drawn = list(rng.choice(drawn, size=MAX_DROPPED_CUES, replace=False))
        if not drawn:
            continue

        for cue in drawn:
            X.loc[idx, CUE_BLOCKS[cue]] = 0.0
            X.loc[idx, f"missing_{cue}"] = 1.0

        if len(drawn) >= 2 and predict_intent(X.loc[idx], fallback_intent="__NO_RULE_MATCH__") != y.loc[idx]:
            y.loc[idx] = RELABEL_INTENT
            n_relabeled += 1

    print(f"[gbt] modality dropout: {n_relabeled} rows relabeled to {RELABEL_INTENT} "
          f"(>=2 cues dropped, remaining cues no longer identify true intent)")
    return X, y


def predict_with_safety_override(model, X, f02_idx):
    proba = model.predict_proba(X)
    argmax_idx = proba.argmax(axis=1)
    preds = model.classes_[argmax_idx]
    escalate = proba[:, f02_idx] >= F02_SAFETY_THRESHOLD
    preds = np.where(escalate, "F02", preds)
    return preds, proba


class GBTFusionModel(mlflow.pyfunc.PythonModel):
    """Wraps the fitted LGBMClassifier + F02 safety override as a loadable
    pyfunc model, so predict() returns exactly the intent predictions used
    to compute this run's accuracy metrics -- not the raw LightGBM argmax,
    which the safety override can outrank.

    model_input must contain aggregate.FEATURE_NAMES columns.
    """

    def __init__(self, lgbm_model):
        self.lgbm_model = lgbm_model

    def predict(self, context, model_input, params=None):
        X = model_input[FEATURE_NAMES]
        f02_idx = list(self.lgbm_model.classes_).index("F02")
        preds, _ = predict_with_safety_override(self.lgbm_model, X, f02_idx)
        return pd.Series(preds, index=model_input.index)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mlflow.models import infer_signature

    from tracking.dataset_logging import log_dataset
    from tracking.hashing import sha256_file
    from tracking.metrics import log_overall_metrics
    from tracking.mlflow_setup import init_tracking

    init_tracking()
    with mlflow.start_run(run_name="gbt"):
        mlflow.set_tag("model_type", "gbt")
        mlflow.set_tag("code_file", os.path.relpath(__file__, REPO_ROOT))
        mlflow.set_tag("code_version_sha256", sha256_file(__file__))

        df = log_dataset(context="training", allow_stale_override=True)
        all_classes = sorted(df["intent"].unique())
        train_df = df[df["split_design_v2"] == "train"].reset_index(drop=True)
        dev_df = df[df["split_design_v2"] == "dev"].reset_index(drop=True)
        test_df = df[df["split_design_v2"] == "test"].reset_index(drop=True)
        mlflow.log_param("n_train", len(train_df))
        mlflow.log_param("n_dev", len(dev_df))
        mlflow.log_param("n_test", len(test_df))

        rng = np.random.default_rng(RANDOM_SEED)
        X_train, y_train = apply_modality_dropout(train_df[FEATURE_NAMES], train_df["intent"], rng)

        # Selected via fusion/tune_gbt.py's scenario-grouped (StratifiedGroupKFold
        # by scenario_dir) 5-fold CV sweep over the train+dev pool -- the shipped
        # dev split isn't scenario-disjoint, so tuning against it (previously
        # 88.1% dev vs 48.1% test) optimized for the wrong kind of generalization.
        # Under CV, this config scores marginally better than the old 300/depth-5
        # default (macro F1 0.273 vs 0.264, within noise) -- capacity reduction
        # alone is not a full fix (see tune_gbt.py's logistic-regression sanity
        # check, which beats both), but it's a strict improvement with no
        # downside, so it replaces the old default.
        model_params = dict(
            objective="multiclass",
            class_weight="balanced",
            n_estimators=50,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.7,
            random_state=RANDOM_SEED,
            verbosity=-1,
        )
        mlflow.log_params(model_params)
        mlflow.log_param("modality_dropout_p", DROPOUT_P)
        mlflow.log_param("f02_safety_threshold", F02_SAFETY_THRESHOLD)

        model = LGBMClassifier(**model_params)
        model.fit(X_train, y_train)
        f02_idx = list(model.classes_).index("F02")

        print(f"[gbt] trained on {len(train_df)} clips, {len(FEATURE_NAMES)} features, "
              f"modality dropout p={DROPOUT_P}, F02 safety threshold={F02_SAFETY_THRESHOLD}")

        fallback = fit_fallback(train_df)
        mlflow.log_param("rule_comparison_fallback_intent", fallback)
        rule_preds_all = predict_all(df, fallback_intent=fallback)
        df["rule_pred"] = rule_preds_all

        recall_rows = []
        for split_name, split_df in [("train", train_df), ("dev", dev_df), ("test", test_df)]:
            if len(split_df) == 0:
                continue
            X = split_df[FEATURE_NAMES]
            preds, proba = predict_with_safety_override(model, X, f02_idx)
            acc = (preds == split_df["intent"].values).mean()
            mlflow.log_metric(f"{split_name}_accuracy", acc)

            rule_sub = df[df["clip_id"].isin(split_df["clip_id"])]
            rule_acc = (rule_sub["rule_pred"] == rule_sub["intent"]).mean()
            mlflow.log_metric(f"rule_comparison_{split_name}_accuracy", rule_acc)

            print(f"\n[gbt] split_design_v2={split_name}: n={len(split_df)}")
            print(f"  GBT accuracy:  {acc:.3f}")
            print(f"  rule accuracy: {rule_acc:.3f}  (same clips, for direct comparison)")
            log_overall_metrics(mlflow, split_df["intent"].values, preds, all_classes,
                                 split_name, print_prefix=f"[gbt]   {split_name}")

            scoreable_mask = split_df["scoreable"].values == "TRUE"
            if scoreable_mask.any():
                acc_scoreable = (preds[scoreable_mask] == split_df["intent"].values[scoreable_mask]).mean()
                mlflow.log_metric(f"{split_name}_accuracy_scoreable_only", acc_scoreable)
                print(f"  GBT accuracy (scoreable-only, n={scoreable_mask.sum()}): {acc_scoreable:.3f}")
                log_overall_metrics(mlflow, split_df["intent"].values[scoreable_mask], preds[scoreable_mask],
                                     all_classes, f"{split_name}_scoreable_only",
                                     print_prefix=f"[gbt]   {split_name} scoreable-only")

            if split_name == "test":
                print("\n[gbt] per-class recall (test):")
                for cls in sorted(df["intent"].unique()):
                    mask = split_df["intent"].values == cls
                    n = mask.sum()
                    if n == 0:
                        continue
                    recall = (preds[mask] == cls).mean()
                    mlflow.log_metric(f"test_recall_{cls}", recall)
                    recall_rows.append({"intent": cls, "n": int(n), "recall": recall})
                    print(f"  {cls}: n={n}, recall={recall:.3f}")

                f02_mask = split_df["intent"].values == "F02"
                f02_recall = (preds[f02_mask] == "F02").mean() if f02_mask.sum() else float("nan")
                f02_false_neg = int((f02_mask & (preds != "F02")).sum())
                mlflow.log_metric("test_f02_recall", f02_recall if f02_mask.sum() else 0.0)
                mlflow.log_metric("test_f02_false_negatives", f02_false_neg)
                print(f"\n[gbt] F02 test recall: {f02_recall:.3f} ({f02_false_neg} false negatives / {f02_mask.sum()} true F02 clips)")

        recall_df = pd.DataFrame(recall_rows)
        mlflow.log_table(data=recall_df, artifact_file="reports/test_per_class_recall.json")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(recall_df["intent"], recall_df["recall"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("recall")
        ax.set_title("gbt -- per-class test recall")
        mlflow.log_figure(fig, "reports/test_per_class_recall.png")
        plt.close(fig)

        print("\n[gbt] feature importances (top 10, gain):")
        importances = pd.Series(model.feature_importances_, index=FEATURE_NAMES).sort_values(ascending=False)
        print(importances.head(10))

        importances_df = importances.reset_index()
        importances_df.columns = ["feature", "importance_gain"]
        mlflow.log_table(data=importances_df, artifact_file="reports/feature_importances.json")

        fig2, ax2 = plt.subplots(figsize=(8, 5))
        top15 = importances.head(15)
        ax2.barh(top15.index[::-1], top15.values[::-1])
        ax2.set_xlabel("importance (gain)")
        ax2.set_title("gbt -- top 15 feature importances")
        fig2.tight_layout()
        mlflow.log_figure(fig2, "reports/feature_importances.png")
        plt.close(fig2)

        signature_input = train_df[FEATURE_NAMES]
        sig_preds, _ = predict_with_safety_override(model, signature_input, f02_idx)
        signature = infer_signature(signature_input, pd.Series(sig_preds))
        mlflow.pyfunc.log_model(
            name="model",
            python_model=GBTFusionModel(model),
            signature=signature,
            input_example=signature_input.head(5),
            registered_model_name="fusion-gbt",
        )


if __name__ == "__main__":
    main()

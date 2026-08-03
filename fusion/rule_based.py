"""
Phase 3 rule-based baseline -- PROVISIONAL (emotion + gesture cue models are
expected to be replaced, same as motion already was; this baseline and its
accuracy number are a moving target, not a final result).

Encodes scenarios.csv's own authoring logic as explicit priority-ordered
IF-THEN rules. As of dataset v2.0.0, this table is DERIVED, not hand-
transcribed: run pipeline/derive_rule_table.py against the current
scenarios.csv (62 rows, machine-readable emotion_v3/gesture_v3/motion_v3/
context -> intent columns) to reproduce the table below and check it hasn't
gone stale. This is what fusion/gbt.py must beat.

v1.0.0's two documented irreducible ambiguities (F02/F07 distinguished only
by scene; F04/F10 an unresolvable tie) are BOTH RESOLVED in v2's richer,
62-scenario table (derive_rule_table.py reports 0 ambiguous combinations):
  - F02 vs F07 is now emotion-dependent, not scene-dependent: both_hands_up
    + (Fear/Surprise) -> F02, both_hands_up + Anger -> F07, identically in
    both contexts.
  - F04 vs F10 is now resolved by idle vs. active gesture: Sad + idle -> F10,
    Sad + thumbs_down/beckoning -> F04. ("idle" is gesture_runner.py's
    Unknown-but-valid state -- a hand/person present but not gesturing --
    distinct from gesture genuinely missing; see gesture_runner.py's own
    docstring on this distinction, and note this branch is unreachable if
    that distinction isn't measured correctly.)
  - F09 no longer exists as an intent class (v2's scenario table folded the
    old "farewell" pattern into F01 -- confirmed absent from clips.csv/
    scenarios.csv's intent values). wave is now purely emotion-dispatched.

Reads a clip's Phase 2 feature row (pipeline/aggregate.py's FEATURE_NAMES)
and returns a predicted intent code. Branches are grouped by gesture (the
table's natural partition), most emergency-relevant first; anything not
covered below (combinations absent from the 62-row table) falls through to
the corpus-mode fallback, same as v1 -- this file does not extrapolate rules
for combinations the data never showed.
"""
import os
import sys

import mlflow.pyfunc
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from aggregate import EMOTION_CLASSES, GESTURE_CLASSES, MOTION_CLASSES, CONTEXT_CLASSES, FEATURE_NAMES  # noqa: E402

DEFAULT_FALLBACK_INTENT = "F05"  # overridden by fit_fallback() with the training corpus's mode


def _dominant(row, prefix, classes):
    """Reads the one-hot/mean-prob block back out as a single dominant label
    (argmax), or None if the cue is flagged missing for this clip."""
    if row.get(f"missing_{prefix}", 0.0) >= 1.0:
        return None
    vals = [row[f"{prefix}_{c}"] for c in classes]
    if max(vals) <= 0.0:
        return None
    return classes[int(np.argmax(vals))]


def predict_intent(row, fallback_intent=DEFAULT_FALLBACK_INTENT):
    emotion = _dominant(row, "emotion", EMOTION_CLASSES)
    gesture = _dominant(row, "gesture", GESTURE_CLASSES)
    motion = _dominant(row, "motion", MOTION_CLASSES)

    # 1. both_hands_up: emotion-dependent (see module docstring -- this
    #    replaces v1's scene-dependent carve-out). Fear/Surprise/anything
    #    else not observed in the data escalates to F02 (emergency); Anger
    #    is frustration (F07), not danger; Neutral is not an emergency at
    #    all (F05, e.g. a startled-but-unbothered stretch).
    if gesture == "both_hands_up":
        if emotion == "Anger":
            return "F07"
        if emotion == "Neutral":
            return "F05"
        return "F02"

    # 2. Unknown-but-valid gesture ("idle" -- a person/hand present but not
    #    gesturing, distinct from gesture genuinely missing below). Resolves
    #    v1's F04/F10 tie: Sad + idle is F10 (discouraged, no directed
    #    signal); Sad + an active gesture (thumbs_down/beckoning, below)
    #    is F04 instead.
    if gesture == "Unknown":
        if emotion == "Sad":
            return "F10"
        if emotion == "Fear":
            return "F02"
        if emotion == "Neutral":
            return "F05"

    # 3. thumbs_down / thumbs_up: purely emotion-dispatched, context-free.
    if gesture in ("thumbs_down", "thumbs_up"):
        if emotion == "Sad":
            return "F04"
        if emotion == "Anger":
            return "F07"
        if emotion == "Disgust":
            return "F08"
        if emotion == "Happy":
            return "F01"

    # 4. raise_hand: Happy is context-dependent (kitchen=positive greeting,
    #    classroom=quietly focused/not to be interrupted); Neutral or
    #    missing-emotion is a help request regardless of context.
    if gesture == "raise_hand":
        context = _dominant(row, "context", CONTEXT_CLASSES)
        if emotion == "Happy":
            return "F01" if context == "kitchen" else "F05"
        if emotion in ("Neutral", None):
            return "F04"

    # 5. beckoning: Neutral is a task summons; Sad is a help request.
    if gesture == "beckoning":
        if emotion == "Neutral":
            return "F03"
        if emotion == "Sad":
            return "F04"

    # 6. point: emotion (+ motion for Anger) dependent.
    if gesture == "point":
        if emotion == "Anger":
            if motion == "walking":
                return "F06"
            if motion == "standing":
                return "F07"
        if emotion == "Disgust":
            return "F06"
        if emotion in ("Happy", "Neutral"):
            return "F03"

    # 7. wave: Anger is frustration (F06, wants the robot to back off);
    #    every other observed emotion (Happy/Neutral/Sad/missing) is F01 --
    #    F09 ("farewell") no longer exists as a separate class in v2.
    if gesture == "wave":
        return "F06" if emotion == "Anger" else "F01"

    # 8. Gesture genuinely missing (not idle -- _dominant() returned None,
    #    i.e. missing_gesture is set or nothing scored above zero).
    if gesture is None:
        if emotion == "Fear":
            return "F02"
        if emotion == "Sad" and motion == "sitting":
            return "F04"
        if emotion in ("Neutral", None) and motion == "walking":
            return "F06"

    return fallback_intent


def fit_fallback(train_df):
    """Corpus-mode fallback for clips matching none of the above rules."""
    return train_df["intent"].mode().iloc[0]


def predict_all(df, fallback_intent=DEFAULT_FALLBACK_INTENT):
    return df.apply(lambda row: predict_intent(row, fallback_intent), axis=1)


class RuleBasedFusionModel(mlflow.pyfunc.PythonModel):
    """Wraps predict_intent() as a loadable pyfunc model so the rule-based
    baseline has the same predict(model_input) interface as fusion/gbt.py's
    logged model, enabling direct comparison/serving via the Model Registry.

    model_input must contain aggregate.FEATURE_NAMES columns (the same
    clip_features.parquet schema _dominant() reads from).
    """

    def __init__(self, fallback_intent):
        self.fallback_intent = fallback_intent

    def predict(self, context, model_input, params=None):
        return model_input[FEATURE_NAMES].apply(
            lambda row: predict_intent(row, self.fallback_intent), axis=1
        )


if __name__ == "__main__":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mlflow.models import infer_signature

    from tracking.dataset_logging import log_dataset
    from tracking.hashing import sha256_file
    from tracking.metrics import log_overall_metrics
    from tracking.mlflow_setup import init_tracking

    init_tracking()
    with mlflow.start_run(run_name="rule_based"):
        mlflow.set_tag("model_type", "rule_based")
        mlflow.set_tag("code_file", os.path.relpath(__file__, REPO_ROOT))
        mlflow.set_tag("code_version_sha256", sha256_file(__file__))

        df = log_dataset(context="training", allow_stale_override=True)
        all_classes = sorted(df["intent"].unique())

        train_df = df[df["split_design_v2"] == "train"]
        fallback = fit_fallback(train_df)
        mlflow.log_param("fallback_intent", fallback)
        mlflow.log_param("default_fallback_intent_constant", DEFAULT_FALLBACK_INTENT)
        print(f"[rule_based] fallback intent (train-set mode): {fallback}")

        preds = predict_all(df, fallback_intent=fallback)
        df["rule_pred"] = preds
        overall_acc = (df["rule_pred"] == df["intent"]).mean()
        mlflow.log_metric("overall_accuracy_all_clips", overall_acc)
        print(f"[rule_based] overall accuracy (all {len(df)} clips, includes train -- not a test number): {overall_acc:.3f}")
        log_overall_metrics(mlflow, df["intent"], df["rule_pred"], all_classes,
                             "overall_accuracy_all_clips", print_prefix="[rule_based] overall")

        scoreable_df = df[df["scoreable"] == "TRUE"]
        overall_acc_scoreable = (scoreable_df["rule_pred"] == scoreable_df["intent"]).mean()
        mlflow.log_metric("overall_accuracy_all_clips_scoreable_only", overall_acc_scoreable)
        print(f"[rule_based] overall accuracy (scoreable-only, n={len(scoreable_df)}): {overall_acc_scoreable:.3f}")
        log_overall_metrics(mlflow, scoreable_df["intent"], scoreable_df["rule_pred"], all_classes,
                             "overall_accuracy_all_clips_scoreable_only", print_prefix="[rule_based] overall scoreable-only")

        for split_name in ["train", "dev", "test"]:
            sub = df[df["split_design_v2"] == split_name]
            if len(sub) == 0:
                continue
            acc = (sub["rule_pred"] == sub["intent"]).mean()
            mlflow.log_metric(f"{split_name}_accuracy", acc)
            mlflow.log_param(f"n_{split_name}", len(sub))
            print(f"[rule_based] split_design_v2={split_name}: n={len(sub)}, accuracy={acc:.3f}")
            log_overall_metrics(mlflow, sub["intent"], sub["rule_pred"], all_classes,
                                 split_name, print_prefix=f"[rule_based]   {split_name}")

            sub_scoreable = sub[sub["scoreable"] == "TRUE"]
            if len(sub_scoreable) > 0:
                acc_scoreable = (sub_scoreable["rule_pred"] == sub_scoreable["intent"]).mean()
                mlflow.log_metric(f"{split_name}_accuracy_scoreable_only", acc_scoreable)
                print(f"[rule_based]   scoreable-only: n={len(sub_scoreable)}, accuracy={acc_scoreable:.3f}")
                log_overall_metrics(mlflow, sub_scoreable["intent"], sub_scoreable["rule_pred"], all_classes,
                                     f"{split_name}_scoreable_only", print_prefix=f"[rule_based]   {split_name} scoreable-only")

        print("\n[rule_based] per-class recall (split_design_v2=test):")
        test_df = df[df["split_design_v2"] == "test"]
        recall_rows = []
        for cls in sorted(df["intent"].unique()):
            sub = test_df[test_df["intent"] == cls]
            if len(sub) == 0:
                continue
            recall = (sub["rule_pred"] == cls).mean()
            mlflow.log_metric(f"test_recall_{cls}", recall)
            recall_rows.append({"intent": cls, "n": len(sub), "recall": recall})
            print(f"  {cls}: n={len(sub)}, recall={recall:.3f}")

        recall_df = pd.DataFrame(recall_rows)
        mlflow.log_table(data=recall_df, artifact_file="reports/test_per_class_recall.json")

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(recall_df["intent"], recall_df["recall"])
        ax.set_ylim(0, 1)
        ax.set_ylabel("recall")
        ax.set_title("rule_based -- per-class test recall")
        mlflow.log_figure(fig, "reports/test_per_class_recall.png")
        plt.close(fig)

        signature_input = train_df[FEATURE_NAMES]
        signature = infer_signature(signature_input, predict_all(train_df, fallback_intent=fallback))
        mlflow.pyfunc.log_model(
            name="model",
            python_model=RuleBasedFusionModel(fallback_intent=fallback),
            signature=signature,
            input_example=signature_input.head(5),
            registered_model_name="fusion-rule-based",
        )

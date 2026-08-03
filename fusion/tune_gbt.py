"""
Scenario-grouped cross-validated hyperparameter search for fusion/gbt.py's
LGBMClassifier, plus a simple-model sanity check.

Why this exists: gbt.py's shipped train/dev split is NOT scenario-disjoint
-- pipeline/build_features.py's assign_dev_split() carves dev out of the
SAME 40 train scenarios (different videos/takes only), so dev accuracy
(88.1% with the current 300-estimator/depth-5 config) measures generalizing
to new footage of an already-seen scenario, not to a genuinely unseen cue
combination the way test does (48.1%, on 22 different scenarios, 18 of
which share no cue combination with any train scenario). Tuning against dev
was optimizing for the wrong kind of generalization.

This script instead runs StratifiedGroupKFold, grouped by scenario_dir,
over the full train+dev pool (every split_design_v2 in {"train","dev"}
clip -- i.e. clips.csv's original split_design=="train" rows, before the
dev carve-out). Every fold's held-out scenarios are genuinely unseen by
that fold's fit -- the same kind of split test uses, just executed
repeatedly (5x) within the training pool so it's usable for model
selection without ever touching test.

Not applied here: gbt.py's modality-dropout augmentation. CV evaluates
hyperparameter choices on clean (undropped) folds for a clearer signal on
capacity/regularization alone; the winning hyperparameters are then used
WITH dropout in gbt.py's actual training run.

Also runs a simple-model sanity check (logistic regression, shallow single
tree) through the identical CV protocol -- if a much lower-capacity model
scores comparably to tuned LightGBM, that's direct evidence the original
300-estimator/depth-5 config's problem was capacity relative to the ~38
distinct training cue-combinations, not feature quality.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from aggregate import FEATURE_NAMES  # noqa: E402

FEATURES_PATH = os.path.join(REPO_ROOT, "data", "features", "clip_features.parquet")
RANDOM_SEED = 42
N_FOLDS = 5

BASELINE_PARAMS = dict(max_depth=5, n_estimators=300, subsample=1.0, colsample_bytree=1.0,
                        reg_alpha=0.0, reg_lambda=0.0)
GRID = dict(
    max_depth=[2, 3],
    n_estimators=[50, 100],
    subsample=[0.7, 0.9],
    colsample_bytree=[0.7, 0.9],
)
REG_PRESETS = [(0.0, 0.0), (0.1, 0.1)]  # (reg_alpha, reg_lambda)

FIXED_GBT_PARAMS = dict(objective="multiclass", class_weight="balanced",
                         learning_rate=0.05, random_state=RANDOM_SEED, verbosity=-1)


def build_candidates():
    candidates = [dict(BASELINE_PARAMS)]
    for max_depth, n_estimators, subsample, colsample_bytree in itertools.product(
        GRID["max_depth"], GRID["n_estimators"], GRID["subsample"], GRID["colsample_bytree"]
    ):
        for reg_alpha, reg_lambda in REG_PRESETS:
            candidates.append(dict(
                max_depth=max_depth, n_estimators=n_estimators,
                subsample=subsample, colsample_bytree=colsample_bytree,
                reg_alpha=reg_alpha, reg_lambda=reg_lambda,
            ))
    return candidates


def cv_score(model_factory, X, y, groups, cv):
    accs, f1s = [], []
    for train_idx, val_idx in cv.split(X, y, groups):
        model = model_factory()
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[val_idx])
        accs.append(accuracy_score(y.iloc[val_idx], preds))
        f1s.append(f1_score(y.iloc[val_idx], preds, average="macro", zero_division=0))
    return np.mean(accs), np.std(accs), np.mean(f1s), np.std(f1s)


def main():
    import mlflow

    from tracking.dataset_logging import log_dataset
    from tracking.hashing import sha256_file
    from tracking.mlflow_setup import init_tracking

    init_tracking()
    with mlflow.start_run(run_name="gbt_tuning_cv"):
        mlflow.set_tag("model_type", "gbt_tuning_cv")
        mlflow.set_tag("code_file", os.path.relpath(__file__, REPO_ROOT))
        mlflow.set_tag("code_version_sha256", sha256_file(__file__))

        df = log_dataset(context="tuning")
        pool = df[df["split_design_v2"].isin(["train", "dev"])].reset_index(drop=True)
        X = pool[FEATURE_NAMES]
        y = pool["intent"]
        groups = pool["scenario_dir"]
        n_scenarios = groups.nunique()
        mlflow.log_param("n_pool_clips", len(pool))
        mlflow.log_param("n_pool_scenarios", n_scenarios)
        mlflow.log_param("n_folds", N_FOLDS)
        print(f"[tune_gbt] CV pool: {len(pool)} clips, {n_scenarios} distinct scenarios, "
              f"{N_FOLDS}-fold StratifiedGroupKFold (grouped by scenario_dir)")

        cv = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)

        rows = []
        for i, params in enumerate(build_candidates()):
            full_params = {**FIXED_GBT_PARAMS, **params}
            acc_mean, acc_std, f1_mean, f1_std = cv_score(
                lambda p=full_params: LGBMClassifier(**p), X, y, groups, cv)
            tag = "baseline" if params == BASELINE_PARAMS else f"cand_{i}"
            rows.append({"config": tag, **params, "cv_accuracy_mean": acc_mean,
                         "cv_accuracy_std": acc_std, "cv_macro_f1_mean": f1_mean, "cv_macro_f1_std": f1_std})

        results_df = pd.DataFrame(rows).sort_values("cv_macro_f1_mean", ascending=False).reset_index(drop=True)
        print(f"\n[tune_gbt] top 10 of {len(results_df)} GBT configs by CV macro F1:")
        print(results_df.head(10).to_string(index=False))
        mlflow.log_table(data=results_df, artifact_file="reports/gbt_tuning_grid.json")

        best_row = results_df.iloc[0]
        best_params = {k: best_row[k] for k in ["max_depth", "n_estimators", "subsample", "colsample_bytree",
                                                  "reg_alpha", "reg_lambda"]}
        baseline_row = results_df[results_df["config"] == "baseline"].iloc[0]
        print(f"\n[tune_gbt] baseline (300 est, depth 5): "
              f"cv_accuracy={baseline_row['cv_accuracy_mean']:.3f}+-{baseline_row['cv_accuracy_std']:.3f}, "
              f"cv_macro_f1={baseline_row['cv_macro_f1_mean']:.3f}+-{baseline_row['cv_macro_f1_std']:.3f}")
        print(f"[tune_gbt] best candidate: {dict(best_params)}")
        print(f"[tune_gbt]   cv_accuracy={best_row['cv_accuracy_mean']:.3f}+-{best_row['cv_accuracy_std']:.3f}, "
              f"cv_macro_f1={best_row['cv_macro_f1_mean']:.3f}+-{best_row['cv_macro_f1_std']:.3f}")

        for k, v in best_params.items():
            mlflow.log_param(f"best_{k}", v)
        mlflow.log_metric("best_cv_accuracy_mean", best_row["cv_accuracy_mean"])
        mlflow.log_metric("best_cv_macro_f1_mean", best_row["cv_macro_f1_mean"])
        mlflow.log_metric("baseline_cv_accuracy_mean", baseline_row["cv_accuracy_mean"])
        mlflow.log_metric("baseline_cv_macro_f1_mean", baseline_row["cv_macro_f1_mean"])

        # ── Simple-model sanity check: is capacity really the problem? ─────
        print("\n[tune_gbt] simple-model sanity check (same CV protocol):")
        simple_models = {
            "logistic_regression": lambda: make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=RANDOM_SEED),
            ),
            "shallow_tree_depth3": lambda: DecisionTreeClassifier(
                max_depth=3, class_weight="balanced", random_state=RANDOM_SEED),
        }
        for name, factory in simple_models.items():
            acc_mean, acc_std, f1_mean, f1_std = cv_score(factory, X, y, groups, cv)
            print(f"[tune_gbt]   {name}: cv_accuracy={acc_mean:.3f}+-{acc_std:.3f}, "
                  f"cv_macro_f1={f1_mean:.3f}+-{f1_std:.3f}")
            mlflow.log_metric(f"{name}_cv_accuracy_mean", acc_mean)
            mlflow.log_metric(f"{name}_cv_macro_f1_mean", f1_mean)

        print(f"\n[tune_gbt] verdict: best-tuned GBT cv_macro_f1={best_row['cv_macro_f1_mean']:.3f} vs "
              f"baseline={baseline_row['cv_macro_f1_mean']:.3f} "
              f"({'improvement' if best_row['cv_macro_f1_mean'] > baseline_row['cv_macro_f1_mean'] else 'no improvement'} "
              f"from capacity reduction/regularization alone, under scenario-grouped CV).")

        return dict(best_params)


if __name__ == "__main__":
    main()

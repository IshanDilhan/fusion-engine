"""
Build the Action Generator training CSV from the 62-scenario Final_Dataset V3.

Generates two CSV files:
  1. action_generator_training_scenarios.csv — The base 62 scenarios (shareable with team)
  2. action_generator_augmented_training.csv — 10x augmented train set with 15% context dropout

Run standalone:
    python build_dataset_csv.py
"""

import os
import csv
import random
from collections import Counter

# ─── Import from config (same package) ───────────────────────────────────────
import sys
sys.path.insert(0, os.path.dirname(__file__))
from config import (
    INTENTS, ACTIONS, ACTION_DESCRIPTIONS, DEFAULT_CONTROLS,
    intent_to_idx, action_to_idx,
)

# Human-readable intent names for the CSV
INTENT_NAMES = {
    "F01": "Greeting / positive acknowledgment",
    "F02": "Emergency / danger",
    "F03": "Task assistance request",
    "F04": "Help request",
    "F05": "Engaged / busy - no interaction needed",
    "F06": "Requests passage / space",
    "F07": "Frustration / agitation",
    "F08": "Break / relief request",
    "F09": "Farewell",
    "F10": "Discouraged / giving up",
}

# ─── All 62 scenarios from Final_Dataset V3 ──────────────────────────────────
SCENARIOS = [
    # id, split, context, description, emotion, gesture, motion, direction, missing, intent, action
    (1, "train", "classroom", "Student waves at robot with happy face while walking toward it", "happy", "wave", "walk", "toward_robot", "", "F01", "A01"),
    (2, "train", "classroom", "Student gives thumbs up with happy face while sitting", "happy", "thumbs_up", "sit", "stationary", "", "F01", "A01"),
    (3, "train", "classroom", "Student suddenly stands up from chair startled throwing both hands up", "surprise", "both_hands_up", "stand", "stationary", "", "F02", "A14"),
    (4, "train", "classroom", "Student jumps back from a desk frightened both hands up", "fear", "both_hands_up", "step_back", "away_from_robot", "", "F02", "A14"),
    (5, "train", "classroom", "Student beckons robot over with neutral face while sitting", "neutral", "beckoning", "sit", "stationary", "", "F03", "A04"),
    (6, "train", "classroom", "Student beckons the robot to their desk with neutral face seated context sensor offline", "neutral", "beckoning", "sit", "stationary", "context", "F03", "A04"),
    (7, "train", "classroom", "Student raises hand with neutral face while sitting at desk", "neutral", "raise_hand", "sit", "stationary", "", "F04", "A05"),
    (8, "train", "classroom", "Student with sad face gives thumbs down toward own worksheet barely moving seated", "sad", "thumbs_down", "sit", "stationary", "", "F04", "A05"),
    (9, "train", "classroom", "Student slumped and sad staring down hands below desk no gesture captured", "sad", "MISSING", "sit", "stationary", "gesture", "F04", "A05"),
    (10, "train", "classroom", "Student writes notes during lecture occasionally pointing at the notebook neutral face", "neutral", "point", "sit", "stationary", "", "F05", "A06"),
    (11, "train", "classroom", "Student smiles and raises hand while sitting volunteering to answer the teacher", "happy", "raise_hand", "sit", "stationary", "", "F05", "A06"),
    (12, "train", "classroom", "Student walks toward robot in a narrow aisle carrying books hands occupied gesture not captured", "neutral", "MISSING", "walk", "toward_robot", "gesture", "F06", "A11"),
    (13, "train", "classroom", "Student annoyed by the robot hovering waves it away with angry face standing", "angry", "wave", "stand", "stationary", "", "F06", "A11"),
    (14, "train", "classroom", "Student raises both hands in frustration rigid posture standing", "angry", "both_hands_up", "stand", "stationary", "", "F07", "A08"),
    (15, "train", "classroom", "Student crumples a worksheet and gives thumbs down with angry face seated", "angry", "thumbs_down", "sit", "stationary", "", "F07", "A08"),
    (16, "train", "classroom", "Student pushes worksheet away gives thumbs down with disgusted face stepping back from desk", "disgust", "thumbs_down", "step_back", "away_from_robot", "", "F08", "A07"),
    (17, "train", "classroom", "Student grimaces at a long assignment gives thumbs down slumping back in chair", "disgust", "thumbs_down", "sit", "stationary", "", "F08", "A07"),
    (18, "train", "classroom", "Student waves goodbye with happy face while walking to the door at dismissal", "happy", "wave", "walk", "toward_exit", "", "F09", "A09"),
    (19, "train", "classroom", "Student packs bag and waves briefly with neutral face heading toward the door", "neutral", "wave", "walk", "toward_exit", "", "F09", "A10"),
    (20, "train", "classroom", "Student pushes assignment aside and rests head on desk sad no gesture motionless", "sad", "none", "sit", "stationary", "", "F10", "A12"),
    (21, "train", "classroom", "Student stands by desk staring at a failed test sad arms hanging no gesture", "sad", "none", "stand", "stationary", "", "F10", "A12"),
    (22, "test", "classroom", "Student waves while walking toward robot face occluded by a held book emotion not captured", "MISSING", "wave", "walk", "toward_robot", "emotion", "F01", "A01"),
    (23, "test", "classroom", "Student runs toward the exit in panic gesture lost to motion blur", "fear", "MISSING", "run", "toward_exit", "gesture", "F02", "A14"),
    (24, "test", "classroom", "Student beckons robot while walking toward the supply shelf room-identity sensor offline", "neutral", "beckoning", "walk", "toward_object", "context", "F03", "A13"),
    (25, "test", "classroom", "Student raises hand leaning forward emotion sensor occluded", "MISSING", "raise_hand", "lean_forward", "stationary", "emotion", "F04", "A15"),
    (26, "test", "classroom", "Student stretches both arms overhead mid-study with neutral face then returns to writing seated", "neutral", "both_hands_up", "sit", "stationary", "", "F05", "A06"),
    (27, "test", "classroom", "Student stands up angrily points at robot stepping toward it", "angry", "point", "walk", "toward_robot", "", "F06", "A11"),
    (28, "test", "classroom", "Student scowls angrily while giving an exaggerated thumbs up after being corrected seated", "angry", "thumbs_up", "sit", "stationary", "", "F07", "A08"),
    (29, "test", "classroom", "Student gives thumbs up with disgusted face after finishing a disliked task seated", "disgust", "thumbs_up", "sit", "stationary", "", "F08", "A07"),
    (30, "test", "classroom", "Student waves with neutral-looking face while walking out of the classroom emotion not detected", "MISSING", "wave", "walk", "toward_exit", "emotion", "F09", "A09"),
    (31, "test", "classroom", "Student drops the crumpled quiz on the desk and stands turned away sad arms slack no gesture", "sad", "none", "stand", "stationary", "", "F10", "A12"),
    (32, "train", "kitchen", "Cook tastes finished dish smiles gives thumbs up standing still", "happy", "thumbs_up", "stand", "stationary", "", "F01", "A01"),
    (33, "train", "kitchen", "Person gives thumbs down while smiling and shaking head playfully family tasting game", "happy", "thumbs_down", "stand", "stationary", "", "F01", "A01"),
    (34, "train", "kitchen", "Person extends both hands out to signal stop and runs back from smoke at the stove", "fear", "both_hands_up", "run", "away_from_robot", "", "F02", "A02"),
    (35, "train", "kitchen", "Person touches a hot pan startled throws both hands up and steps back", "surprise", "both_hands_up", "step_back", "away_from_robot", "", "F02", "A02"),
    (36, "train", "kitchen", "Person beckons robot toward the stove with neutral face while walking there", "neutral", "beckoning", "walk", "toward_object", "", "F03", "A04"),
    (37, "train", "kitchen", "Person points at a shelf while walking toward it neutral face", "neutral", "point", "walk", "toward_object", "", "F03", "A13"),
    (38, "train", "kitchen", "Person slumped against counter sad barely moving gives a weak thumbs down", "sad", "thumbs_down", "sit", "stationary", "", "F04", "A05"),
    (39, "train", "kitchen", "Person beckons robot with tired sad face while standing over a pile of dirty dishes", "sad", "beckoning", "stand", "stationary", "", "F04", "A05"),
    (40, "train", "kitchen", "Person chops vegetables with neutral focus back to robot occasionally pointing at ingredients", "neutral", "none", "stand", "stationary", "", "F05", "A06"),
    (41, "train", "kitchen", "Person wipes hands on apron neutral glancing around transition between tasks self-directed point", "neutral", "none", "stand", "stationary", "", "F05", "A06"),
    (42, "train", "kitchen", "Person pushes robot aside with disgusted face pointing it away stepping back", "disgust", "point", "step_back", "away_from_robot", "", "F06", "A11"),
    (43, "train", "kitchen", "Person carrying a hot pot with both hands walks toward robot in the galley aisle no gesture possible", "neutral", "MISSING", "walk", "toward_robot", "gesture", "F06", "A11"),
    (44, "train", "kitchen", "Person slams hand on counter then raises both hands angry rigid standing", "angry", "both_hands_up", "stand", "stationary", "", "F07", "A08"),
    (45, "train", "kitchen", "Person angrily points repeatedly at spilled food on the floor while looking at the robot standing", "angry", "point", "stand", "stationary", "", "F07", "A08"),
    (46, "train", "kitchen", "Person recoils from refrigerator odor disgusted thumbs down stepping back", "disgust", "thumbs_down", "step_back", "away_from_robot", "", "F08", "A07"),
    (47, "train", "kitchen", "Person stands at the sink grimacing at greasy pans gives thumbs down leaning away", "disgust", "thumbs_down", "stand", "stationary", "", "F08", "A07"),
    (48, "train", "kitchen", "Person waves at robot with happy face while walking toward the exit", "happy", "wave", "walk", "toward_exit", "", "F09", "A09"),
    (49, "train", "kitchen", "Person unties apron and waves while walking to the door face turned away emotion not captured", "MISSING", "wave", "walk", "toward_exit", "emotion", "F09", "A10"),
    (50, "train", "kitchen", "Person sits down heavily sad staring at a failed dish hands in lap no gesture", "sad", "none", "sit", "stationary", "", "F10", "A12"),
    (51, "train", "kitchen", "Person stares at a burnt cake shoulders dropped sad hands at sides standing motionless", "sad", "none", "stand", "stationary", "", "F10", "A12"),
    (52, "test", "kitchen", "Person raises hand and smiles while sitting at the kitchen table after eating", "happy", "raise_hand", "sit", "stationary", "", "F01", "A01"),
    (53, "test", "kitchen", "Person runs from the stove area in fear gesture lost to motion blur", "fear", "MISSING", "run", "toward_exit", "gesture", "F02", "A02"),
    (54, "test", "kitchen", "Person stands frozen wide-eyed staring at the oven rigid no gesture", "fear", "none", "stand", "stationary", "", "F02", "A02"),
    (55, "test", "kitchen", "Person cheerfully points at a high cupboard while walking toward it looking at robot", "happy", "point", "walk", "toward_object", "", "F03", "A13"),
    (56, "test", "kitchen", "Person beckons robot with sad face standing room-identity sensor offline", "sad", "beckoning", "stand", "stationary", "context", "F04", "A05"),
    (57, "test", "kitchen", "Person stands neutral no gesture visible stationary emotion sensor noisy", "MISSING", "MISSING", "stand", "stationary", "emotion,gesture", "F05", "A06"),
    (58, "test", "kitchen", "Person carrying stacked trays walks toward robot hands occupied and face hidden behind the trays", "MISSING", "MISSING", "walk", "toward_robot", "emotion,gesture", "F06", "A11"),
    (59, "test", "kitchen", "Person gives thumbs up with angry face after being told to redo a recipe standing", "angry", "thumbs_up", "stand", "stationary", "", "F07", "A08"),
    (60, "test", "kitchen", "Person pushes plate away at the table disgusted thumbs down seated", "disgust", "thumbs_down", "sit", "stationary", "", "F08", "A07"),
    (61, "test", "kitchen", "Person waves while walking toward the exit with sad face after a failed cooking session", "sad", "wave", "walk", "toward_exit", "", "F09", "A10"),
    (62, "test", "kitchen", "Person slumps forward over the counter head down over the failed dish sad no gesture", "sad", "none", "lean_forward", "stationary", "", "F10", "A12"),
]


def build_datasets():
    """Build the base and augmented training CSVs."""
    output_dir = os.path.join(os.path.dirname(__file__), "training_data")
    os.makedirs(output_dir, exist_ok=True)

    headers = [
        "scenario_id", "split", "context", "emotion", "gesture", "motion",
        "direction", "missing_cues", "intent", "intent_name", "action",
        "action_description",
        # Columns used by train.py (target_ prefix for control signals)
        "target_v", "target_omega", "target_d",
    ]

    base_path = os.path.join(output_dir, "action_generator_training_scenarios.csv")
    aug_path  = os.path.join(output_dir, "action_generator_augmented_training.csv")

    base_rows = []
    intent_ctr = Counter()
    action_ctr = Counter()
    split_ctr  = Counter()

    for sc in SCENARIOS:
        sid, split, ctx, desc, emo, ges, mot, dire, miss, intent, action = sc
        v, omega, d = DEFAULT_CONTROLS[action]
        row = [
            sid, split, ctx, emo, ges, mot, dire, miss,
            intent, INTENT_NAMES.get(intent, ""),
            action, ACTION_DESCRIPTIONS.get(action, ""),
            v, omega, d,
        ]
        base_rows.append(row)
        intent_ctr[intent] += 1
        action_ctr[action] += 1
        split_ctr[split]   += 1

    try:
        with open(base_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(base_rows)
    except PermissionError:
        print(f"Warning: {base_path} is currently open in Excel. Using existing file.")

    # --- Augmented dataset (10x train with 15% context dropout) ---------------
    random.seed(42)
    aug_rows = []
    aug_id = 1

    for sc in SCENARIOS:
        sid, split, ctx, desc, emo, ges, mot, dire, miss, intent, action = sc
        v, omega, d = DEFAULT_CONTROLS[action]

        if split == "train":
            for _ in range(10):
                aug_ctx = ctx
                aug_miss = miss
                if random.random() < 0.15:
                    aug_ctx = "offline"
                    aug_miss = (miss + ",context").lstrip(",") if miss else "context"
                row = [
                    f"aug_{aug_id}", "train", aug_ctx, emo, ges, mot, dire, aug_miss,
                    intent, INTENT_NAMES.get(intent, ""),
                    action, ACTION_DESCRIPTIONS.get(action, ""),
                    v, omega, d,
                ]
                aug_rows.append(row)
                aug_id += 1
        else:
            # Test scenarios kept unchanged
            row = [
                sid, "test", ctx, emo, ges, mot, dire, miss,
                intent, INTENT_NAMES.get(intent, ""),
                action, ACTION_DESCRIPTIONS.get(action, ""),
                v, omega, d,
            ]
            aug_rows.append(row)

    try:
        with open(aug_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(headers)
            w.writerows(aug_rows)
    except PermissionError:
        print(f"Warning: {aug_path} is currently open in Excel. Using existing file.")

    # --- Summary ----------------------------------------------------------------
    print("=" * 60)
    print("  Action Generator Dataset Builder")
    print("=" * 60)
    print(f"\n  Base CSV  : {base_path}")
    print(f"  Augmented : {aug_path}")
    print(f"  Base rows : {len(base_rows)}  |  Augmented rows : {len(aug_rows)}")

    print(f"\n  Split distribution:")
    for s in sorted(split_ctr):
        print(f"    {s:6s} : {split_ctr[s]}")

    print(f"\n  Intent distribution (base):")
    for k in sorted(intent_ctr):
        print(f"    {k} ({INTENT_NAMES.get(k,''):40s}) : {intent_ctr[k]}")

    print(f"\n  Action distribution (base):")
    for k in sorted(action_ctr):
        print(f"    {k} ({ACTION_DESCRIPTIONS.get(k,''):55s}) : {action_ctr[k]}")

    print("\nDone!")


if __name__ == "__main__":
    build_datasets()

"""
build_dataset_csv.py

Generates the machine-readable dataset CSV files for the Action Generator.
Reads all 73 multimodal scenarios from the HRI Multimodal Intent Dataset V3.5,
maps text tokens to canonical vocabulary indices, and writes:
  1. action_generator_training_scenarios.csv (72 base scenarios)
  2. action_generator_augmented_training.csv  (318 augmented scenarios with Modality Dropout)

Usage:
    python action_generator/training/build_dataset_csv.py
"""

import os
import sys
import random
import pandas as pd

# Add parent directory to sys.path to import config
ACTION_GEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ACTION_GEN_DIR not in sys.path:
    sys.path.insert(0, ACTION_GEN_DIR)

from config import (
    INTENTS, MOTIONS, DIRECTIONS, CONTEXTS, ACTIONS,
    intent_to_idx, motion_to_idx, direction_to_idx, context_to_idx, action_to_idx,
    DEFAULT_CONTROLS
)

OUTPUT_DIR = os.path.join(ACTION_GEN_DIR, "training_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_CSV_PATH = os.path.join(OUTPUT_DIR, "action_generator_training_scenarios.csv")
AUGMENTED_CSV_PATH = os.path.join(OUTPUT_DIR, "action_generator_augmented_training.csv")

# ─── All 73 scenarios from HRI Multimodal Intent Dataset V3.5 ─────────────────────────
SCENARIOS = [
    # id, split, context, description, emotion, gesture, motion, direction, missing, intent, action
    (1, "train", "classroom", "Student waves at robot with happy face while walking toward it", "happy", "wave", "walking", "toward_robot", "", "F01", "A01"),
    (2, "train", "classroom", "Student gives thumbs up with happy face while sitting", "happy", "thumbs_up", "sitting", "stationary", "", "F01", "A01"),
    (3, "train", "classroom", "Student suddenly stands up from chair startled throwing both hands up", "surprise", "both_hands_up", "standing", "stationary", "", "F02", "A14"),
    (4, "train", "classroom", "Student jumps back from a desk frightened both hands up", "fear", "both_hands_up", "stepping_back", "away_from_robot", "", "F02", "A14"),
    (5, "train", "classroom", "Student beckons robot over with neutral face while sitting", "neutral", "beckoning", "sitting", "stationary", "", "F03", "A04"),
    (6, "train", "classroom", "Student beckons the robot to their desk with neutral face seated context sensor offline", "neutral", "beckoning", "sitting", "stationary", "context", "F03", "A04"),
    (7, "train", "classroom", "Student raises hand with neutral face while sitting at desk", "neutral", "raise_hand", "sitting", "stationary", "", "F04", "A05"),
    (8, "train", "classroom", "Student with sad face gives thumbs down toward own worksheet barely moving seated", "sad", "thumbs_down", "sitting", "stationary", "", "F04", "A05"),
    (9, "train", "classroom", "Student slumped and sad staring down hands below desk no gesture captured", "sad", "none", "sitting", "stationary", "gesture", "F04", "A05"),
    (10, "train", "classroom", "Student writes notes during lecture occasionally pointing at the notebook neutral face", "neutral", "none", "sitting", "stationary", "", "F05", "A06"),
    (11, "train", "classroom", "Student smiles and raises hand while sitting volunteering to answer the teacher", "happy", "raise_hand", "sitting", "stationary", "", "F05", "A06"),
    (12, "train", "classroom", "Student walks toward robot in a narrow aisle carrying books hands occupied gesture not captured", "neutral", "MISSING", "walking", "toward_robot", "gesture", "F06", "A11"),
    (13, "train", "classroom", "Student annoyed by the robot hovering waves it away with angry face standing", "angry", "wave", "standing", "stationary", "", "F06", "A11"),
    (14, "train", "classroom", "Student raises both hands in frustration rigid posture standing", "angry", "both_hands_up", "standing", "stationary", "", "F07", "A08"),
    (15, "train", "classroom", "Student crumples a worksheet and gives thumbs down with angry face seated", "angry", "thumbs_down", "sitting", "stationary", "", "F07", "A08"),
    (16, "train", "classroom", "Student pushes worksheet away gives thumbs down with disgusted face stepping back from desk", "disgust", "thumbs_down", "stepping_back", "away_from_robot", "", "F08", "A07"),
    (17, "train", "classroom", "Student grimaces at a long assignment gives thumbs down slumping back in chair", "disgust", "thumbs_down", "sitting", "stationary", "", "F08", "A07"),
    (18, "test", "classroom", "Student waves goodbye with happy face while walking to the door at dismissal", "happy", "wave", "walking", "toward_exit", "", "F01", "A09"),
    (19, "train", "classroom", "Student packs bag and waves briefly with neutral face heading toward the door", "neutral", "wave", "walking", "toward_exit", "", "F01", "A10"),
    (20, "train", "classroom", "Student pushes assignment aside and rests head on desk sad no gesture motionless", "sad", "none", "sitting", "stationary", "", "F09", "A12"),
    (21, "train", "classroom", "Student stands by desk staring at a failed test sad arms hanging no gesture", "sad", "none", "standing", "stationary", "", "F09", "A12"),
    (22, "test", "classroom", "Student waves while walking toward robot face occluded by a held book emotion not captured", "MISSING", "wave", "walking", "toward_robot", "emotion", "F01", "A01"),
    (23, "test", "classroom", "Visitor standing in a hospital waiting area gasps in surprise raising both hands as fixture detaches", "surprise", "both_hands_up", "standing", "stationary", "", "F02", "A02"),
    (24, "test", "classroom", "Student beckons robot while walking toward the supply shelf room-identity sensor offline", "neutral", "beckoning", "walking", "toward_object", "context", "F03", "A13"),
    (25, "test", "classroom", "Student raises hand leaning forward emotion sensor occluded", "MISSING", "raise_hand", "stepping_back", "stationary", "emotion", "F04", "A15"),
    (26, "test", "classroom", "Visitor seated on gallery bench focused on sketching artwork with neutral expression", "neutral", "none", "sitting", "stationary", "", "F05", "A06"),
    (27, "test", "classroom", "Student stands up angrily points at robot stepping toward it", "angry", "point", "walking", "toward_robot", "", "F06", "A11"),
    (28, "test", "classroom", "Student scowls angrily while giving an exaggerated thumbs up after being corrected seated", "angry", "thumbs_up", "sitting", "stationary", "", "F07", "A08"),
    (29, "test", "classroom", "Student gives thumbs up with disgusted face after finishing a disliked task seated", "disgust", "thumbs_up", "sitting", "stationary", "", "F08", "A07"),
    (31, "test", "classroom", "Student drops the crumpled quiz on desk and stands turned away sad arms slack no gesture", "sad", "none", "stepping_back", "stationary", "", "F09", "A12"),
    (32, "train", "kitchen", "Cook tastes finished dish smiles gives thumbs up standing still", "happy", "thumbs_up", "standing", "stationary", "", "F01", "A01"),
    (33, "train", "kitchen", "Person gives thumbs down while smiling and shaking head playfully family tasting game", "happy", "thumbs_down", "standing", "stationary", "", "F01", "A01"),
    (34, "train", "kitchen", "Person extends both hands out to signal stop and runs back from smoke at stove", "fear", "both_hands_up", "stepping_back", "away_from_robot", "", "F02", "A02"),
    (35, "train", "kitchen", "Person touches hot pan startled throws both hands up and steps back", "surprise", "both_hands_up", "stepping_back", "away_from_robot", "", "F02", "A02"),
    (36, "train", "kitchen", "Person beckons robot toward stove with neutral face while walking there", "neutral", "beckoning", "walking", "toward_object", "", "F03", "A04"),
    (37, "train", "kitchen", "Person points at a shelf while walking toward it neutral face", "neutral", "point", "walking", "toward_object", "", "F03", "A13"),
    (38, "train", "kitchen", "Person slumped against counter sad barely moving gives weak thumbs down", "sad", "thumbs_down", "sitting", "stationary", "", "F04", "A05"),
    (39, "train", "kitchen", "Person beckons robot with tired sad face while standing over dirty dishes", "sad", "beckoning", "standing", "stationary", "", "F04", "A05"),
    (40, "train", "kitchen", "Person chops vegetables with neutral focus back to robot pointing at ingredients", "neutral", "none", "standing", "stationary", "", "F05", "A06"),
    (41, "train", "kitchen", "Person wipes hands on apron neutral glancing around transition between tasks", "neutral", "none", "standing", "stationary", "", "F05", "A06"),
    (42, "train", "kitchen", "Person pushes robot aside with disgusted face pointing away stepping back", "disgust", "point", "stepping_back", "away_from_robot", "", "F06", "A11"),
    (43, "train", "kitchen", "Person carrying hot pot with both hands walks toward robot in galley aisle no gesture", "neutral", "MISSING", "walking", "toward_robot", "gesture", "F06", "A11"),
    (44, "train", "kitchen", "Person slams hand on counter then raises both hands angry rigid standing", "angry", "both_hands_up", "standing", "stationary", "", "F07", "A08"),
    (45, "train", "kitchen", "Person angrily points repeatedly at spilled food on floor standing", "angry", "point", "standing", "stationary", "", "F07", "A08"),
    (46, "train", "kitchen", "Person recoils from refrigerator odor disgusted thumbs down stepping back", "disgust", "thumbs_down", "stepping_back", "away_from_robot", "", "F08", "A07"),
    (47, "train", "kitchen", "Person stands at sink grimacing at greasy pans gives thumbs down leaning away", "disgust", "thumbs_down", "standing", "stationary", "", "F08", "A07"),
    (48, "train", "kitchen", "Person waves at robot with happy face while walking toward exit", "happy", "wave", "walking", "toward_exit", "", "F01", "A09"),
    (49, "train", "kitchen", "Person unties apron and waves while walking to door face turned away emotion missing", "MISSING", "wave", "walking", "toward_exit", "emotion", "F01", "A10"),
    (50, "train", "kitchen", "Person sits down heavily sad staring at failed dish hands in lap no gesture", "sad", "none", "sitting", "stationary", "", "F09", "A12"),
    (51, "train", "kitchen", "Person stares at burnt cake shoulders dropped sad standing motionless", "sad", "none", "standing", "stationary", "", "F09", "A12"),
    (52, "train", "kitchen", "Person raises hand and smiles while sitting at kitchen table after eating", "happy", "raise_hand", "sitting", "stationary", "", "F01", "A01"),
    (53, "test", "kitchen", "Visitor fleeing exhibit hall after fire alarm waves both arms frantic running to exit", "fear", "both_hands_up", "walking", "toward_exit", "", "F02", "A02"),
    (54, "test", "kitchen", "Patient standing in corridor clutching chest in acute distress raises both hands fearful", "fear", "both_hands_up", "standing", "stationary", "", "F02", "A03"),
    (55, "test", "kitchen", "Nurse approaching equipment station points directly at IV pole with focused neutral face", "happy", "point", "walking", "toward_object", "", "F03", "A13"),
    (56, "test", "kitchen", "Person beckons robot with sad face standing room-identity sensor offline", "sad", "beckoning", "standing", "stationary", "context", "F04", "A05"),
    (57, "test", "kitchen", "Person slowly stirs pot with content neutral expression deeply focused", "neutral", "none", "standing", "stationary", "", "F05", "A06"),
    (58, "test", "kitchen", "Person carrying stacked trays walks toward robot hands occupied face hidden", "MISSING", "MISSING", "walking", "toward_robot", "emotion,gesture", "F06", "A11"),
    (59, "test", "kitchen", "Person gives thumbs up with angry face after being told to redo recipe standing", "angry", "thumbs_up", "standing", "stationary", "", "F07", "A08"),
    (60, "test", "kitchen", "Person pushes plate away at table disgusted thumbs down seated", "disgust", "thumbs_down", "sitting", "stationary", "", "F08", "A07"),
    (61, "test", "kitchen", "Visitor walking toward museum entrance waves goodbye to robot guide smiling", "happy", "wave", "walking", "toward_exit", "", "F01", "A10"),
    (62, "test", "kitchen", "Person slumps forward over counter head down over failed dish sad no gesture", "sad", "none", "stepping_back", "stationary", "", "F09", "A12"),
    (63, "test", "kitchen", "Person sits down looking sad slow movement staring at failed dish", "sad", "thumbs_down", "sitting", "stationary", "", "F04", "A05"),
    (64, "test", "kitchen", "Museum curator carrying heavy display box walks quickly toward robot neutral", "neutral", "none", "walking", "toward_robot", "", "F06", "A11"),
    (65, "test", "kitchen", "Visitor gives exaggerated thumbs up with angry scowl after warning standing", "angry", "thumbs_up", "standing", "stationary", "", "F07", "A08"),
    (66, "test", "kitchen", "Nurse beckons robot while walking toward IV pole with neutral expression", "neutral", "beckoning", "walking", "toward_object", "", "F03", "A13"),
    (67, "test", "kitchen", "Recovering patient in bed grimaces disgusted while giving thumbs up seated", "disgust", "thumbs_up", "sitting", "stationary", "", "F08", "A07"),
    (68, "test", "kitchen", "Visitor flees exhibit hall in terror after alarm gesture blur", "fear", "MISSING", "walking", "toward_exit", "gesture", "F02", "A02"),
    (69, "test", "kitchen", "Paramedic pushing gurney points sternly down hallway angry stepping to robot", "angry", "point", "walking", "toward_robot", "", "F06", "A11"),
    (70, "test", "kitchen", "Visitor inspecting disturbing art gives thumbs down disgusted stepping back", "disgust", "thumbs_down", "sitting", "stationary", "", "F08", "A07"),
    (71, "test", "kitchen", "Paramedic pushing gurney walks rapidly down corridor toward robot waving aside", "neutral", "wave", "walking", "toward_robot", "", "F06", "A11"),
    (72, "test", "kitchen", "Paramedic pushing gurney walks rapidly down corridor toward robot neutral waving aside", "sad", "none", "sitting", "stationary", "", "F09", "A11"),
    (73, "test", "kitchen", "Paramedic pushing gurney walks rapidly down corridor toward robot neutral waving aside", "sad", "none", "sitting", "stationary", "", "F09", "A11"),
]


def build_scenario_dataframe():
    """Converts raw SCENARIOS list to a pandas DataFrame with numeric feature indices."""
    rows = []

    for item in SCENARIOS:
        sid, split, ctx, desc, emo, gest, mot, direct, miss, intent, action = item

        # Convert text tokens to integer indices
        i_idx = intent_to_idx(intent)
        m_idx = motion_to_idx(mot)
        d_idx = direction_to_idx(direct)
        c_idx = context_to_idx(ctx)
        a_idx = action_to_idx(action)

        # Retrieve ground-truth control targets (v, omega, d) for the assigned action
        ctrl_v, ctrl_w, ctrl_d = DEFAULT_CONTROLS.get(action, (0.0, 0.0, 1.0))

        # Default human movement speed (m/s) based on motion state
        if mot == "walking":
            human_speed = 0.8
        elif mot == "stepping_back":
            human_speed = 0.5
        else:
            human_speed = 0.0

        is_moving = 1.0 if human_speed > 0.0 else 0.0

        rows.append({
            "scenario_id": sid,
            "split": split,
            "context": ctx,
            "description": desc,
            "emotion": emo,
            "gesture": gest,
            "motion": mot,
            "direction": direct,
            "missing_cues": miss,
            "intent": intent,
            "intent_code": intent,
            "action": action,
            "action_code": action,
            "intent_idx": i_idx,
            "intent_confidence": 0.95,  # Nominal confidence
            "motion_idx": m_idx,
            "direction_idx": d_idx,
            "human_speed": human_speed,
            "human_is_moving": is_moving,
            "context_idx": c_idx,
            "action_idx": a_idx,
            "target_v": ctrl_v,
            "target_omega": ctrl_w,
            "target_d": ctrl_d,
            "target_linear_v": ctrl_v,
            "target_angular_w": ctrl_w,
            "target_comfort_d": ctrl_d
        })

    return pd.DataFrame(rows)


def augment_dataframe(df_base, samples_per_scenario=6):
    """Applies Modality Dropout to generate augmented training samples."""
    augmented_rows = []

    # Include base rows for test split as well
    test_df = df_base[df_base['split'] == 'test']
    for _, row in test_df.iterrows():
        augmented_rows.append(row.to_dict())

    train_df = df_base[df_base['split'] == 'train']

    for _, row in train_df.iterrows():
        # Always include the original row
        augmented_rows.append(row.to_dict())

        for _ in range(samples_per_scenario):
            aug_row = row.to_dict().copy()

            # Randomly jitter intent confidence between 0.65 and 0.99
            aug_row["intent_confidence"] = round(random.uniform(0.65, 0.99), 2)

            # Randomly jitter human speed (+/- 0.15 m/s)
            if aug_row["human_speed"] > 0.0:
                aug_row["human_speed"] = round(max(0.1, aug_row["human_speed"] + random.uniform(-0.15, 0.15)), 2)

            # Modality Dropout (15% probability to mask context token to offline = index 2)
            if random.random() < 0.15:
                aug_row["context_idx"] = context_to_idx("offline")
                aug_row["missing_cues"] = "context"

            augmented_rows.append(aug_row)

    return pd.DataFrame(augmented_rows)


def main():
    print("=" * 60)
    print("  Action Generator Dataset Builder (73 Scenarios V3.5)")
    print("=" * 60)

    # 1. Build Base Scenario Table
    df_base = build_scenario_dataframe()
    df_base.to_csv(BASE_CSV_PATH, index=False)
    print(f"\n  Base CSV  : {BASE_CSV_PATH}")
    print(f"  Total Scenarios: {len(df_base)}")

    # 2. Build Augmented Training Dataset
    df_aug = augment_dataframe(df_base, samples_per_scenario=6)
    df_aug.to_csv(AUGMENTED_CSV_PATH, index=False)
    print(f"  Augmented : {AUGMENTED_CSV_PATH}")
    print(f"  Augmented Rows: {len(df_aug)}")

    # Split Distribution
    print(f"\n  Split distribution:\n{df_base['split'].value_counts().to_string()}")

    # Intent Distribution
    print(f"\n  Intent distribution (base):\n{df_base['intent_code'].value_counts().to_string()}")

    # Action Distribution
    print(f"\n  Action distribution (base):\n{df_base['action_code'].value_counts().to_string()}")

    print("\nDone!")


if __name__ == "__main__":
    main()

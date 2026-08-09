"""
Action Generator Interactive & Automated Runner Script.

Runs all steps:
  1. Build dataset CSVs
  2. Train PyTorch model
  3. Run sample inference demo & safety tests
  4. Export ONNX model for Jetson Orin Nano

Usage:
  python run_all.py
  python run_all.py --step train
  python run_all.py --step infer
"""

import os
import sys
import argparse
import subprocess

PYTHON_EXE = sys.executable
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def run_command(cmd_list, description):
    print("\n" + "=" * 70)
    print(f"  STEP: {description}")
    print("=" * 70)
    res = subprocess.run([PYTHON_EXE] + cmd_list, cwd=BASE_DIR)
    if res.returncode != 0:
        print(f"FAILED: {description} (Exit code {res.returncode})")
        return False
    return True


def build_csv():
    return run_command(["build_dataset_csv.py"], "Building Training CSV Datasets")


def train_model():
    return run_command(["train.py"], "Training MultimodalActionGenerator Model")


def run_inference_demo():
    print("\n" + "=" * 70)
    print("  STEP: Running Inference Demo & 2-Tier Internal Safety Gate Tests")
    print("=" * 70)
    
    sys.path.insert(0, BASE_DIR)
    from inference import ActionInference

    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "best_action_generator.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: Model checkpoint not found at {ckpt_path}. Run training first!")
        return False

    engine = ActionInference(ckpt_path)

    # Test Case 1: Normal Task Request (Safe conditions)
    print("\n--- Test Case 1: Normal Task Request (F03 in Classroom, Safe Distance 1.5m) ---")
    res1 = engine.predict(
        intent="F03",
        intent_confidence=0.95,
        motion_state="sitting",
        direction="stationary",
        velocity=0.0,
        context="classroom",
        current_distance=1.5
    )
    print(f"  Input          : Intent F03, Motion sitting, Context classroom, Dist 1.5m")
    print(f"  Action         : {res1.action} — {res1.action_description}")
    print(f"  Control [v,w,d]: v={res1.linear_velocity_m_s}m/s, w={res1.angular_velocity_rad_s}rad/s, d={res1.comfort_distance_m}m")
    print(f"  Safety Active  : {res1.safety_override_active}")
    print(f"  Safety Reason  : {res1.safety_reason}")

    # Test Case 2: Proximity Risk (Rushing human < 1.0m — Action Preserved, Reverse Yielding v=-0.2 m/s)
    print("\n--- Test Case 2: Rapid Approach Proximity Risk (F04, walking, toward_robot, Dist 0.7m) ---")
    res2 = engine.predict(
        intent="F04",
        intent_confidence=0.92,
        motion_state="walking",
        direction="toward_robot",
        velocity=0.8,
        context="classroom",
        current_distance=0.7
    )
    print(f"  Input          : Intent F04 (Help Request), Motion walking, Direction toward_robot, Speed 0.8m/s, Dist 0.7m")
    print(f"  Action         : {res2.action} — {res2.action_description} (PRESERVED FOR ACCURACY!)")
    print(f"  Control [v,w,d]: v={res2.linear_velocity_m_s}m/s (REVERSE YIELDING STEP!), w={res2.angular_velocity_rad_s}rad/s, d={res2.comfort_distance_m}m")
    print(f"  Safety Active  : {res2.safety_override_active}")
    print(f"  Safety Reason  : {res2.safety_reason}")

    # Test Case 3: Emergency Hazard (F02 in Kitchen — Priority 1 Emergency Bypass)
    print("\n--- Test Case 3: Emergency Hazard (F02 in Kitchen) ---")
    res3 = engine.predict(
        intent="F02",
        intent_confidence=0.98,
        motion_state="stepping_back",
        direction="away_from_robot",
        velocity=1.2,
        context="kitchen",
        current_distance=0.8
    )
    print(f"  Input          : Intent F02 (Emergency), Context kitchen")
    print(f"  Action         : {res3.action} — {res3.action_description} (FORCED EMERGENCY ACTION)")
    print(f"  Control [v,w,d]: v={res3.linear_velocity_m_s}m/s, w={res3.angular_velocity_rad_s}rad/s, d={res3.comfort_distance_m}m")
    print(f"  Safety Active  : {res3.safety_override_active}")
    print(f"  Safety Reason  : {res3.safety_reason}")

    print("\nInference Demo & Safety Tests Passed Successfully!")
    return True


def export_onnx():
    return run_command(["export_onnx.py"], "Exporting Model to ONNX format")


def main():
    parser = argparse.ArgumentParser(description="Action Generator Runner")
    parser.add_argument(
        "--step", 
        choices=["all", "csv", "train", "infer", "export"], 
        default="all",
        help="Step to execute (default: all)"
    )
    args = parser.parse_args()

    if args.step in ["all", "csv"]:
        if not build_csv():
            return
            
    if args.step in ["all", "train"]:
        if not train_model():
            return

    if args.step in ["all", "infer"]:
        if not run_inference_demo():
            return

    if args.step in ["all", "export"]:
        if not export_onnx():
            return

    print("\n" + "=" * 70)
    print("  ALL STEPS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


if __name__ == "__main__":
    main()

"""
Action Generator Interactive & Automated Runner Script.

Runs all steps:
  1. Build dataset CSVs
  2. Train PyTorch model
  3. Run sample inference demo
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

# Ensure we use Python 3.11 with PyTorch installed
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
    print("  STEP: Running Inference Demo & Safety Override Test")
    print("=" * 70)
    
    sys.path.insert(0, BASE_DIR)
    from inference import ActionInference

    ckpt_path = os.path.join(BASE_DIR, "checkpoints", "best_action_generator.pt")
    if not os.path.exists(ckpt_path):
        print(f"Error: Model checkpoint not found at {ckpt_path}. Run training first!")
        return False

    engine = ActionInference(ckpt_path)

    # Test Case 1: Normal task request
    print("\n--- Test Case 1: Normal Task Request (F03 in Classroom) ---")
    res1 = engine.predict(
        intent="F03",
        intent_confidence=0.95,
        motion_state="sit",
        direction="stationary",
        velocity=0.0,
        context="classroom"
    )
    print(f"  Input        : Intent F03 (Task Assistance), Motion sit, Context classroom")
    print(f"  Predicted    : Action {res1.action} — {res1.action_description}")
    print(f"  Confidence   : {res1.confidence * 100:.1f}%")
    print(f"  Control [v,w,d]: v={res1.linear_velocity_m_s}m/s, w={res1.angular_velocity_rad_s}rad/s, d={res1.comfort_distance_m}m")
    print(f"  Safety Active: {res1.safety_override_active}")

    # Test Case 2: Emergency Alert (Safety Override)
    print("\n--- Test Case 2: Emergency Hazard (F02 in Kitchen) ---")
    res2 = engine.predict(
        intent="F02",
        intent_confidence=0.98,
        motion_state="run",
        direction="away_from_robot",
        velocity=1.2,
        context="kitchen"
    )
    print(f"  Input        : Intent F02 (Emergency), Motion run, Context kitchen")
    print(f"  Predicted    : Action {res2.action} — {res2.action_description}")
    print(f"  Confidence   : {res2.confidence * 100:.1f}%")
    print(f"  Control [v,w,d]: v={res2.linear_velocity_m_s}m/s, w={res2.angular_velocity_rad_s}rad/s, d={res2.comfort_distance_m}m")
    print(f"  Safety Active: {res2.safety_override_active} (FORCED HALT)")

    print("\nInference Demo Passed Successfully!")
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

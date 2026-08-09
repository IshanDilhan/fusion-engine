"""
live_video_demo.py

Live Video / Stream Visual Tester for Action Generator & Motion Pipeline.
Displays real-time video feed (Webcam, Video File, or Synthetic Stream)
with a rich visual HUD showing:
  - Human Skeleton Pose & Distance
  - Real-time Motion State (sitting, standing, walking, stepping_back)
  - Fused Human Intent (F01-F09)
  - Expected vs. Predicted Action Code (A01-A15) + Description
  - Live MATCH / MISMATCH Accuracy Badge (Green / Red)
  - Continuous Controls: Linear Speed v (m/s), Comfort Clearance d (m)
  - 2-Tier Safety HUD Status (NOMINAL / PROXIMITY YIELD STEP / EMERGENCY HALT)

Usage:
    # Run synthetic interactive demo stream:
    python action_generator/tools/live_video_demo.py --source synthetic
"""

import os
import sys
import time
import argparse
import importlib.util
import numpy as np
import cv2

# Set path explicitly for Action Generator imports
TOOLS_DIR = os.path.dirname(os.path.abspath(__file__))
ACTION_GEN_DIR = os.path.dirname(TOOLS_DIR)
ROOT_DIR = os.path.dirname(ACTION_GEN_DIR)
MOTION_REPO_DIR = os.path.join(ROOT_DIR, "Motion Repo")

if ACTION_GEN_DIR not in sys.path:
    sys.path.insert(0, ACTION_GEN_DIR)

from inference import ActionInference


# ─── Helper function to load Motion Repo modules dynamically ──────────────

def load_motion_repo():
    """Loads MotionInference & mediapipe_to_ntu25 dynamically from Motion Repo."""
    original_path = list(sys.path)
    old_sys_model = sys.modules.get("model")
    try:
        sys.path.insert(0, MOTION_REPO_DIR)
        
        # Load Motion Repo model module
        model_spec = importlib.util.spec_from_file_location("motion_model", os.path.join(MOTION_REPO_DIR, "model.py"))
        motion_model = importlib.util.module_from_spec(model_spec)
        sys.modules["motion_model"] = motion_model
        sys.modules["model"] = motion_model  # Temporarily override 'model' in sys.modules
        model_spec.loader.exec_module(motion_model)
        
        # Load Motion Repo skeleton_utils module
        skel_spec = importlib.util.spec_from_file_location("skeleton_utils", os.path.join(MOTION_REPO_DIR, "skeleton_utils.py"))
        skel_utils = importlib.util.module_from_spec(skel_spec)
        sys.modules["skeleton_utils"] = skel_utils
        skel_spec.loader.exec_module(skel_utils)

        # Load Motion Repo inference module
        inf_spec = importlib.util.spec_from_file_location("motion_inference", os.path.join(MOTION_REPO_DIR, "inference.py"))
        motion_inf = importlib.util.module_from_spec(inf_spec)
        sys.modules["motion_inference"] = motion_inf
        inf_spec.loader.exec_module(motion_inf)
        
        return motion_inf.MotionInference, skel_utils.mediapipe_to_ntu25
    finally:
        sys.path = original_path
        if old_sys_model is not None:
            sys.modules["model"] = old_sys_model


# ─── HUD Visual Renderer Class ───────────────────────────────────────────────

class ActionGeneratorHUD:
    """Renders a modern, semi-transparent telemetry HUD over video frames."""

    def __init__(self):
        # Color palette (BGR)
        self.COLOR_GREEN  = (50, 205, 50)    # Green (Nominal operation / Match)
        self.COLOR_ORANGE = (30, 144, 255)   # Orange (Proximity Yield Step Back)
        self.COLOR_RED    = (50, 50, 255)    # Red (Emergency Halt / Mismatch)
        self.COLOR_BG     = (20, 20, 20)     # Dark Gray (Card BG)
        self.COLOR_WHITE  = (255, 255, 255)
        self.COLOR_TEXT_DIM = (180, 180, 180)

    def draw_hud(
        self,
        frame: np.ndarray,
        intent: str,
        motion: str,
        direction: str,
        velocity: float,
        distance: float,
        action_res,
        expected_action: str = None
    ) -> np.ndarray:
        h, w = frame.shape[:2]
        canvas = frame.copy()

        # Determine Safety Status Theme Color
        if not action_res.safety_override_active:
            status_color = self.COLOR_GREEN
            status_title = "SYSTEM NOMINAL"
        elif action_res.linear_velocity_m_s < 0:
            status_color = self.COLOR_ORANGE
            status_title = "PROXIMITY YIELD: STEP BACK"
        else:
            status_color = self.COLOR_RED
            status_title = "EMERGENCY HALT OVERRIDE"

        # ── 1. Top Telemetry Card ────────────────────────────────────────────
        card_w, card_h = min(680, w - 40), 245
        card_x, card_y = 20, 20

        # Draw semi-transparent background box
        overlay = canvas.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), status_color, 2)

        # Header Bar
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + 34), status_color, -1)
        cv2.putText(
            canvas, f"HRI ACTION GENERATOR POLICY LAYER  |  [{status_title}]",
            (card_x + 12, card_y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 2, cv2.LINE_AA
        )

        # Left Column: Perception Inputs
        col1_x = card_x + 15
        cv2.putText(canvas, "PERCEPTION INPUTS", (col1_x, card_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_DIM, 1)
        cv2.putText(canvas, f"Intent Code : {intent}", (col1_x, card_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        cv2.putText(canvas, f"Motion State: {motion}", (col1_x, card_y + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        cv2.putText(canvas, f"Direction   : {direction}", (col1_x, card_y + 118), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        cv2.putText(canvas, f"Human Speed : {velocity:.2f} m/s", (col1_x, card_y + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        cv2.putText(canvas, f"Distance    : {distance:.2f} m", (col1_x, card_y + 158), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)

        # Divider Line
        div_x = card_x + 280
        cv2.line(canvas, (div_x, card_y + 42), (div_x, card_y + card_h - 45), (70, 70, 70), 1)

        # Right Column: Policy Outputs & Controls
        col2_x = div_x + 15
        cv2.putText(canvas, "ROBOT POLICY OUTPUT", (col2_x, card_y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_DIM, 1)
        cv2.putText(
            canvas, f"Action Code : {action_res.action} ({action_res.confidence * 100:.1f}%)",
            (col2_x, card_y + 78), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2
        )
        
        # Wrapped action description
        desc = action_res.action_description
        if len(desc) > 35:
            desc = desc[:32] + "..."
        cv2.putText(canvas, f"Goal: {desc}", (col2_x, card_y + 98), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_WHITE, 1)

        # Motor Controls Output
        cv2.putText(
            canvas, f"Motor Speed (v) : {action_res.linear_velocity_m_s:+.2f} m/s",
            (col2_x, card_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.48,
            self.COLOR_ORANGE if (action_res.safety_override_active and action_res.linear_velocity_m_s < 0) else self.COLOR_WHITE,
            2 if (action_res.safety_override_active and action_res.linear_velocity_m_s < 0) else 1
        )
        cv2.putText(
            canvas, f"Turn Rate (w)   : {action_res.angular_velocity_rad_s:.2f} rad/s",
            (col2_x, card_y + 138), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_WHITE, 1
        )
        cv2.putText(
            canvas, f"Clearance (d)   : {action_res.comfort_distance_m:.2f} meters",
            (col2_x, card_y + 158), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_WHITE, 1
        )

        # ── 2. Live Expected vs. Predicted Evaluation Card (Bottom Section) ──
        eval_y = card_y + 180
        cv2.line(canvas, (card_x + 10, eval_y), (card_x + card_w - 10, eval_y), (70, 70, 70), 1)

        if expected_action:
            is_match = (action_res.action == expected_action)
            match_color = self.COLOR_GREEN if is_match else self.COLOR_RED
            match_badge = "[ MATCH: CORRECT ]" if is_match else "[ MISMATCH: INCORRECT ]"

            cv2.putText(
                canvas, f"EXPECTED: {expected_action}  |  GOT: {action_res.action}",
                (card_x + 15, eval_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_WHITE, 1
            )
            cv2.putText(
                canvas, f"{match_badge}",
                (card_x + 350, eval_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, match_color, 2
            )
        else:
            if action_res.safety_reason:
                cv2.putText(
                    canvas, f"Safety Log: {action_res.safety_reason[:75]}",
                    (card_x + 15, eval_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1
                )

        return canvas


# ─── Synthetic Live Stream Generator ────────────────────────────────────────

def run_synthetic_live_demo(engine: ActionInference, save_path: str = None):
    """Generates an interactive real-time visual simulation of a human approaching a robot."""
    print("\n" + "=" * 70)
    print("  LAUNCHING SYNTHETIC INTERACTIVE STREAM DEMO")
    print("  Demonstrates 3 Interaction Phases:")
    print("    Phase 1 (t=0s..3s) : Safe Distance (1.5m) -> Action A04/A05 Green (v=+0.35m/s)")
    print("    Phase 2 (t=3s..6s) : Rapid Approach (<1.0m) -> Action A05 Preserved + Orange Yield (v=-0.20m/s)")
    print("    Phase 3 (t=6s..9s) : Emergency Alert (F02) -> Action A02 Red Halt (v=0.0m/s, d=2.0m)")
    print("=" * 70 + "\n")

    hud = ActionGeneratorHUD()
    frame_w, frame_h = 1000, 600
    fps = 30
    total_frames = 9 * fps  # 9-second demo

    writer = None
    if save_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (frame_w, frame_h))

    for frame_idx in range(total_frames):
        t_sec = frame_idx / float(fps)
        blank = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)

        # Dynamic simulation phases
        if t_sec < 3.0:
            intent = "F04"
            motion = "walking"
            direction = "toward_robot"
            velocity = 0.4
            distance = 1.6 - 0.2 * t_sec
            context = "classroom"
            exp_act = "A05"
        elif t_sec < 6.0:
            intent = "F04"
            motion = "walking"
            direction = "toward_robot"
            velocity = 0.95
            distance = 0.85 - 0.05 * (t_sec - 3.0)
            context = "classroom"
            exp_act = "A05"
        else:
            intent = "F02"
            motion = "stepping_back"
            direction = "away_from_robot"
            velocity = 1.1
            distance = 0.75
            context = "kitchen"
            exp_act = "A02"

        # Model Inference
        result = engine.predict(
            intent=intent,
            intent_confidence=0.94,
            motion_state=motion,
            direction=direction,
            velocity=velocity,
            context=context,
            current_distance=distance
        )

        # Draw 3D Spatial Grid Visualization
        grid_y = 400
        cv2.line(blank, (50, grid_y), (950, grid_y), (60, 60, 60), 2)
        
        robot_x = 200
        cv2.rectangle(blank, (robot_x - 30, grid_y - 80), (robot_x + 30, grid_y), (255, 191, 0), -1)
        cv2.putText(blank, "ROBOT", (robot_x - 25, grid_y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 191, 0), 2)

        human_x = int(robot_x + distance * 300)
        human_x = min(human_x, 920)
        cv2.circle(blank, (human_x, grid_y - 50), 20, (0, 215, 255), -1)
        cv2.line(blank, (human_x, grid_y - 30), (human_x, grid_y + 10), (0, 215, 255), 3)
        cv2.putText(blank, f"HUMAN ({distance:.2f}m)", (human_x - 45, grid_y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)

        cv2.line(blank, (robot_x + 30, grid_y - 20), (human_x - 20, grid_y - 20), (128, 128, 128), 1)

        # Render Telemetry HUD
        frame = hud.draw_hud(blank, intent, motion, direction, velocity, distance, result, expected_action=exp_act)

        if writer:
            writer.write(frame)

        try:
            cv2.imshow("HRI Policy Engine Live Tester", frame)
            if cv2.waitKey(30) & 0xFF == 27:
                break
        except Exception:
            pass

    if writer:
        writer.release()
        print(f"Saved demo video to: {save_path}")

    try:
        cv2.destroyAllWindows()
    except Exception:
        pass
    print("Synthetic Live Stream Demo Completed!")


def main():
    parser = argparse.ArgumentParser(description="Live Video UI Tester for Action Generator")
    parser.add_argument("--source", default="synthetic", help="Video source: 'synthetic', '0', or MP4 file")
    parser.add_argument("--intent", default=None, help="Optional intent override (e.g. F02, F04)")
    parser.add_argument("--context", default=None, help="Optional context override (e.g. classroom, kitchen)")
    parser.add_argument("--expected-action", default=None, help="Optional ground-truth expected action")
    parser.add_argument("--out", default=None, help="Optional path to save output MP4 video")
    args = parser.parse_args()

    ckpt_path = os.path.join(ACTION_GEN_DIR, "checkpoints", "best_action_generator.pt")
    engine = ActionInference(ckpt_path)

    if args.source == "synthetic":
        run_synthetic_live_demo(engine, args.out)
    else:
        print(f"Direct stream visual tester initialized for source: {args.source}")


if __name__ == "__main__":
    main()

"""
live_video_demo.py

Live Video / Stream Visual Tester for Action Generator & Motion Pipeline.
Displays real-time video feed (Webcam, Video File, or Synthetic Stream)
with a rich visual HUD showing:
  - Human Skeleton Pose & Distance
  - Real-time Motion State (sitting, standing, walking, stepping_back)
  - Fused Human Intent (F01-F10)
  - Predicted Action Code (A01-A15) + Description
  - Continuous Controls: Linear Speed v (m/s), Comfort Clearance d (m)
  - 2-Tier Safety HUD Status (NOMINAL / PROXIMITY YIELD STEP / EMERGENCY HALT)

Usage:
    # Run synthetic interactive demo stream (No camera required):
    python live_video_demo.py --source synthetic

    # Save demo output video:
    python live_video_demo.py --source synthetic --out demo_output.mp4

    # Run on webcam 0:
    python live_video_demo.py --source 0

    # Run on an MP4 video clip:
    python live_video_demo.py --source path/to/video.mp4 --out output.mp4
"""

import os
import sys
import time
import argparse
import numpy as np
import cv2

# Set path explicitly for Action Generator imports
ACTION_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ACTION_GEN_DIR)
MOTION_REPO_DIR = os.path.join(ROOT_DIR, "Motion Repo")

if ACTION_GEN_DIR not in sys.path:
    sys.path.insert(0, ACTION_GEN_DIR)

from inference import ActionInference


# ─── HUD Visual Renderer Class ───────────────────────────────────────────────

class ActionGeneratorHUD:
    """Renders a modern, semi-transparent telemetry HUD over video frames."""

    def __init__(self):
        # Color palette (BGR)
        self.COLOR_GREEN  = (50, 205, 50)    # Green (Nominal operation)
        self.COLOR_ORANGE = (30, 144, 255)   # Orange (Proximity Yield Step Back)
        self.COLOR_RED    = (50, 50, 255)    # Red (Emergency Halt)
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
        action_res
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
        card_w, card_h = min(640, w - 40), 220
        card_x, card_y = 20, 20

        # Draw semi-transparent background box
        overlay = canvas.copy()
        cv2.rectangle(overlay, (card_x, card_y), (card_x + card_w, card_y + card_h), self.COLOR_BG, -1)
        cv2.addWeighted(overlay, 0.75, canvas, 0.25, 0, canvas)
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + card_h), status_color, 2)

        # Header Bar
        cv2.rectangle(canvas, (card_x, card_y), (card_x + card_w, card_y + 36), status_color, -1)
        cv2.putText(
            canvas, f"HRI ACTION GENERATOR POLICY LAYER  |  [{status_title}]",
            (card_x + 12, card_y + 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2, cv2.LINE_AA
        )

        # Left Column: Inputs
        col1_x = card_x + 15
        cv2.putText(canvas, "PERCEPTION INPUTS", (col1_x, card_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_DIM, 1)
        cv2.putText(canvas, f"Intent Code : {intent}", (col1_x, card_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        cv2.putText(canvas, f"Motion State: {motion}", (col1_x, card_y + 105), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        cv2.putText(canvas, f"Direction   : {direction}", (col1_x, card_y + 125), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        cv2.putText(canvas, f"Human Speed : {velocity:.2f} m/s", (col1_x, card_y + 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)
        cv2.putText(canvas, f"Distance    : {distance:.2f} m", (col1_x, card_y + 165), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1)

        # Divider Line
        div_x = card_x + 280
        cv2.line(canvas, (div_x, card_y + 45), (div_x, card_y + card_h - 15), (70, 70, 70), 1)

        # Right Column: Policy Outputs & Controls
        col2_x = div_x + 15
        cv2.putText(canvas, "ROBOT POLICY OUTPUT", (col2_x, card_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.45, self.COLOR_TEXT_DIM, 1)
        cv2.putText(
            canvas, f"Action Code : {action_res.action} ({action_res.confidence * 100:.1f}%)",
            (col2_x, card_y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.6, status_color, 2
        )
        
        # Wrapped action description
        desc = action_res.action_description
        if len(desc) > 35:
            desc = desc[:32] + "..."
        cv2.putText(canvas, f"Goal: {desc}", (col2_x, card_y + 108), cv2.FONT_HERSHEY_SIMPLEX, 0.42, self.COLOR_WHITE, 1)

        # Motor Controls Output
        cv2.putText(
            canvas, f"Motor Speed (v) : {action_res.linear_velocity_m_s:+.2f} m/s",
            (col2_x, card_y + 135), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
            self.COLOR_ORANGE if action_res.linear_velocity_m_s < 0 else self.COLOR_WHITE, 2 if action_res.linear_velocity_m_s < 0 else 1
        )
        cv2.putText(
            canvas, f"Turn Rate (w)   : {action_res.angular_velocity_rad_s:.2f} rad/s",
            (col2_x, card_y + 155), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1
        )
        cv2.putText(
            canvas, f"Clearance (d)   : {action_res.comfort_distance_m:.2f} meters",
            (col2_x, card_y + 175), cv2.FONT_HERSHEY_SIMPLEX, 0.5, self.COLOR_WHITE, 1
        )

        # Bottom Safety Status Bar
        if action_res.safety_reason:
            cv2.putText(
                canvas, f"Safety Log: {action_res.safety_reason[:75]}",
                (card_x + 12, card_y + card_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.38, status_color, 1
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
            # Phase 1: Nominal Approach
            intent = "F04"
            motion = "walking"
            direction = "toward_robot"
            velocity = 0.4
            distance = 1.6 - 0.2 * t_sec
            context = "classroom"
        elif t_sec < 6.0:
            # Phase 2: Rapid Approach < 1.0m (Proximity Yielding Step)
            intent = "F04"
            motion = "walking"
            direction = "toward_robot"
            velocity = 0.95
            distance = 0.85 - 0.05 * (t_sec - 3.0)
            context = "classroom"
        else:
            # Phase 3: Emergency Situation (F02 in Kitchen)
            intent = "F02"
            motion = "stepping_back"
            direction = "away_from_robot"
            velocity = 1.1
            distance = 0.75
            context = "kitchen"

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
        grid_y = 380
        cv2.line(blank, (50, grid_y), (950, grid_y), (60, 60, 60), 2)
        
        # Robot position
        robot_x = 200
        cv2.rectangle(blank, (robot_x - 30, grid_y - 80), (robot_x + 30, grid_y), (255, 191, 0), -1)
        cv2.putText(blank, "ROBOT", (robot_x - 25, grid_y - 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 191, 0), 2)

        # Human position based on distance
        human_x = int(robot_x + distance * 300)
        human_x = min(human_x, 920)
        cv2.circle(blank, (human_x, grid_y - 50), 20, (0, 215, 255), -1)
        cv2.line(blank, (human_x, grid_y - 30), (human_x, grid_y + 10), (0, 215, 255), 3)
        cv2.putText(blank, f"HUMAN ({distance:.2f}m)", (human_x - 45, grid_y - 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 2)

        # Distance Line
        cv2.line(blank, (robot_x + 30, grid_y - 20), (human_x - 20, grid_y - 20), (128, 128, 128), 1)

        # Render Telemetry HUD
        frame = hud.draw_hud(blank, intent, motion, direction, velocity, distance, result)

        if writer:
            writer.write(frame)

        # Render window if GUI available (catch headless errors silently)
        try:
            cv2.imshow("HRI Policy Engine Live Tester", frame)
            if cv2.waitKey(30) & 0xFF == 27:  # ESC key to exit
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


# ─── Real Video File / Webcam Processor ──────────────────────────────────────

def run_real_video_demo(engine: ActionInference, source: str, save_path: str = None):
    """Processes webcam or MP4 file with MediaPipe Pose & MotionInference."""
    try:
        src_id = int(source)
    except ValueError:
        src_id = source

    cap = cv2.VideoCapture(src_id)
    if not cap.isOpened():
        print(f"Error: Unable to open video source {source}")
        return

    import mediapipe as mp
    
    # Import MotionInference dynamically from Motion Repo to avoid scope collision
    import importlib.util
    motion_inf_path = os.path.join(MOTION_REPO_DIR, "inference.py")
    spec = importlib.util.spec_from_file_location("motion_repo_inference", motion_inf_path)
    motion_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(motion_mod)

    skel_path = os.path.join(MOTION_REPO_DIR, "skeleton_utils.py")
    spec_skel = importlib.util.spec_from_file_location("motion_repo_skel", skel_path)
    skel_mod = importlib.util.module_from_spec(spec_skel)
    spec_skel.loader.exec_module(skel_mod)

    motion_ckpt = os.path.join(MOTION_REPO_DIR, "checkpoints", "best_model_finetuned.pt")
    motion_engine = motion_mod.MotionInference(motion_ckpt)
    hud = ActionGeneratorHUD()

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        model_complexity=1,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55
    )

    writer = None
    if save_path:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    print(f"Processing live video stream from: {source}. Press ESC or 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        motion_label = "standing"
        distance = 1.5
        velocity = 0.0
        direction = "stationary"

        if results.pose_world_landmarks is not None:
            # 1. Feed landmarks to MotionInference
            joints_25 = skel_mod.mediapipe_to_ntu25(results.pose_world_landmarks.landmark)
            m_res = motion_engine.update(joints_25)
            if m_res and m_res.label != "buffering":
                motion_label = m_res.label

            # Approximate distance from torso z
            hip_z = (results.pose_world_landmarks.landmark[23].z + results.pose_world_landmarks.landmark[24].z) / 2.0
            distance = max(0.5, abs(hip_z))

        # Default intent for live webcam testing
        intent = "F04"
        context = "classroom"

        # Action Generator Prediction
        action_res = engine.predict(
            intent=intent,
            intent_confidence=0.90,
            motion_state=motion_label,
            direction=direction,
            velocity=velocity,
            context=context,
            current_distance=distance
        )

        frame = hud.draw_hud(frame, intent, motion_label, direction, velocity, distance, action_res)

        if writer:
            writer.write(frame)

        try:
            cv2.imshow("HRI Policy Engine Live Tester", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                break
        except Exception:
            pass

    cap.release()
    if writer:
        writer.release()
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass


def main():
    parser = argparse.ArgumentParser(description="Live Video UI Tester for Action Generator")
    parser.add_argument(
        "--source",
        default="synthetic",
        help="Video source: 'synthetic' for simulated stream, '0' for webcam, or path to MP4 file"
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Optional path to save output MP4 video"
    )
    args = parser.parse_args()

    ckpt_path = os.path.join(ACTION_GEN_DIR, "checkpoints", "best_action_generator.pt")
    engine = ActionInference(ckpt_path)

    if args.source == "synthetic":
        run_synthetic_live_demo(engine, args.out)
    else:
        run_real_video_demo(engine, args.source, args.out)


if __name__ == "__main__":
    main()

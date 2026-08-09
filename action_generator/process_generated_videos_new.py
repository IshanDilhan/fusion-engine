"""
process_generated_videos_new.py

Batch processor for Generated_Videos_new folder.
Iterates over folders (1, 2, 4, 5, 6), selects up to 3 video files per folder,
runs live_video_demo with the assigned intent, context, and expected action,
and saves rendered output videos with HUD overlays.
"""

import os
import sys
import glob
import subprocess

PYTHON_EXE = sys.executable
ACTION_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ACTION_GEN_DIR)
GEN_VIDEOS_NEW_DIR = os.path.join(ROOT_DIR, "Generated_Videos_new")

# Scenario Map for Generated_Videos_new
SCENARIO_MAP = {
    "1": {"intent": "F01", "action": "A01", "context": "classroom"},
    "2": {"intent": "F01", "action": "A01", "context": "classroom"},
    "4": {"intent": "F02", "action": "A02", "context": "classroom"},
    "5": {"intent": "F01", "action": "A09", "context": "classroom"},
    "6": {"intent": "F02", "action": "A02", "context": "classroom"},
}


def process_all_folders(max_videos_per_folder=3):
    print("=" * 80)
    print("  GENERATED_VIDEOS_NEW BATCH PROCESSOR")
    print(f"  Target Directory: {GEN_VIDEOS_NEW_DIR}")
    print(f"  Max Videos per Folder: {max_videos_per_folder}")
    print("=" * 80)

    if not os.path.exists(GEN_VIDEOS_NEW_DIR):
        print(f"Error: Generated_Videos_new directory not found at {GEN_VIDEOS_NEW_DIR}")
        return

    demo_script = os.path.join(ACTION_GEN_DIR, "live_video_demo.py")
    total_processed = 0

    folder_names = sorted(os.listdir(GEN_VIDEOS_NEW_DIR), key=lambda x: int(x) if x.isdigit() else 999)
    for fname in folder_names:
        folder_path = os.path.join(GEN_VIDEOS_NEW_DIR, fname)
        if not os.path.isdir(folder_path):
            continue

        meta = SCENARIO_MAP.get(fname, {"intent": "F04", "action": "A05", "context": "classroom"})
        intent = meta["intent"]
        action = meta["action"]
        context = meta["context"]

        # Find raw MP4 video files (exclude existing output HUD files)
        all_videos = [
            f for f in glob.glob(os.path.join(folder_path, "*.mp4"))
            if not f.endswith("_hud.mp4") and "output_with_hud" not in f
        ]

        selected_videos = all_videos[:max_videos_per_folder]
        print(f"\n--- Folder: {fname} (Found {len(all_videos)} raw videos, Processing {len(selected_videos)}) ---")
        print(f"    Intent: {intent}  | Expected Action: {action}  | Context: {context}")

        for vid_path in selected_videos:
            base_name = os.path.splitext(os.path.basename(vid_path))[0]
            out_path = os.path.join(folder_path, f"{base_name}_hud.mp4")

            print(f"  > Input : {os.path.basename(vid_path)}")
            print(f"    Output: {os.path.basename(out_path)}")

            cmd = [
                PYTHON_EXE, demo_script,
                "--source", vid_path,
                "--intent", intent,
                "--context", context,
                "--expected-action", action,
                "--out", out_path
            ]

            res = subprocess.run(cmd, cwd=ACTION_GEN_DIR)
            if res.returncode == 0:
                total_processed += 1
            else:
                print(f"    [!] Error processing {vid_path}")

    print("\n" + "=" * 80)
    print(f"  BATCH PROCESSING COMPLETE: {total_processed} videos rendered with HUD!")
    print("=" * 80)


if __name__ == "__main__":
    process_all_folders(max_videos_per_folder=3)

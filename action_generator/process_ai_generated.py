"""
process_ai_generated.py

Batch processor for AI Genated videos.
Iterates over all scenario folders in 'AI Genated/Classroom' and 'AI Genated/Kitchen',
selects up to 3 video files per folder, runs live_video_demo with the appropriate
intent, context, and expected action, and saves rendered output videos with HUD overlays.
"""

import os
import sys
import glob
import subprocess

PYTHON_EXE = sys.executable
ACTION_GEN_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(ACTION_GEN_DIR)
AI_GEN_DIR = os.path.join(ROOT_DIR, "AI Genated")

# 73-Scenario Intent & Expected Action Mapping Registry
SCENARIO_MAP = {
    # Classroom Scenarios (1-13)
    "Classroom/1":  {"intent": "F01", "action": "A01", "context": "classroom"},
    "Classroom/2":  {"intent": "F01", "action": "A01", "context": "classroom"},
    "Classroom/3":  {"intent": "F02", "action": "A14", "context": "classroom"},
    "Classroom/4":  {"intent": "F02", "action": "A14", "context": "classroom"},
    "Classroom/5":  {"intent": "F03", "action": "A04", "context": "classroom"},
    "Classroom/6":  {"intent": "F03", "action": "A04", "context": "classroom"},
    "Classroom/7":  {"intent": "F04", "action": "A05", "context": "classroom"},
    "Classroom/8":  {"intent": "F04", "action": "A05", "context": "classroom"},
    "Classroom/9":  {"intent": "F04", "action": "A05", "context": "classroom"},
    "Classroom/10": {"intent": "F05", "action": "A06", "context": "classroom"},
    "Classroom/11": {"intent": "F05", "action": "A06", "context": "classroom"},
    "Classroom/12": {"intent": "F06", "action": "A11", "context": "classroom"},
    "Classroom/13": {"intent": "F06", "action": "A11", "context": "classroom"},

    # Kitchen Scenarios (18-26)
    "Kitchen/18": {"intent": "F01", "action": "A01", "context": "kitchen"},
    "Kitchen/19": {"intent": "F02", "action": "A02", "context": "kitchen"},
    "Kitchen/20": {"intent": "F03", "action": "A04", "context": "kitchen"},
    "Kitchen/21": {"intent": "F04", "action": "A05", "context": "kitchen"},
    "Kitchen/22": {"intent": "F05", "action": "A06", "context": "kitchen"},
    "Kitchen/23": {"intent": "F06", "action": "A11", "context": "kitchen"},
    "Kitchen/24": {"intent": "F07", "action": "A08", "context": "kitchen"},
    "Kitchen/25": {"intent": "F01", "action": "A09", "context": "kitchen"},
    "Kitchen/26": {"intent": "F02", "action": "A02", "context": "kitchen"},
}


def process_all_folders(max_videos_per_folder=3):
    print("=" * 80)
    print("  AI GENATED BATCH VIDEO PROCESSOR (73-Scenario Registry)")
    print(f"  Target Directory: {AI_GEN_DIR}")
    print(f"  Max Videos per Folder: {max_videos_per_folder}")
    print("=" * 80)

    if not os.path.exists(AI_GEN_DIR):
        print(f"Error: AI Genated directory not found at {AI_GEN_DIR}")
        return

    demo_script = os.path.join(ACTION_GEN_DIR, "live_video_demo.py")
    total_processed = 0

    for domain in ["Classroom", "Kitchen"]:
        domain_dir = os.path.join(AI_GEN_DIR, domain)
        if not os.path.exists(domain_dir):
            continue

        folder_names = sorted(os.listdir(domain_dir), key=lambda x: int(x) if x.isdigit() else 999)
        for fname in folder_names:
            folder_path = os.path.join(domain_dir, fname)
            if not os.path.isdir(folder_path):
                continue

            rel_key = f"{domain}/{fname}"
            meta = SCENARIO_MAP.get(rel_key, {"intent": "F04", "action": "A05", "context": domain.lower()})
            intent = meta["intent"]
            action = meta["action"]
            context = meta["context"]

            # Find raw MP4 video files (exclude existing output HUD files)
            all_videos = [
                f for f in glob.glob(os.path.join(folder_path, "*.mp4"))
                if not f.endswith("_hud.mp4") and "output_with_hud" not in f
            ]

            selected_videos = all_videos[:max_videos_per_folder]
            print(f"\n--- Folder: {rel_key} (Found {len(all_videos)} raw videos, Processing {len(selected_videos)}) ---")
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

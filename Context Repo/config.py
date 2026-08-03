"""
Top-level configuration for the context modality (scene + VLM pipeline).

The scene classifier keeps its own config (scene_classification/config.py).
This file holds the settings of the pipeline layer that combines the CLIP
scene label with the small-VLM situation analysis into a ContextState.
"""
import os

ROOT = os.path.dirname(os.path.abspath(__file__))

# Scene changes slowly, so we only re-classify every N frames to save compute.
SCENE_EVERY = 5

# ── VLM (overall context) ────────────────────────────────────────────────
# Small vision-language model that describes the situation (people, activity,
# attention, objects, summary). Chosen for the Jetson Orin Nano 8GB budget.
VLM_MODEL = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"
VLM_MAX_NEW_TOKENS = 160

# Realtime (async) mode: analyse a new frame at most this often. The camera
# loop never blocks on the VLM — it always carries the latest finished result.
VLM_INTERVAL_SEC = 2.0

# Offline video (sync) mode: analyse every Nth processed frame.
VLM_EVERY_FRAMES = 60

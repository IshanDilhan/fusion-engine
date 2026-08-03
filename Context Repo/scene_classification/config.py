"""
Central configuration for the scene-classification sub-model of the context
modality.

All paths are absolute (derived from this file's location) so every script
works regardless of the current working directory.

Dataset:  Places365 subset, ImageFolder layout under ../data/scene/.
Classes:  the two environments we deploy in — classroom, kitchen.
Target:   lightweight CNN for Jetson Orin Nano scene understanding.
"""
import os

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.normpath(os.path.join(ROOT, "..", "data", "scene"))
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
REPORT_DIR = os.path.join(ROOT, "reports")
CLASSES_FILE = os.path.join(CHECKPOINT_DIR, "classes.json")

# ── Dataset / model ──────────────────────────────────────────────────────
# Class count of the trained CNN baseline (its label order lives in
# checkpoints/classes.json, written at training time from the dataset folders).
NUM_CLASSES = 2
IMAGE_SIZE = 224

# The deployed scene vocabulary (zero-shot CLIP backend). Each label needs a
# prompt list in SCENE_PROMPTS below — extending this list is the whole
# process of adding an environment, no dataset or retraining required.
# (The CNN baseline still covers only classroom/kitchen via classes.json.)
SCENE_LABELS = ["classroom", "kitchen", "hospital", "cloth_store", "museum"]

# ── Default hyper-parameters ─────────────────────────────────────────────
BATCH_SIZE = 32
NUM_WORKERS = 4
LR = 1e-4                 # stage-2 (full fine-tune) base LR
HEAD_LR = 1e-3            # stage-1 (head-only) LR
WEIGHT_DECAY = 1e-5
LABEL_SMOOTHING = 0.05
MIXUP_ALPHA = 0.2
WARMUP_EPOCHS = 1
STAGE1_EPOCHS = 5         # head-only warm-up
STAGE2_EPOCHS = 15        # full fine-tune
SEED = 42

# ImageNet normalization (pretrained backbones expect this)
NORM_MEAN = [0.485, 0.456, 0.406]
NORM_STD = [0.229, 0.224, 0.225]

# Deployment budget: models above this size are flagged in the comparison.
SIZE_BUDGET_MB = 20.0

# ── Deployed model (what inference ships with) ───────────────────────────
# EfficientNet-B0 won the CNN comparison (best accuracy within the size budget).
# To ship a different CNN: drop its .pth in checkpoints/ and update both lines.
DEFAULT_MODEL = "EfficientNet-B0"
DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_EfficientNet_B0.pth")

# ── Scene backend selection ──────────────────────────────────────────────
# "clip": zero-shot CLIP image-text matching (deployed). On the captured clips
#         it scores 99.5% overall vs the trained CNN's 82.2% (kitchen domain
#         gap) — see reports/zero_shot/ZERO_SHOT_REPORT.md. Adding a scene
#         class is a prompt edit below, no retraining.
# "cnn":  the trained EfficientNet-B0 (kept as the evaluated baseline).
SCENE_BACKEND = "clip"

CLIP_MODEL = "ViT-B-32-quickgelu"   # open_clip model name
CLIP_PRETRAINED = "openai"          # weights tag (auto-downloads on first use)

# Prompt ensembles per scene class. Keys MUST match SCENE_LABELS order.
SCENE_PROMPTS = {
    "classroom": [
        "a photo of a classroom",
        "a photo taken inside a classroom",
        "a classroom with desks and chairs",
        "a lecture room in a university",
        "students sitting in a classroom",
        "a whiteboard at the front of a classroom",
    ],
    "kitchen": [
        "a photo of a kitchen",
        "a photo taken inside a kitchen",
        "a kitchen with cabinets and appliances",
        "a person cooking in a kitchen",
        "a kitchen countertop with utensils",
        "a stove and a sink in a kitchen",
    ],
    "hospital": [
        "a photo of a hospital ward",
        "a photo taken inside a hospital",
        "a hospital corridor with medical equipment",
        "a nurse or doctor beside a hospital bed",
        "a patient room in a hospital",
        "a medical clinic examination room",
    ],
    "cloth_store": [
        "a photo of a clothing store",
        "a photo taken inside a clothes shop",
        "racks of clothes hanging in a store",
        "shelves of folded clothing in a shop",
        "a fitting room in a clothing store",
        "a boutique with clothes on display",
    ],
    "museum": [
        "a photo of a museum",
        "a photo taken inside a museum gallery",
        "exhibits in glass display cases in a museum",
        "paintings on the walls of an art gallery",
        "a museum hall with statues and artifacts",
        "visitors looking at museum exhibits",
    ],
}

# Frames matching these better than any scene are reported as "uncertain"
# (face fills the frame, no scene content visible).
ABSTAIN_PROMPTS = [
    "a close-up photo of a person's face",
    "a selfie of a person",
    "a portrait of a person looking at the camera",
]

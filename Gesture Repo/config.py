"""
Central configuration for the gesture-recognition modality (v2).

Approach: MediaPipe Holistic pose+hand landmarks -> 185-dim per-frame
features -> 32-frame window -> small temporal network (BiGRU / TCN /
TinyTransformer) -> 8 gesture classes.

Full design spec (source of truth): docs/GESTURE_V2_DESIGN_AND_HPC_GUIDE.md
"""
import os

# ── Paths ────────────────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.abspath(__file__))

# Raw datasets (Jester / NTU RGB+D / custom clips) live outside the repo on
# the HPC. Point GESTURE_DATA_ROOT at them; defaults to a local data/raw.
DATA_ROOT = os.environ.get("GESTURE_DATA_ROOT", os.path.join(ROOT, "data", "raw"))
JESTER_DIR = os.path.join(DATA_ROOT, "jester")
NTU_DIR = os.path.join(DATA_ROOT, "ntu")
CUSTOM_DIR = os.path.join(DATA_ROOT, "custom")
LANDMARK_DIR = os.path.join(DATA_ROOT, "landmarks")  # extraction output (.npz)

DATA_DIR = os.path.join(ROOT, "data")
INDEX_DIR = os.path.join(DATA_DIR, "index")          # split CSVs (committable)
CHECKPOINT_DIR = os.path.join(ROOT, "checkpoints")
REPORT_DIR = os.path.join(ROOT, "reports")
OUTPUT_DIR = os.path.join(ROOT, "outputs")

# ── Classes ──────────────────────────────────────────────────────────────
GESTURE_LABELS = [
    "idle",           # 0 — no target gesture (real trained class, not a fallback)
    "wave",           # 1
    "point",          # 2
    "thumbs_up",      # 3
    "thumbs_down",    # 4
    "beckoning",      # 5
    "raise_hand",     # 6
    "both_hands_up",  # 7
]
NUM_CLASSES = len(GESTURE_LABELS)
LABEL_TO_ID = {name: i for i, name in enumerate(GESTURE_LABELS)}

# ── Feature / window spec (guide §3 — do not change without updating it) ─
WINDOW = 32            # frames per training sequence / inference window
POSE_FEATS = 33 * 3    # x, y, visibility per pose landmark
HAND_FEATS = 21 * 2    # x, y per hand landmark (wrist-relative)
FEATURE_DIM = POSE_FEATS + 2 * (HAND_FEATS + 1)  # 185 (+1 = presence flag)

# ── Default hyper-parameters (Stage 3 tuning refines these) ─────────────
BATCH_SIZE = 256
NUM_WORKERS = 4
LR = 1e-3
WEIGHT_DECAY = 1e-4
LABEL_SMOOTHING = 0.1
DROPOUT = 0.3
HIDDEN_SIZE = 128
EPOCHS = 100
PATIENCE = 15          # early stopping on val macro-F1
WARMUP_EPOCHS = 3
SEED = 42

# ── Inference engine ─────────────────────────────────────────────────────
# Training clips (~2-4 s) are resampled to WINDOW frames, so inference must
# cover a similar time span: keep ~2 s of frames and resample to WINDOW.
ENGINE_BUFFER_FRAMES = 64   # ≈ 2.1 s at 30 fps
EMA_ALPHA = 0.25            # softmax smoothing (same as motion modality)
CONF_THRESHOLD = 0.60       # below this the engine emits "idle"
DEBOUNCE_S = 0.30           # a new label must win for this long to be emitted

# ── Deployed model ───────────────────────────────────────────────────────
# Single place deciding what inference loads. compare_models.py / tuning may
# change the winner; update these two lines (and model_config.json) to ship it.
DEFAULT_MODEL = "TCN"
DEFAULT_CHECKPOINT = os.path.join(CHECKPOINT_DIR, "best_TCN.pth")
DEFAULT_MODEL_CONFIG = os.path.join(CHECKPOINT_DIR, "model_config.json")

# ── Display colors (BGR) for UI overlays ────────────────────────────────
GESTURE_COLORS = {
    "idle":          (160, 160, 160),  # grey
    "wave":          (0, 200, 255),    # amber
    "point":         (255, 210, 0),    # cyan-yellow
    "thumbs_up":     (80, 220, 60),    # green
    "thumbs_down":   (60, 60, 230),    # red
    "beckoning":     (255, 120, 0),    # blue-orange
    "raise_hand":    (200, 100, 255),  # pink-purple
    "both_hands_up": (0, 255, 220),    # lime
    "no_person":     (90, 90, 90),     # dark grey
}

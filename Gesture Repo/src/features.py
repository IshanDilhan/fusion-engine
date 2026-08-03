"""
Landmark -> feature-vector engineering (guide §3).

Raw landmarks (as stored by scripts/extract_landmarks.py, NaN where absent):
    pose        float32 [T, 33, 4]   x, y, z, visibility (image-normalized)
    left_hand   float32 [T, 21, 3]   x, y, z
    right_hand  float32 [T, 21, 3]

Per-frame feature vector (FEATURE_DIM = 185):
    [ pose 33*(x,y,vis) | left hand 21*(x,y) | L flag | right hand 21*(x,y) | R flag ]

Pose is translated to the mid-shoulder origin and scaled by shoulder width
(hips may be off-frame at close range — never depended on). Hands are
wrist-relative and scaled by the wrist<->middle-MCP distance, so hand blocks
encode shape only; hand *position* is carried by the pose wrist landmarks.
Presence flags are the fix for close-up (Jester) vs full-body (NTU/live)
framing and for MediaPipe hand dropouts at distance.
"""
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from config import FEATURE_DIM, HAND_FEATS, POSE_FEATS, WINDOW

# MediaPipe pose landmark indices
L_SHOULDER, R_SHOULDER = 11, 12
# MediaPipe hand landmark indices
WRIST, MIDDLE_MCP = 0, 9

# Pose left<->right landmark permutation for horizontal mirroring
_POSE_LR_PAIRS = [(1, 4), (2, 5), (3, 6), (7, 8), (9, 10), (11, 12), (13, 14),
                  (15, 16), (17, 18), (19, 20), (21, 22), (23, 24), (25, 26),
                  (27, 28), (29, 30), (31, 32)]
POSE_MIRROR_PERM = np.arange(33)
for _l, _r in _POSE_LR_PAIRS:
    POSE_MIRROR_PERM[_l], POSE_MIRROR_PERM[_r] = _r, _l

_EPS = 1e-3


# ── temporal sampling ────────────────────────────────────────────────────
def uniform_indices(n_frames, target=WINDOW):
    """Uniformly spread `target` frame indices over [0, n_frames)."""
    if n_frames <= 0:
        return np.zeros(target, dtype=np.int64)
    return np.round(np.linspace(0, n_frames - 1, target)).astype(np.int64)


def sample_window(n_frames, target=WINDOW, rng=None,
                  speed_range=(0.8, 1.2)):
    """
    Training-time temporal sampling: pick a random speed factor and a random
    start-crop, then uniformly sample `target` indices from the crop.
    Nearest-frame sampling (no interpolation) keeps presence flags binary.
    """
    if rng is None:
        return uniform_indices(n_frames, target)
    speed = rng.uniform(*speed_range)
    crop = int(round(n_frames / speed))
    crop = int(np.clip(crop, min(4, n_frames), n_frames))
    start = rng.integers(0, n_frames - crop + 1)
    return start + uniform_indices(crop, target)


# ── normalization ────────────────────────────────────────────────────────
def normalize_pose(pose):
    """[T,33,4] raw -> [T,33,3] (x, y, vis), mid-shoulder origin, shoulder-width scale."""
    xy = pose[:, :, :2].astype(np.float32)
    vis = pose[:, :, 3].astype(np.float32)
    absent = np.isnan(xy).any(axis=(1, 2))  # frames with no pose at all

    center = (xy[:, L_SHOULDER] + xy[:, R_SHOULDER]) / 2.0          # [T,2]
    scale = np.linalg.norm(xy[:, L_SHOULDER] - xy[:, R_SHOULDER], axis=1)
    scale = np.maximum(scale, _EPS)[:, None, None]

    out_xy = (xy - center[:, None, :]) / scale
    out = np.concatenate([out_xy, vis[:, :, None]], axis=2)
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[absent] = 0.0
    return out


def normalize_hand(hand):
    """[T,21,3] raw -> ([T,21,2] wrist-relative scaled, [T] presence flag)."""
    xy = hand[:, :, :2].astype(np.float32)
    present = ~np.isnan(xy).any(axis=(1, 2))

    wrist = xy[:, WRIST]                                            # [T,2]
    scale = np.linalg.norm(xy[:, MIDDLE_MCP] - wrist, axis=1)
    scale = np.maximum(scale, _EPS)[:, None, None]

    out = (xy - wrist[:, None, :]) / scale
    out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
    out[~present] = 0.0
    return out, present.astype(np.float32)


def build_features(pose, left_hand, right_hand):
    """Raw landmark arrays -> [T, FEATURE_DIM] feature matrix."""
    T = pose.shape[0]
    p = normalize_pose(pose).reshape(T, POSE_FEATS)
    lh, lflag = normalize_hand(left_hand)
    rh, rflag = normalize_hand(right_hand)
    feats = np.concatenate(
        [p, lh.reshape(T, HAND_FEATS), lflag[:, None],
         rh.reshape(T, HAND_FEATS), rflag[:, None]],
        axis=1,
    ).astype(np.float32)
    assert feats.shape[1] == FEATURE_DIM
    return feats


# ── augmentation (guide §6) ──────────────────────────────────────────────
def mirror_raw(pose, left_hand, right_hand):
    """Horizontal mirror on RAW landmarks: flip x, swap L/R body sides & hands."""
    pose_m = pose[:, POSE_MIRROR_PERM].copy()
    pose_m[:, :, 0] = 1.0 - pose_m[:, :, 0]
    lh_m = right_hand.copy()
    rh_m = left_hand.copy()
    lh_m[:, :, 0] = 1.0 - lh_m[:, :, 0]
    rh_m[:, :, 0] = 1.0 - rh_m[:, :, 0]
    return pose_m, lh_m, rh_m


def _rotate_scale_block(block_xy, cos_a, sin_a, scale):
    x = block_xy[..., 0].copy()
    y = block_xy[..., 1].copy()
    block_xy[..., 0] = scale * (cos_a * x - sin_a * y)
    block_xy[..., 1] = scale * (sin_a * x + cos_a * y)


def augment_features(feats, rng, max_rot_deg=10.0, max_scale=0.10,
                     jitter_sigma=0.01, hand_dropout_p=0.15):
    """
    In-place feature-space augmentation on [W, FEATURE_DIM]:
    small global rotation & scale, Gaussian jitter, and random hand dropout
    (zeroing a hand block + flag over a contiguous span — simulates MediaPipe
    dropouts at distance and teaches reliance on the presence mask).
    """
    W = feats.shape[0]
    ang = np.deg2rad(rng.uniform(-max_rot_deg, max_rot_deg))
    cos_a, sin_a = np.cos(ang), np.sin(ang)
    scale = 1.0 + rng.uniform(-max_scale, max_scale)

    pose = feats[:, :POSE_FEATS].reshape(W, 33, 3)
    _rotate_scale_block(pose[:, :, :2], cos_a, sin_a, scale)

    lh_start = POSE_FEATS
    rh_start = POSE_FEATS + HAND_FEATS + 1
    for start in (lh_start, rh_start):
        hand = feats[:, start:start + HAND_FEATS].reshape(W, 21, 2)
        _rotate_scale_block(hand, cos_a, sin_a, scale)

    feats += rng.normal(0.0, jitter_sigma, feats.shape).astype(np.float32)
    # jitter must not corrupt the binary presence flags
    feats[:, lh_start + HAND_FEATS] = np.round(np.clip(feats[:, lh_start + HAND_FEATS], 0, 1))
    feats[:, rh_start + HAND_FEATS] = np.round(np.clip(feats[:, rh_start + HAND_FEATS], 0, 1))

    for start in (lh_start, rh_start):
        if rng.random() < hand_dropout_p:
            span = rng.integers(W // 4, W // 2 + 1)
            t0 = rng.integers(0, W - span + 1)
            feats[t0:t0 + span, start:start + HAND_FEATS + 1] = 0.0
    return feats

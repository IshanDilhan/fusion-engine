"""
Confidence floors used by all four runners to compute NormalisedFrameCue.valid.
Values match the handover document's configs/schema.yaml plan (section 3),
except gesture -- see note below.
Stdlib-only — see schema.py for why.
"""

CONFIDENCE_FLOOR = {
    # Recalibrated via pipeline/calibrate_confidence_floor.py's ground-truth
    # sweep (scenarios.csv's authored emotion_v3, excluding emotion_masked
    # clips): clip-level accuracy is flat at ~0.669 for floor in [0.05, 0.30]
    # (was 0.611 at the old 0.50), then degrades steadily above that (0.318
    # at 0.80). Note the ceiling itself (0.669) is still fairly low -- a real
    # property of the emotion model, not fully fixed by recalibration, same
    # order of magnitude as gesture's ~0.66 ceiling. Context was swept too
    # and found already well-calibrated at 0.50 (flat 0.986 through 0.05-0.50)
    # -- left unchanged.
    "emotion": 0.25,
    # Recalibrated for the v2 GestureEngine (was 0.80, tuned for the old
    # per-frame keypoint classifier). Swept against scenarios.csv's authored
    # intended-gesture ground truth (clip-level majority-vote agreement):
    # accuracy is flat at ~0.66 for any floor in [0.05, 0.30], then degrades
    # steadily as the floor rises -- down to 0.39 at the old 0.80, where
    # "idle" specifically scored 0.0 (its confidence rarely clears 0.65-0.70
    # even when correct for a whole clip). 0.20 sits at the end of the flat
    # plateau, clear of pure noise (floor=0.0 alone scores 0.596) without yet
    # losing clips to the missing-fraction threshold.
    "gesture": 0.20,
    "motion": 0.50,
    "context": 0.50,
}

# Canonical gesture vocabulary (schema.yaml's gesture_classes), used by
# gesture_runner.py to map the native scenario-resolver strings.
GESTURE_SCENARIO_TO_CANONICAL = {
    "Wave": "wave",
    "Brief wave": "wave",
    "Arms waving": "wave",
    "Pointing": "point",
    "Thumbs up": "thumbs_up",
    "Thumbs down": "thumbs_down",
    "One hand raised": "raise_hand",
    "Arms up": "both_hands_up",
    "Beckoning": "beckoning",
    "None": "Unknown",
    # A hand tracked and confidently classified as "Open Palm"/"Close" but not
    # otherwise gesturing -- a real idle observation, distinct from "None"
    # (no confident hand-state at all). Both map to the same "Unknown" label
    # (see canonical_map.py) but are scored differently for valid/confidence --
    # see runners/gesture_runner.py's idle-vs-missing handling.
    "Idle": "Unknown",
}

"""
Configuration and vocabulary mappings for the Action Generator module.

This file contains all constants, mappings, hyperparameters, and default control
signals used by the lightweight multi-task neural policy module.
"""

from typing import Tuple, Dict

# -----------------------------------------------------------------------------
# Vocabularies
# -----------------------------------------------------------------------------

# Intents (F01-F10)
INTENTS = [
    "F01",  # Greeting / positive acknowledgment
    "F02",  # Emergency / danger
    "F03",  # Task assistance request
    "F04",  # Help request
    "F05",  # Engaged / busy - no interaction needed
    "F06",  # Requests passage / space
    "F07",  # Frustration / agitation
    "F08",  # Break / relief request
    "F09",  # Farewell
    "F10",  # Discouraged / giving up
]

# Motion States (6 classes from V3 dataset)
MOTIONS = [
    "sit",
    "stand",
    "walk",
    "run",
    "step_back",
    "lean_forward"
]

# Direction Values
DIRECTIONS = [
    "toward_robot",
    "away_from_robot",
    "toward_object",
    "toward_exit",
    "lateral",
    "stationary"
]

# Context Scenes (includes 'offline' as a special token for missing/unknown)
CONTEXTS = [
    "classroom",
    "kitchen",
    "offline"
]

# Robot Actions (A01-A15)
ACTIONS = [
    "A01",  # Positive acknowledgment; prompt/prepare next task
    "A02",  # Halt all motion + fire/smoke alert
    "A03",  # Halt all motion + medical alert
    "A04",  # Approach/follow to indicated spot; await instruction
    "A05",  # Offer task guidance / answer, supportive tone
    "A06",  # Hold position; do not interrupt; stay aware
    "A07",  # Acknowledge; suggest break/alternative activity
    "A08",  # Calm assistance, soft tone (de-escalation)
    "A09",  # Wave back; do not follow
    "A10",  # Acknowledge farewell; hold position
    "A11",  # Move aside promptly; clear path
    "A12",  # Encourage; suggest alternative approach
    "A13",  # Follow; offer to fetch/carry
    "A14",  # Halt risky action; check surroundings; notify supervisor
    "A15",  # Ask clarifying question
]

ACTION_DESCRIPTIONS = {
    "A01": "Positive acknowledgment; prompt/prepare next task",
    "A02": "Halt all motion + fire/smoke alert",
    "A03": "Halt all motion + medical alert",
    "A04": "Approach/follow to indicated spot; await instruction",
    "A05": "Offer task guidance / answer, supportive tone",
    "A06": "Hold position; do not interrupt; stay aware",
    "A07": "Acknowledge; suggest break/alternative activity",
    "A08": "Calm assistance, soft tone (de-escalation)",
    "A09": "Wave back; do not follow",
    "A10": "Acknowledge farewell; hold position",
    "A11": "Move aside promptly; clear path",
    "A12": "Encourage; suggest alternative approach",
    "A13": "Follow; offer to fetch/carry",
    "A14": "Halt risky action; check surroundings; notify supervisor",
    "A15": "Ask clarifying question",
}

# -----------------------------------------------------------------------------
# Default Control Signals per Action
# -----------------------------------------------------------------------------
# Format: (linear_velocity_m_s, angular_velocity_rad_s, comfort_distance_m)
DEFAULT_CONTROLS: Dict[str, Tuple[float, float, float]] = {
    "A01": (0.0, 0.0, 1.0),
    "A02": (0.0, 0.0, 2.0),
    "A03": (0.0, 0.0, 2.0),
    "A04": (0.3, 0.2, 0.8),
    "A05": (0.0, 0.1, 1.0),
    "A06": (0.0, 0.0, 1.5),
    "A07": (0.0, 0.0, 1.2),
    "A08": (0.0, 0.05, 1.5),
    "A09": (0.0, 0.0, 1.5),
    "A10": (0.0, 0.0, 1.5),
    "A11": (0.3, 0.5, 0.5),
    "A12": (0.0, 0.0, 1.0),
    "A13": (0.4, 0.2, 1.0),
    "A14": (0.0, 0.1, 1.8),
    "A15": (0.0, 0.0, 1.0),
}

# -----------------------------------------------------------------------------
# Hyperparameters
# -----------------------------------------------------------------------------
EMBEDDING_DIM = 16
HIDDEN_DIM = 128
DROPOUT = 0.20
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
LR_MIN = 1e-5
EPOCHS = 100
BATCH_SIZE = 32
MODALITY_DROPOUT_P = 0.15
FOCAL_LOSS_GAMMA = 2.0
CONTROL_LOSS_WEIGHT = 0.5
F02_SAFETY_THRESHOLD = 0.15

# -----------------------------------------------------------------------------
# Vocabulary Sizes
# -----------------------------------------------------------------------------
NUM_INTENTS = len(INTENTS)        # 10
NUM_MOTIONS = len(MOTIONS)        # 6
NUM_DIRECTIONS = len(DIRECTIONS)  # 6
NUM_CONTEXTS = len(CONTEXTS)      # 3
NUM_ACTIONS = len(ACTIONS)        # 15
NUM_CONTROL_OUTPUTS = 3           # v, omega, d

# -----------------------------------------------------------------------------
# Mappings (String <-> Index)
# -----------------------------------------------------------------------------
_INTENT_TO_IDX = {intent: idx for idx, intent in enumerate(INTENTS)}
_IDX_TO_INTENT = {idx: intent for idx, intent in enumerate(INTENTS)}

_MOTION_TO_IDX = {motion: idx for idx, motion in enumerate(MOTIONS)}
_DIRECTION_TO_IDX = {direction: idx for idx, direction in enumerate(DIRECTIONS)}
_CONTEXT_TO_IDX = {context: idx for idx, context in enumerate(CONTEXTS)}

_ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTIONS)}
_IDX_TO_ACTION = {idx: action for idx, action in enumerate(ACTIONS)}

# -----------------------------------------------------------------------------
# Helper Functions
# -----------------------------------------------------------------------------

def intent_to_idx(code: str) -> int:
    """Get the integer index for an intent code (e.g., 'F01')."""
    return _INTENT_TO_IDX[code]

def idx_to_intent(idx: int) -> str:
    """Get the intent code (e.g., 'F01') for an integer index."""
    return _IDX_TO_INTENT[idx]

def motion_to_idx(state: str) -> int:
    """Get the integer index for a motion state."""
    return _MOTION_TO_IDX[state]

def direction_to_idx(dir: str) -> int:
    """Get the integer index for a direction value."""
    return _DIRECTION_TO_IDX[dir]

def context_to_idx(scene: str) -> int:
    """Get the integer index for a context scene. 'offline' is valid."""
    return _CONTEXT_TO_IDX[scene]

def action_to_idx(code: str) -> int:
    """Get the integer index for a robot action code (e.g., 'A01')."""
    return _ACTION_TO_IDX[code]

def idx_to_action(idx: int) -> str:
    """Get the robot action code (e.g., 'A01') for an integer index."""
    return _IDX_TO_ACTION[idx]

def get_action_description(code: str) -> str:
    """Get the human-readable description for a robot action code."""
    return ACTION_DESCRIPTIONS[code]

def get_default_control(code: str) -> Tuple[float, float, float]:
    """Get the default control signals (v, omega, d) for an action code."""
    return DEFAULT_CONTROLS[code]

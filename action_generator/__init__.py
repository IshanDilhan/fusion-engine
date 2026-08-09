"""
Action Generator Package.

Multimodal Neural Policy Module for HRI.
Predicts Robot Actions (A01-A15) and continuous Motion Controls [v, w, d]
from fused Intent (F01-F09), Motion State, Direction, and Context.
"""

from .config import INTENTS, ACTIONS, MOTIONS, DIRECTIONS, CONTEXTS
from .model import MultimodalActionGenerator
from .safety_override import apply_safety_override
from .inference import ActionInference

__all__ = [
    "MultimodalActionGenerator",
    "apply_safety_override",
    "ActionInference",
    "INTENTS",
    "ACTIONS",
    "MOTIONS",
    "DIRECTIONS",
    "CONTEXTS",
]

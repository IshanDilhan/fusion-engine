"""The context model's output contract.

`ContextState` is the single structured object the context model emits for the
downstream policy / fusion module. It combines:
  - the environment label from the zero-shot CLIP scene classifier, and
  - the structured situation understanding from a small VLM (SmolVLM2):
    what the person is doing, how many people are present, what the main
    person is focused on, the key objects, and a one-sentence summary.
"""

from dataclasses import dataclass, field, asdict
import time


@dataclass
class VLMContext:
    """Structured output parsed from one VLM analysis of a frame."""
    people: int = 0                     # number of people visible
    activity: str = "unknown"           # what the main person is doing
    attention: str = "unknown"          # what the main person is focused on
    objects: list = field(default_factory=list)  # up to ~5 key objects
    summary: str = ""                   # one-sentence situation description
    raw_text: str = ""                  # unparsed VLM output (debugging)
    latency_ms: float = 0.0             # how long the VLM call took
    frame_timestamp: float = 0.0        # when the analysed frame was captured


@dataclass
class ContextState:
    scene: str = "unknown"              # from the CLIP scene classifier
    scene_confidence: float = 0.0
    vlm: VLMContext = field(default_factory=VLMContext)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self):
        return asdict(self)

"""Small-VLM context analyzer (SmolVLM2-500M).

Asks the VLM for a plain scene description — its genuine strength — rather
than a strict multi-field format: tested against real classroom/kitchen
frames, SmolVLM2-500M gives accurate, specific free-form captions ("a man
sitting at a desk with an open book, writing") but does not reliably follow a
rigid PEOPLE:/ACTIVITY:/... template even with a worked example in the prompt
(a known limitation of sub-1B VLMs). So the caption becomes `VLMContext.summary`
(the reliable signal), and `caption_extract.py` pulls approximate structured
fields out of it locally with regex/keyword rules — no extra model calls, no
added Jetson latency.
"""

import time
from pathlib import Path
import sys

import cv2
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modalities.context.config import VLM_MODEL, VLM_MAX_NEW_TOKENS
from modalities.context.src.caption_extract import extract_fields
from modalities.context.src.context_state import VLMContext

_PROMPT = (
    "Describe this scene in one or two sentences: how many people are "
    "present, what the main person is doing, and what objects are visible."
)


class VLMContextAnalyzer:
    def __init__(self, model_id=VLM_MODEL, device=None, max_new_tokens=VLM_MAX_NEW_TOKENS):
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.max_new_tokens = max_new_tokens

        self.processor = AutoProcessor.from_pretrained(model_id)
        self.model = AutoModelForImageTextToText.from_pretrained(
            model_id, dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    @torch.no_grad()
    def analyze(self, frame_bgr, frame_timestamp=None):
        """Run one VLM analysis of a BGR frame. Returns a VLMContext."""
        t0 = time.time()
        pil = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))

        messages = [{
            "role": "user",
            "content": [
                {"type": "image", "image": pil},
                {"type": "text", "text": _PROMPT},
            ],
        }]
        inputs = self.processor.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_dict=True, return_tensors="pt",
        ).to(self.device)
        if "pixel_values" in inputs:
            inputs["pixel_values"] = inputs["pixel_values"].to(self.dtype)

        generated = self.model.generate(
            **inputs, max_new_tokens=self.max_new_tokens, do_sample=False,
            repetition_penalty=1.3, no_repeat_ngram_size=3)
        caption = self.processor.batch_decode(
            generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0].strip()

        fields = extract_fields(caption)
        ctx = VLMContext(
            people=fields["people"],
            activity=fields["activity"],
            attention=fields["attention"],
            objects=fields["objects"],
            summary=caption,
            raw_text=caption,
        )
        ctx.latency_ms = (time.time() - t0) * 1000.0
        ctx.frame_timestamp = frame_timestamp if frame_timestamp is not None else t0
        return ctx

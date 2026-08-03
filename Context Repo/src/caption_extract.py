"""Best-effort structured-field extraction from a free-form VLM caption.

SmolVLM2-500M reliably produces good natural-language scene descriptions but,
like most sub-1B VLMs, does not reliably follow a strict multi-field output
format even with a worked example in the prompt (tested — see
reports in scene_classification/reports/zero_shot for the analogous scene
finding). So the VLM is asked for a plain description (its real strength), and
this module pulls approximate structured fields out of that text with
lightweight regex/keyword rules — no extra model calls, negligible latency,
Jetson-friendly.

These are heuristics, not NLP-grade extraction: `summary` (the raw caption) is
the reliable signal; `people` / `activity` / `attention` / `objects` are a
best-effort approximation for callers that want quick structured access.
"""

import re

_WORD_NUMBERS = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
}
_SINGULAR_PERSON = re.compile(
    r"\b(a|the|one)\s+(man|woman|person|boy|girl|student|chef|nurse|doctor|kid|child)\b",
    re.IGNORECASE,
)
_COUNTED_PEOPLE = re.compile(
    r"\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten)\s+"
    r"(people|persons?|men|women|kids?|boys?|girls?|students?|children)\b",
    re.IGNORECASE,
)
_PLURAL_PEOPLE = re.compile(
    r"\b(people|men|women|kids|boys|girls|students|children)\b", re.IGNORECASE
)
_PRONOUN_PERSON = re.compile(r"\b(he|she|his|her)\b", re.IGNORECASE)

# Curated activity verb stems. Matched against any inflection (cook / cooks /
# cooked / cooking) and always reported in the -ing form. Only these stems are
# reported, avoiding unrelated gerunds/participles like "checkered" or
# "matching".
_ACTIVITY_VERBS = [
    "cook", "clean", "writ", "read", "typ", "draw", "point", "hold", "sit",
    "stand", "walk", "talk", "eat", "drink", "work", "study", "wash",
    "chop", "stir", "mix", "pour", "cut", "teach", "listen", "present",
    "discuss", "browse", "search", "pack", "serv", "prepar", "explain",
]
_ACTIVITY_DISPLAY = {  # stem -> canonical -ing display form
    "writ": "writing", "stud": "studying", "serv": "serving",
    "prepar": "preparing",
}
_ACTIVITY_RE = re.compile(
    r"\b(" + "|".join(_ACTIVITY_VERBS) + r")(?:ing|s|ed|e)?\b", re.IGNORECASE
)

# What the person is engaged with: the nearest known object mentioned shortly
# after one of these engagement verbs (anchoring to the object vocabulary,
# rather than grabbing a raw word-window, avoids picking up stray prepositions
# like "under his" or unrelated words like "up two fingers").
_ATTENTION_TRIGGER_RE = re.compile(
    r"(?:holding|looking at|using|pointing (?:to|at)|working on|"
    r"writing (?:on|in)|reading)\b",
    re.IGNORECASE,
)

# Curated object vocabulary spanning the deployed environments (classroom,
# kitchen) plus common general items a small VLM tends to name. Maps each
# surface form to a canonical display name (handles synonyms/variants).
_OBJECT_CANON = {
    "book": "book", "books": "book", "pen": "pen", "pens": "pen",
    "pencil": "pencil", "paper": "paper", "papers": "paper", "desk": "desk",
    "chair": "chair", "chairs": "chair", "table": "table",
    "tablecloth": "tablecloth", "whiteboard": "whiteboard", "laptop": "laptop",
    "computer": "computer", "bag": "bag", "glasses": "glasses",
    "notebook": "notebook", "phone": "phone", "stove": "stove",
    "stovetop": "stove", "pan": "pan", "pot": "pot", "plate": "plate",
    "plates": "plate", "cup": "cup", "cups": "cup", "bowl": "bowl",
    "bowls": "bowl", "knife": "knife", "spoon": "spoon", "fork": "fork",
    "sink": "sink", "fridge": "fridge", "refrigerator": "fridge",
    "oven": "oven", "kettle": "kettle", "cabinet": "cabinet",
    "cabinets": "cabinet", "counter": "counter", "countertop": "counter",
    "fan": "fan", "window": "window", "door": "door", "bed": "bed",
    "couch": "couch", "sofa": "couch", "tv": "tv", "television": "tv",
    "camera": "camera", "microphone": "microphone", "board": "board",
    "marker": "marker", "bottle": "bottle", "glass": "glass", "food": "food",
    "utensils": "utensils",
}
_OBJECT_RE = re.compile(
    r"\b(" + "|".join(sorted(_OBJECT_CANON, key=len, reverse=True)) + r")\b",
    re.IGNORECASE,
)


def extract_fields(caption):
    """Parse a free-form VLM caption into approximate structured fields.

    Returns a dict: people (int), activity (str), attention (str), objects
    (list[str], up to 5, in order of first appearance, de-duplicated).
    """
    if not caption:
        return {"people": 0, "activity": "unknown", "attention": "unknown", "objects": []}

    people = _count_people(caption)

    act_match = _ACTIVITY_RE.search(caption)
    if act_match:
        stem = act_match.group(1).lower()
        activity = _ACTIVITY_DISPLAY.get(stem, stem + "ing")
    else:
        activity = "unknown"

    attention = _find_attention(caption)

    seen, objects = set(), []
    for m in _OBJECT_RE.finditer(caption):
        word = _OBJECT_CANON[m.group(1).lower()]
        if word not in seen:
            seen.add(word)
            objects.append(word)
        if len(objects) >= 5:
            break

    return {"people": people, "activity": activity, "attention": attention, "objects": objects}


def _find_attention(caption):
    """First known object mentioned shortly after an engagement verb."""
    trigger = _ATTENTION_TRIGGER_RE.search(caption)
    if not trigger:
        return "unknown"
    window = caption[trigger.end(): trigger.end() + 40]
    obj = _OBJECT_RE.search(window)
    return _OBJECT_CANON[obj.group(1).lower()] if obj else "unknown"


def _count_people(caption):
    m = _COUNTED_PEOPLE.search(caption)
    if m:
        token = m.group(1).lower()
        return int(token) if token.isdigit() else _WORD_NUMBERS.get(token, 1)
    if _SINGULAR_PERSON.search(caption):
        return 1
    if _PLURAL_PEOPLE.search(caption):
        return 2  # plural mentioned without a count — approximate
    if _PRONOUN_PERSON.search(caption):
        return 1
    return 0

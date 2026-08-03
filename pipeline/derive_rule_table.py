"""
Derives fusion/rule_based.py's IF-THEN rule table directly from scenarios.csv,
instead of hand-transcribing it from the scenario table by inspection (v1's
approach -- see the historical note in fusion/rule_based.py's own docstring).

v2.0.0's scenarios.csv (62 rows, keyed by v3_row) already gives the
(context, emotion_v3, gesture_v3, motion_v3) -> intent mapping directly and
machine-readably, unlike v1's ~22-28 row table which had to be read by eye.

Groups scenarios.csv by its normalized (context, emotion, gesture, motion)
4-tuple (via canonical_map.py, the same normalization agreement_report.py
already uses) and asserts each group maps to exactly one intent -- loud
failure if the dataset is ever revised and a genuine ambiguity reappears,
rather than a silently-stale rule table. Run this after any scenarios.csv
change, before hand-updating fusion/rule_based.py's predict_intent().

Stdlib-only.
"""
import csv
import os
import sys
from collections import defaultdict

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(REPO_ROOT, "pipeline"))
from canonical_map import map_intended  # noqa: E402
from dataset_config import DATASET_ROOT  # noqa: E402

SCENARIOS_CSV = os.path.join(DATASET_ROOT, "annotations", "scenarios.csv")


def normalized_rows():
    with open(SCENARIOS_CSV, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    out = []
    for r in rows:
        key = (
            map_intended("context", r["context"]),
            map_intended("emotion", r["emotion_v3"]),
            map_intended("gesture", r["gesture_v3"]),
            map_intended("motion", r["motion_v3"]),
        )
        out.append((key, r["intent"], r["v3_row"]))
    return out


def build_table():
    """Returns {(context, emotion, gesture, motion): intent}, raising if any
    combination maps to more than one intent."""
    groups = defaultdict(set)
    rows_by_key = defaultdict(list)
    for key, intent, v3_row in normalized_rows():
        groups[key].add(intent)
        rows_by_key[key].append(v3_row)

    ambiguous = {k: v for k, v in groups.items() if len(v) > 1}
    if ambiguous:
        lines = [f"  {k} -> {v} (v3_rows {rows_by_key[k]})" for k, v in ambiguous.items()]
        raise AssertionError(
            "scenarios.csv has genuinely ambiguous cue combinations -- "
            "predict_intent() cannot resolve these from the 4 measured cues "
            "alone:\n" + "\n".join(lines)
        )
    return {k: next(iter(v)) for k, v in groups.items()}


def _fmt(v):
    return "[missing]" if v is None else v


def main():
    table = build_table()
    print(f"[derive_rule_table] {len(table)} distinct (context, emotion, gesture, motion) "
          f"combinations, 0 ambiguous\n")

    by_gesture = defaultdict(list)
    for (context, emotion, gesture, motion), intent in table.items():
        by_gesture[_fmt(gesture)].append((_fmt(context), _fmt(emotion), _fmt(motion), intent))

    for gesture in sorted(by_gesture):
        print(f"gesture={gesture}:")
        for context, emotion, motion, intent in sorted(by_gesture[gesture]):
            print(f"  context={context:10s} emotion={emotion:10s} motion={motion:15s} -> {intent}")
        print()


if __name__ == "__main__":
    main()

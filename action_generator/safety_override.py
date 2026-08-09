"""
safety_override.py

Internal Physical Safety Gate for the Action Generator module.

Implements a 2-tier priority safety policy:
  Priority 1 (Emergency Bypass): Emergency intents (F02) or high hazard probabilities
             trigger immediate emergency halt/alert (A02 or A14), v=0.0 m/s, d=2.0 m.
  Priority 2 (Dynamic Proximity Gate): When a human approaches rapidly (< 1.0 m, speed > 0.5 m/s),
             the predicted Action Code is PRESERVED (e.g. A05) to protect model accuracy,
             while motor controls are clamped to v=-0.2 m/s (reverse yield step) and d=1.5 m.
"""

from typing import Dict, Any, Tuple, Optional


def apply_safety_override(
    intent: str,
    action_probs: Dict[str, float],
    context: str,
    direction: str = "stationary",
    velocity: float = 0.0,
    current_distance: float = 1.5,
    pred_v: float = 0.0,
    pred_omega: float = 0.0,
    pred_d: float = 1.0,
) -> Dict[str, Any]:
    """
    2-Tier Internal Safety Gate.
    
    Returns a dict with:
        - override_active (bool): True if any safety condition triggered
        - forced_action (Optional[str]): Action Code string if emergency forced, else None (keep predicted)
        - final_v (float): Safe linear velocity target (m/s)
        - final_omega (float): Safe angular velocity target (rad/s)
        - final_d (float): Safe comfort distance target (m)
        - safety_reason (str): Human-readable safety status explanation
    """
    
    # -------------------------------------------------------------------------
    # PRIORITY 1: TRUE EMERGENCY BYPASS (F02 / Hazard Probabilities >= 0.15)
    # Emergency protocol takes priority and overrides standard distance rules.
    # -------------------------------------------------------------------------
    is_emergency = (
        intent == "F02" or
        action_probs.get("A02", 0.0) >= 0.15 or
        action_probs.get("A03", 0.0) >= 0.15 or
        action_probs.get("A14", 0.0) >= 0.15
    )

    if is_emergency:
        forced_action = "A02" if context == "kitchen" else "A14"
        return {
            "override_active": True,
            "forced_action": forced_action,
            "final_v": 0.0,
            "final_omega": 0.0,
            "final_d": 2.0,
            "safety_reason": f"Emergency hazard bypass triggered: forced action {forced_action}, halt motion (0.0 m/s), max clearance 2.0m"
        }

    # -------------------------------------------------------------------------
    # PRIORITY 2: DYNAMIC PROXIMITY SAFETY GATE (Proximity & Speed Yielding)
    # Action Code is NOT changed (e.g. A05 stays A05 to preserve 90.48% accuracy).
    # Linear velocity is clamped to -0.2 m/s (reverse yield) when distance < 1.0m.
    # -------------------------------------------------------------------------
    is_proximity_hazard = (
        direction == "toward_robot" and
        velocity > 0.5 and
        current_distance < 1.0
    )

    if is_proximity_hazard:
        return {
            "override_active": True,
            "forced_action": None,         # PRESERVED PREDICTED ACTION (No classification error!)
            "final_v": -0.2,               # Reverse yielding speed (steps backward to create space)
            "final_omega": 0.0,            # Zero turning rate during yield
            "final_d": max(pred_d, 1.5),    # Enforce minimum 1.5m safety clearance
            "safety_reason": "Rapid approach proximity risk (<1.0m): action preserved, linear velocity set to -0.2 m/s (reverse yield step)"
        }

    # -------------------------------------------------------------------------
    # STANDARD PASS-THROUGH (No Safety Violations)
    # -------------------------------------------------------------------------
    return {
        "override_active": False,
        "forced_action": None,
        "final_v": pred_v,
        "final_omega": pred_omega,
        "final_d": pred_d,
        "safety_reason": "Standard nominal operation: no safety rule violations"
    }

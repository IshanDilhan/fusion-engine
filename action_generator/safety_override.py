"""
safety_override.py

Standalone safety override module for Action Generator.
"""

from typing import Dict, Any

def apply_safety_override(intent: str, action_probs: Dict[str, float], context: str) -> Dict[str, Any]:
    """Hard-coded post-prediction safety override.
    
    This is NEVER learned — always enforced at runtime.
    Emergency intents (F02) and high-probability emergency actions
    trigger immediate halt with maximum safety distance.
    
    Returns:
        dict with keys: override_active, forced_action, forced_velocity, forced_comfort_distance
    """
    override_active = False
    
    # 1. If intent == 'F02': override active
    if intent == 'F02':
        override_active = True
    # 2. If action_probs.get('A02', 0) >= 0.15: override active
    elif action_probs.get('A02', 0.0) >= 0.15:
        override_active = True
    # 3. If action_probs.get('A03', 0) >= 0.15: override active
    elif action_probs.get('A03', 0.0) >= 0.15:
        override_active = True
    # 4. If action_probs.get('A14', 0) >= 0.15: override active
    elif action_probs.get('A14', 0.0) >= 0.15:
        override_active = True
        
    forced_action = None
    forced_velocity = None
    forced_comfort_distance = None
    
    # 5. When override active:
    if override_active:
        if context == 'kitchen':
            forced_action = 'A02'
        elif context == 'classroom':
            forced_action = 'A14'
        else:
            forced_action = 'A02'  # err on side of caution
            
        forced_velocity = 0.0
        forced_comfort_distance = 2.0
        
    return {
        'override_active': override_active,
        'forced_action': forced_action,
        'forced_velocity': forced_velocity,
        'forced_comfort_distance': forced_comfort_distance
    }

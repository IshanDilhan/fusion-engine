"""
inference.py

Runtime inference wrapper for Action Generator.
"""

import os
import torch
import numpy as np
from typing import Dict, Optional
from dataclasses import dataclass

from safety_override import apply_safety_override

# ─── Result dataclass ─────────────────────────────────────────────────────────

@dataclass
class ActionResult:
    """Result from the Action Generator inference."""
    action: str              # e.g. 'A14'
    action_description: str  # e.g. 'Halt risky action; check surroundings; notify supervisor'
    confidence: float        # top-1 softmax probability
    probabilities: Dict[str, float]  # full A01-A15 probability distribution
    linear_velocity_m_s: float
    angular_velocity_rad_s: float  
    comfort_distance_m: float
    safety_override_active: bool
    
    def to_dict(self) -> dict:
        return {
            "action": self.action,
            "action_description": self.action_description,
            "confidence": round(self.confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in self.probabilities.items()},
            "linear_velocity_m_s": round(self.linear_velocity_m_s, 3),
            "angular_velocity_rad_s": round(self.angular_velocity_rad_s, 3),
            "comfort_distance_m": round(self.comfort_distance_m, 3),
            "safety_override_active": self.safety_override_active,
        }

# ─── Main inference class ─────────────────────────────────────────────────────

class ActionInference:
    """Stateless inference engine for the MultimodalActionGenerator.
    
    Usage:
        engine = ActionInference('checkpoints/best_action_generator.pt')
        result = engine.predict(
            intent='F02',
            intent_confidence=0.92,
            motion_state='step_back',
            direction='away_from_robot',
            velocity=0.8,
            context='classroom'
        )
        print(result.action)  # 'A14'
    """

    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cpu'
    ):
        # ── Device ────────────────────────────────────────────────────────────
        self.device = torch.device(device)
        
        # ── Load model (Placeholder logic for loading model) ──────────────────
        # checkpoint = torch.load(checkpoint_path, map_location=self.device)
        # self.model = MultimodalActionGenerator(...)
        # self.model.load_state_dict(checkpoint["model_state_dict"])
        # self.model.eval()
        
        print(f"ActionInference ready | device={self.device} | checkpoint={checkpoint_path}")

    # ── Public API ────────────────────────────────────────────────────────────

    def predict(
        self,
        intent: str,
        intent_confidence: float,
        motion_state: str,
        direction: str,
        velocity: float,
        context: str
    ) -> ActionResult:
        """
        Run inference to predict the action and motion parameters.
        """
        # Converts string inputs to indices (Placeholder)
        # intent_idx = INTENT_VOCAB[intent]
        # motion_idx = MOTION_VOCAB[motion_state]
        
        # Constructs velocity features [speed, 0.0 (accel placeholder), is_moving_flag]
        is_moving_flag = 1.0 if velocity > 0.0 else 0.0
        vel_features = torch.tensor(
            [velocity, 0.0, is_moving_flag],
            dtype=torch.float32
        ).to(self.device)
        
        # Runs forward pass under torch.no_grad()
        # with torch.no_grad():
        #     logits = self.model(...)
        #     probs = F.softmax(logits, dim=-1)
        
        # Dummy prediction variables
        action_probs = {f"A{i:02d}": 0.0 for i in range(1, 16)}
        action_probs['A14'] = 0.20
        action_probs['A01'] = 0.80
        
        predicted_action = 'A01'
        confidence = 0.80
        action_desc = "Default safe action"
        lin_vel = 0.5
        ang_vel = 0.0
        comf_dist = 1.0
        
        # Applies safety override
        override_res = apply_safety_override(intent, action_probs, context)
        
        override_active = override_res['override_active']
        if override_active:
            predicted_action = override_res['forced_action']
            lin_vel = override_res['forced_velocity']
            comf_dist = override_res['forced_comfort_distance']
            
            # Simplified description update
            if predicted_action == 'A14':
                action_desc = "Halt risky action; check surroundings; notify supervisor"
            elif predicted_action == 'A02':
                action_desc = "Fire/smoke alert"

        return ActionResult(
            action=predicted_action,
            action_description=action_desc,
            confidence=confidence,
            probabilities=action_probs,
            linear_velocity_m_s=lin_vel,
            angular_velocity_rad_s=ang_vel,
            comfort_distance_m=comf_dist,
            safety_override_active=override_active
        )
        
    def reset(self):
        """No-op for API compatibility with streaming interfaces."""
        pass

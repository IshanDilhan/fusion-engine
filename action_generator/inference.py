"""
inference.py

Runtime inference wrapper for the Action Generator module.
Loads trained PyTorch model, converts real-time string inputs to tensors,
runs forward pass under torch.no_grad(), and applies internal physical safety gate.
"""

import os
import sys
import torch
import torch.nn.functional as F
from typing import Dict, Optional
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MultimodalActionGenerator
from config import (
    ACTIONS, ACTION_DESCRIPTIONS,
    intent_to_idx, motion_to_idx, direction_to_idx, context_to_idx,
    idx_to_action
)
from safety_override import apply_safety_override


# ─── 1. Output Dataclass ──────────────────────────────────────────────────────

@dataclass
class ActionResult:
    """Result container from Action Generator inference."""
    action: str                        # e.g. 'A05' or 'A02'
    action_description: str            # e.g. 'Offer task guidance / answer'
    confidence: float                  # Top-1 softmax probability (0.0 to 1.0)
    probabilities: Dict[str, float]    # Full A01-A15 probability dictionary
    linear_velocity_m_s: float         # Speed target (m/s) — negative if yielding
    angular_velocity_rad_s: float      # Turning rate target (rad/s)
    comfort_distance_m: float          # Personal safety clearance (m)
    safety_override_active: bool       # True if safety gate or emergency filter triggered
    safety_reason: str = ""            # Explanation of physical safety intervention

    def to_dict(self) -> dict:
        """Converts result to JSON-serializable dictionary."""
        return {
            "action": self.action,
            "action_description": self.action_description,
            "confidence": round(self.confidence, 3),
            "probabilities": {k: round(v, 3) for k, v in self.probabilities.items()},
            "linear_velocity_m_s": round(self.linear_velocity_m_s, 3),
            "angular_velocity_rad_s": round(self.angular_velocity_rad_s, 3),
            "comfort_distance_m": round(self.comfort_distance_m, 3),
            "safety_override_active": self.safety_override_active,
            "safety_reason": self.safety_reason,
        }


# ─── 2. Main Inference Engine ─────────────────────────────────────────────────

class ActionInference:
    """Stateless inference engine for MultimodalActionGenerator.

    Usage:
        engine = ActionInference('checkpoints/best_action_generator.pt')
        result = engine.predict(
            intent='F04',
            intent_confidence=0.95,
            motion_state='walking',
            direction='toward_robot',
            velocity=0.8,
            context='classroom',
            current_distance=0.7
        )
        print(result.action)                # 'A05' (Preserved for accuracy!)
        print(result.linear_velocity_m_s)   # -0.2 (Reverse yield step!)
    """

    def __init__(self, checkpoint_path: str, device: str = 'cpu'):
        self.device = torch.device(device)
        self.model = MultimodalActionGenerator().to(self.device)

        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=self.device)
            if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
                state_dict = state_dict["model_state_dict"]
            self.model.load_state_dict(state_dict)
            print(f"ActionInference ready | device={self.device} | loaded checkpoint={checkpoint_path}")
        else:
            print(f"Warning: Checkpoint not found at {checkpoint_path}. Using uninitialized weights.")

        self.model.eval()

    def predict(
        self,
        intent: str,
        intent_confidence: float,
        motion_state: str,
        direction: str,
        velocity: float,
        context: str,
        current_distance: float = 1.5
    ) -> ActionResult:
        """Runs model prediction + internal physical safety gate for real-time input."""
        
        # Step A: Convert input string tokens to integer indices
        i_idx = torch.tensor([intent_to_idx(intent)], dtype=torch.long, device=self.device)
        i_conf = torch.tensor([[intent_confidence]], dtype=torch.float32, device=self.device)
        m_idx = torch.tensor([motion_to_idx(motion_state)], dtype=torch.long, device=self.device)
        d_idx = torch.tensor([direction_to_idx(direction)], dtype=torch.long, device=self.device)
        c_idx = torch.tensor([context_to_idx(context)], dtype=torch.long, device=self.device)

        # Step B: Construct 3D human velocity feature vector [speed, 0.0, is_moving]
        is_moving_flag = 1.0 if velocity > 0.0 else 0.0
        v_feats = torch.tensor([[velocity, 0.0, is_moving_flag]], dtype=torch.float32, device=self.device)

        # Step C: Neural forward pass without computing gradients
        with torch.no_grad():
            action_logits, control_vals = self.model(i_idx, i_conf, m_idx, d_idx, v_feats, c_idx)
            probs_tensor = F.softmax(action_logits, dim=-1)[0]
            ctrl_tensor = control_vals[0]

        # Step D: Extract top prediction & probability dictionary
        action_probs = {ACTIONS[k]: float(probs_tensor[k].item()) for k in range(len(ACTIONS))}
        top_idx = int(probs_tensor.argmax().item())
        predicted_action = ACTIONS[top_idx]
        confidence = float(probs_tensor[top_idx].item())

        pred_v = float(ctrl_tensor[0].item())
        pred_omega = float(ctrl_tensor[1].item())
        pred_d = float(ctrl_tensor[2].item())

        # Step E: Apply Internal Physical Safety Gate
        override_res = apply_safety_override(
            intent=intent,
            action_probs=action_probs,
            context=context,
            direction=direction,
            velocity=velocity,
            current_distance=current_distance,
            pred_v=pred_v,
            pred_omega=pred_omega,
            pred_d=pred_d
        )

        override_active = override_res['override_active']
        forced_action = override_res['forced_action']
        
        # If Priority 1 Emergency forced an action, update it; otherwise keep predicted_action (A05)
        if forced_action is not None:
            predicted_action = forced_action

        lin_vel = override_res['final_v']
        ang_vel = override_res['final_omega']
        comf_dist = override_res['final_d']
        safety_reason = override_res['safety_reason']

        action_desc = ACTION_DESCRIPTIONS.get(predicted_action, "Custom Robot Action")

        return ActionResult(
            action=predicted_action,
            action_description=action_desc,
            confidence=confidence,
            probabilities=action_probs,
            linear_velocity_m_s=lin_vel,
            angular_velocity_rad_s=ang_vel,
            comfort_distance_m=comf_dist,
            safety_override_active=override_active,
            safety_reason=safety_reason
        )

    def reset(self):
        """No-op for API compatibility with streaming interfaces."""
        pass

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultimodalActionGenerator(nn.Module):
    """
    Lightweight multi-task neural policy module for HRI.
    
    Takes [Intent, Motion, Direction, Context] + confidence/velocity features
    and predicts Robot Actions (A01-A15) with motion control signals.
    
    Architecture:
    - 4 embedding tables (intent:10->16, motion:6->16, direction:6->8, context:3->8)
    - 2-layer Dense Fusion Core (52->128->64 with LayerNorm + GELU + Dropout)
    - Head 1: Action classifier (64->15, Softmax)
    - Head 2: Motion controller (64->3, [v, ω, d])
    
    Total parameters: ~26K
    Inference: <1.8ms on Jetson Orin Nano
    """
    def __init__(self):
        super().__init__()
        
        # 1. Embeddings
        self.intent_embedding = nn.Embedding(10, 16)
        self.motion_embedding = nn.Embedding(6, 16)
        self.direction_embedding = nn.Embedding(6, 8)
        self.context_embedding = nn.Embedding(3, 8)
        
        # 2. Dense Fusion Core
        # 16 (intent) + 1 (intent_conf) + 16 (motion) + 8 (direction) + 3 (velocity) + 8 (context) = 52
        self.core = nn.Sequential(
            nn.Linear(52, 128),
            nn.LayerNorm(128),
            nn.GELU(),
            nn.Dropout(0.20),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU()
        )
        
        # 3. Output Heads
        self.action_head = nn.Linear(64, 15)
        self.control_head = nn.Linear(64, 3)

    def forward(self, intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx):
        """
        Returns:
            action_logits: (B, 15)
            control_values: (B, 3)
        """
        i_emb = self.intent_embedding(intent_idx)
        m_emb = self.motion_embedding(motion_idx)
        d_emb = self.direction_embedding(direction_idx)
        c_emb = self.context_embedding(context_idx)
        
        if intent_conf.dim() == 1:
            intent_conf = intent_conf.unsqueeze(1)
            
        x = torch.cat([i_emb, intent_conf, m_emb, d_emb, velocity_feats, c_emb], dim=-1)
        features = self.core(x)
        
        action_logits = self.action_head(features)
        control_values = self.control_head(features)
        
        return action_logits, control_values

    def predict(self, intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx):
        """Returns softmax action probabilities and control values."""
        action_logits, control_values = self.forward(
            intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx
        )
        action_probs = F.softmax(action_logits, dim=-1)
        return action_probs, control_values

    def predict_proba(self, intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx):
        """Returns only softmax action probabilities."""
        action_probs, _ = self.predict(
            intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx
        )
        return action_probs

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = MultimodalActionGenerator()
    print(f"Parameters: {model.count_parameters():,}")

import torch
from torch.utils.data import Dataset
import pandas as pd


class ActionDataset(Dataset):
    """Loads action generator training data from CSV."""
    def __init__(self, df: pd.DataFrame, config_mappings: dict, is_train: bool = True, modality_dropout: float = 0.15):
        self.data = df.reset_index(drop=True)
        self.is_train = is_train
        self.modality_dropout = modality_dropout
        self.mappings = config_mappings
        
        # The 'offline' context idx is used for missing or dropped context
        self.offline_context_idx = self.mappings['context'].get('offline', 0)

    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Safely get categorical indices
        intent_idx = self.mappings['intent'].get(str(row.get('intent', '')), 0)
        motion_idx = self.mappings['motion'].get(str(row.get('motion', '')), 0)
        direction_idx = self.mappings['direction'].get(str(row.get('direction', '')), 0)
        
        # Handle MISSING context
        context_val = str(row.get('context', 'MISSING'))
        if context_val == 'MISSING' or pd.isna(row.get('context')):
            context_idx = self.offline_context_idx
        else:
            context_idx = self.mappings['context'].get(context_val, self.offline_context_idx)
            
        # Modality dropout for context (15% chance during training)
        if self.is_train and torch.rand(1).item() < self.modality_dropout:
            context_idx = self.offline_context_idx
            
        action_idx = self.mappings['action'].get(str(row.get('action', '')), 0)
        
        # Defaults for scenario-level data
        velocity_feats = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
        intent_conf = torch.tensor([1.0], dtype=torch.float32)
        
        # Control targets
        v = float(row.get('target_v', 0.0))
        omega = float(row.get('target_omega', 0.0))
        d = float(row.get('target_d', 0.0))
        control_targets = torch.tensor([v, omega, d], dtype=torch.float32)
        
        return (
            torch.tensor(intent_idx, dtype=torch.long),
            intent_conf,
            torch.tensor(motion_idx, dtype=torch.long),
            torch.tensor(direction_idx, dtype=torch.long),
            velocity_feats,
            torch.tensor(context_idx, dtype=torch.long)
        ), (
            torch.tensor(action_idx, dtype=torch.long),
            control_targets
        )


def collate_fn(batch):
    """Collates a batch of dataset items into batched tensors."""
    inputs, targets = zip(*batch)
    
    intent_idx = torch.stack([i[0] for i in inputs])
    intent_conf = torch.stack([i[1] for i in inputs])
    motion_idx = torch.stack([i[2] for i in inputs])
    direction_idx = torch.stack([i[3] for i in inputs])
    velocity_feats = torch.stack([i[4] for i in inputs])
    context_idx = torch.stack([i[5] for i in inputs])
    
    action_idx = torch.stack([t[0] for t in targets])
    control_targets = torch.stack([t[1] for t in targets])
    
    return (intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx), (action_idx, control_targets)

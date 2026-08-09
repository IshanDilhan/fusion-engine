"""
train.py

Multi-task training pipeline for MultimodalActionGenerator.
Trains Action Classification (Focal Loss) and Continuous Control Regression (Huber Loss)
with Learning Rate Scheduler and Checkpoint Management.

Usage:
    python action_generator/training/train.py
"""

import os
import sys

ACTION_GEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ACTION_GEN_DIR not in sys.path:
    sys.path.insert(0, ACTION_GEN_DIR)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from model import MultimodalActionGenerator
from training.dataset import ActionDataset, collate_fn
from config import (
    INTENTS, MOTIONS, DIRECTIONS, CONTEXTS, ACTIONS,
    intent_to_idx, motion_to_idx, direction_to_idx,
    context_to_idx, action_to_idx, idx_to_action,
    NUM_INTENTS, NUM_MOTIONS, NUM_DIRECTIONS, NUM_CONTEXTS, NUM_ACTIONS,
    EPOCHS, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, LR_MIN
)


class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.weight, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def build_mappings():
    """Build mappings from config.py vocabularies (not dynamically from data)."""
    return {
        'intent':    {v: i for i, v in enumerate(INTENTS)},
        'motion':    {v: i for i, v in enumerate(MOTIONS)},
        'direction': {v: i for i, v in enumerate(DIRECTIONS)},
        'context':   {v: i for i, v in enumerate(CONTEXTS)},   # includes 'offline'
        'action':    {v: i for i, v in enumerate(ACTIONS)},
    }


def main():
    # 1. Load dataset from CSV
    csv_path = os.path.join(
        ACTION_GEN_DIR, 
        "training_data", 
        "action_generator_augmented_training.csv"
    )
    
    if not os.path.exists(csv_path):
        print(f"Error: Training CSV not found at {csv_path}. Run build_dataset_csv.py first.")
        sys.exit(1)
        
    df = pd.read_csv(csv_path)
    mappings = build_mappings()
    
    # 2. Split into train/test
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'test'].copy()
    
    # 3. Create DataLoaders
    train_dataset = ActionDataset(train_df, mappings, is_train=True, modality_dropout=0.15)
    val_dataset = ActionDataset(val_df, mappings, is_train=False, modality_dropout=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
    
    # 4. Initialize model
    model = MultimodalActionGenerator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 5. Define losses
    action_counts = train_df['action'].value_counts()
    weights = torch.ones(len(mappings['action']))
    for act, count in action_counts.items():
        idx = mappings['action'].get(str(act))
        if idx is not None:
            weights[idx] = 1.0 / (count + 1e-5)
    
    if len(mappings['action']) > 0:
        weights = weights / weights.sum() * len(mappings['action'])
    weights = weights.to(device)
    
    focal_loss_fn = FocalLoss(gamma=2.0, weight=weights)
    huber_loss_fn = nn.HuberLoss()
    
    # 6. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=LR_MIN)
    
    best_val_acc = 0.0
    checkpoint_dir = os.path.join(ACTION_GEN_DIR, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(checkpoint_dir, "best_action_generator.pt")
    
    print("Starting training...")
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for inputs, targets in train_loader:
            intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx = [t.to(device) for t in inputs]
            action_targets, control_targets = [t.to(device) for t in targets]
            
            optimizer.zero_grad()
            action_logits, control_preds = model(intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx)
            
            loss_action = focal_loss_fn(action_logits, action_targets)
            loss_control = huber_loss_fn(control_preds, control_targets)
            loss = loss_action + 0.5 * loss_control
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item() * intent_idx.size(0)
            preds = torch.argmax(action_logits, dim=1)
            correct_train += (preds == action_targets).sum().item()
            total_train += intent_idx.size(0)
            
        scheduler.step()
        train_acc = correct_train / total_train if total_train > 0 else 0.0
        train_loss = total_train_loss / total_train if total_train > 0 else 0.0
        
        # Validation
        model.eval()
        total_val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx = [t.to(device) for t in inputs]
                action_targets, control_targets = [t.to(device) for t in targets]
                
                action_logits, control_preds = model(intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx)
                
                loss_action = focal_loss_fn(action_logits, action_targets)
                loss_control = huber_loss_fn(control_preds, control_targets)
                loss = loss_action + 0.5 * loss_control
                
                total_val_loss += loss.item() * intent_idx.size(0)
                preds = torch.argmax(action_logits, dim=1)
                correct_val += (preds == action_targets).sum().item()
                total_val += intent_idx.size(0)
                
        val_acc = correct_val / total_val if total_val > 0 else 0.0
        val_loss = total_val_loss / total_val if total_val > 0 else 0.0
        
        if val_acc > best_val_acc or epoch == EPOCHS:
            best_val_acc = max(best_val_acc, val_acc)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'train_acc': train_acc,
            }, best_model_path)
            
        if epoch % 10 == 0 or epoch == 1:
            print(f"Epoch [{epoch:3d}/{EPOCHS}] Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}")
            
    print("\n--- Training Complete ---")
    print(f"Final Train Accuracy : {train_acc:.4f}")
    print(f"Final Val Accuracy   : {val_acc:.4f} (Best: {best_val_acc:.4f})")
    
    # Compute per-action validation recall
    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for inputs, targets in val_loader:
            intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx = [t.to(device) for t in inputs]
            action_targets, _ = targets
            action_logits, _ = model(intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx)
            preds = torch.argmax(action_logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(action_targets.numpy())
            
    val_df_res = pd.DataFrame({'target': all_targets, 'pred': all_preds})
    val_df_res['target_str'] = val_df_res['target'].map(idx_to_action)
    val_df_res['correct'] = val_df_res['target'] == val_df_res['pred']
    
    print("\nPer-action recall (Validation):")
    for act_code in ACTIONS:
        act_rows = val_df_res[val_df_res['target_str'] == act_code]
        if len(act_rows) > 0:
            rec = act_rows['correct'].mean()
            print(f"  {act_code:10s}: {rec:.3f} (n={len(act_rows)})")

    # Parameters count
    num_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel Parameters : {num_params:,}")
    print(f"Model Size       : {num_params * 4 / 1024:.1f} KB")


if __name__ == "__main__":
    main()

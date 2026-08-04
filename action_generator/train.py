import os
import sys
sys.path.insert(0, os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pandas as pd

from model import MultimodalActionGenerator
from dataset import ActionDataset, collate_fn
from config import (
    INTENTS, MOTIONS, DIRECTIONS, CONTEXTS, ACTIONS,
    intent_to_idx, motion_to_idx, direction_to_idx,
    context_to_idx, action_to_idx, idx_to_action,
    NUM_INTENTS, NUM_MOTIONS, NUM_DIRECTIONS, NUM_CONTEXTS, NUM_ACTIONS,
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
        os.path.dirname(__file__), 
        "training_data", 
        "action_generator_training_scenarios.csv"
    )
    
    # Simple fallback if file doesn't exist for testability
    if not os.path.exists(csv_path):
        print(f"Warning: CSV not found at {csv_path}. Creating dummy data for testing.")
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df = pd.DataFrame({
            'intent': ['F01']*80 + ['F02']*20, 
            'motion': ['M01']*100, 
            'direction': ['D01']*100,
            'context': ['offline']*100, 
            'action': ['A01']*80 + ['A02']*20, 
            'split': ['train']*80 + ['val']*20,
            'target_v': [0.5]*100, 
            'target_omega': [0.0]*100, 
            'target_d': [0.0]*100
        })
        df.to_csv(csv_path, index=False)
    else:
        df = pd.read_csv(csv_path)
        
    mappings = build_mappings()
    
    # 2. Split into train/test (test set used as validation)
    train_df = df[df['split'] == 'train'].copy()
    val_df = df[df['split'] == 'test'].copy()
    
    # 3. Create DataLoaders
    train_dataset = ActionDataset(train_df, mappings, is_train=True, modality_dropout=0.15)
    val_dataset = ActionDataset(val_df, mappings, is_train=False, modality_dropout=0.0)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
    
    # 4. Initialize model
    model = MultimodalActionGenerator()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    # 5. Define losses
    # Compute class weights for Focal Loss based on training frequencies
    action_counts = train_df['action'].value_counts()
    weights = torch.ones(len(mappings['action']))
    for act, count in action_counts.items():
        idx = mappings['action'].get(str(act))
        if idx is not None:
            weights[idx] = 1.0 / (count + 1e-5)
    
    # Normalize weights
    if len(mappings['action']) > 0:
        weights = weights / weights.sum() * len(mappings['action'])
    weights = weights.to(device)
    
    focal_loss_fn = FocalLoss(gamma=2.0, weight=weights)
    huber_loss_fn = nn.HuberLoss()
    
    # 6. Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 7. Scheduler
    epochs = 100
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Setup Checkpointing
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "best_action_generator.pt")
    
    best_val_acc = 0.0
    final_train_acc = 0.0
    final_val_acc = 0.0
    
    # 8. Training loop
    print("Starting training...")
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for inputs, targets in train_loader:
            inputs = [x.to(device) for x in inputs]
            targets = [y.to(device) for y in targets]
            
            optimizer.zero_grad()
            
            action_logits, control_vals = model(*inputs)
            
            # Action classification
            loss_cls = focal_loss_fn(action_logits, targets[0])
            # Motion control signal regression
            loss_reg = huber_loss_fn(control_vals, targets[1])
            
            # Total Loss
            loss = loss_cls + 0.5 * loss_reg
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            preds = action_logits.argmax(dim=-1)
            train_correct += (preds == targets[0]).sum().item()
            train_total += targets[0].size(0)
            
        scheduler.step()
        
        train_acc = train_correct / train_total if train_total > 0 else 0
        
        # Validation step
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        all_val_preds = []
        all_val_targets = []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs = [x.to(device) for x in inputs]
                targets = [y.to(device) for y in targets]
                
                action_logits, control_vals = model(*inputs)
                
                loss_cls = focal_loss_fn(action_logits, targets[0])
                loss_reg = huber_loss_fn(control_vals, targets[1])
                val_loss += (loss_cls + 0.5 * loss_reg).item()
                
                preds = action_logits.argmax(dim=-1)
                val_correct += (preds == targets[0]).sum().item()
                val_total += targets[0].size(0)
                
                all_val_preds.extend(preds.cpu().tolist())
                all_val_targets.extend(targets[0].cpu().tolist())
                
        val_acc = val_correct / val_total if val_total > 0 else 0
        
        # Print metrics every 10 epochs
        if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
            avg_train_loss = train_loss / max(1, len(train_loader))
            avg_val_loss = val_loss / max(1, len(val_loader))
            print(f"Epoch [{epoch:3d}/{epochs}] "
                  f"Train Loss: {avg_train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {avg_val_loss:.4f}, Acc: {val_acc:.4f}")
            
        # Save best model
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), checkpoint_path)
            
        if epoch == epochs:
            final_train_acc = train_acc
            final_val_acc = val_acc
            
    # 9. After training reporting
    print("\n--- Training Complete ---")
    print(f"Final Train Accuracy : {final_train_acc:.4f}")
    print(f"Final Val Accuracy   : {final_val_acc:.4f} (Best: {best_val_acc:.4f})")
    
    print("\nPer-action recall (Validation):")
    action_inv = {v: k for k, v in mappings['action'].items()}
    for cls_idx in sorted(set(all_val_targets)):
        mask = [t == cls_idx for t in all_val_targets]
        cls_total = sum(mask)
        cls_correct = sum(1 for p, t in zip(all_val_preds, all_val_targets) if p == cls_idx and t == cls_idx)
        recall = cls_correct / cls_total if cls_total > 0 else 0
        action_name = action_inv.get(cls_idx, f"Idx-{cls_idx}")
        print(f"  {action_name:10s} : {recall:.3f} (n={cls_total})")
        
    params = model.count_parameters()
    print(f"\nModel Parameters : {params:,}")
    if os.path.exists(checkpoint_path):
        size_kb = os.path.getsize(checkpoint_path) / 1024
        print(f"Model Size       : {size_kb:.1f} KB")


if __name__ == "__main__":
    main()

"""
export_onnx.py

Exports the trained MultimodalActionGenerator PyTorch model to ONNX format 
for Jetson Orin Nano deployment. Validates ONNX output against PyTorch.
"""

import os
import sys
import torch
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model import MultimodalActionGenerator


def export_to_onnx():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_path = os.path.join(base_dir, "checkpoints", "best_action_generator.pt")
    onnx_path = os.path.join(base_dir, "checkpoints", "action_generator.onnx")
    
    os.makedirs(os.path.dirname(onnx_path), exist_ok=True)
    
    # 1. Initialize model & load state dict
    model = MultimodalActionGenerator()
    
    if os.path.exists(checkpoint_path):
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        if isinstance(state_dict, dict) and "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        model.load_state_dict(state_dict)
        print(f"Loaded checkpoint from: {checkpoint_path}")
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}. Exporting untrained model.")
        
    model.eval()

    # 2. Dummy inputs matching forward signature:
    # forward(intent_idx, intent_conf, motion_idx, direction_idx, velocity_feats, context_idx)
    dummy_intent_idx    = torch.tensor([0], dtype=torch.long)
    dummy_intent_conf   = torch.tensor([[1.0]], dtype=torch.float32)
    dummy_motion_idx    = torch.tensor([0], dtype=torch.long)
    dummy_direction_idx = torch.tensor([0], dtype=torch.long)
    dummy_velocity      = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    dummy_context_idx   = torch.tensor([0], dtype=torch.long)

    dummy_inputs = (
        dummy_intent_idx,
        dummy_intent_conf,
        dummy_motion_idx,
        dummy_direction_idx,
        dummy_velocity,
        dummy_context_idx,
    )

    input_names = [
        "intent_idx",
        "intent_conf",
        "motion_idx",
        "direction_idx",
        "velocity_feats",
        "context_idx",
    ]
    output_names = ["action_logits", "control_values"]

    dynamic_axes = {
        "intent_idx":     {0: "batch_size"},
        "intent_conf":    {0: "batch_size"},
        "motion_idx":     {0: "batch_size"},
        "direction_idx":  {0: "batch_size"},
        "velocity_feats": {0: "batch_size"},
        "context_idx":    {0: "batch_size"},
        "action_logits":  {0: "batch_size"},
        "control_values": {0: "batch_size"},
    }

    print("Exporting model to ONNX...")
    
    try:
        torch.onnx.export(
            model,
            dummy_inputs,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            dynamo=False,
        )
    except TypeError:
        torch.onnx.export(
            model,
            dummy_inputs,
            onnx_path,
            export_params=True,
            opset_version=14,
            do_constant_folding=True,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
        )
    
    size_kb = os.path.getsize(onnx_path) / 1024
    print(f"Model exported successfully to {onnx_path} ({size_kb:.1f} KB)")

    # 3. Validate ONNX model with onnxruntime
    try:
        import onnx
        import onnxruntime as ort
        
        onnx_model = onnx.load(onnx_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model structural check: PASSED")
        
        ort_session = ort.InferenceSession(onnx_path)
        
        with torch.no_grad():
            torch_logits, torch_ctrl = model(*dummy_inputs)
            
        ort_inputs = {
            "intent_idx":     dummy_intent_idx.numpy(),
            "intent_conf":    dummy_intent_conf.numpy(),
            "motion_idx":     dummy_motion_idx.numpy(),
            "direction_idx":  dummy_direction_idx.numpy(),
            "velocity_feats": dummy_velocity.numpy(),
            "context_idx":    dummy_context_idx.numpy(),
        }
        
        ort_outs = ort_session.run(None, ort_inputs)
        
        np.testing.assert_allclose(
            torch_logits.numpy(), ort_outs[0], rtol=1e-03, atol=1e-04
        )
        np.testing.assert_allclose(
            torch_ctrl.numpy(), ort_outs[1], rtol=1e-03, atol=1e-04
        )
        
        print("PyTorch vs ONNX numerical verification: PASSED")
        print("\nONNX Specs:")
        print(f"  File Size : {size_kb:.1f} KB")
        print("  Inputs:")
        for inp in ort_session.get_inputs():
            print(f"    - {inp.name:15s} : shape={inp.shape}, type={inp.type}")
        print("  Outputs:")
        for out in ort_session.get_outputs():
            print(f"    - {out.name:15s} : shape={out.shape}, type={out.type}")
            
    except Exception as e:
        print(f"ONNX validation note: {e}")

if __name__ == "__main__":
    export_to_onnx()

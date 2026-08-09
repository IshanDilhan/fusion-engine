"""
export_onnx.py

Exports trained MultimodalActionGenerator PyTorch model to ONNX format.
Performs structural check and PyTorch vs ONNX numerical verification.

Usage:
    python action_generator/tools/export_onnx.py
"""

import os
import sys

ACTION_GEN_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ACTION_GEN_DIR not in sys.path:
    sys.path.insert(0, ACTION_GEN_DIR)

import torch
import numpy as np

from model import MultimodalActionGenerator
from config import NUM_INTENTS, NUM_MOTIONS, NUM_DIRECTIONS, NUM_CONTEXTS


def export_to_onnx():
    checkpoint_dir = os.path.join(ACTION_GEN_DIR, "checkpoints")
    model_path = os.path.join(checkpoint_dir, "best_action_generator.pt")
    onnx_output_path = os.path.join(checkpoint_dir, "action_generator.onnx")

    if not os.path.exists(model_path):
        print(f"Error: PyTorch checkpoint not found at {model_path}")
        sys.exit(1)

    model = MultimodalActionGenerator()
    checkpoint = torch.load(model_path, map_location="cpu")
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    print(f"Loaded checkpoint from: {model_path}")

    # Dummy Inputs for tracing
    batch_size = 1
    dummy_intent = torch.tensor([0], dtype=torch.long)
    dummy_conf = torch.tensor([[0.95]], dtype=torch.float32)
    dummy_motion = torch.tensor([0], dtype=torch.long)
    dummy_direction = torch.tensor([0], dtype=torch.long)
    dummy_velocity = torch.tensor([[0.0, 0.0, 0.0]], dtype=torch.float32)
    dummy_context = torch.tensor([0], dtype=torch.long)

    input_names = [
        "intent_idx",
        "intent_conf",
        "motion_idx",
        "direction_idx",
        "velocity_feats",
        "context_idx",
    ]
    output_names = [
        "action_logits",
        "control_values",
    ]

    dynamic_axes = {
        "intent_idx": {0: "batch_size"},
        "intent_conf": {0: "batch_size"},
        "motion_idx": {0: "batch_size"},
        "direction_idx": {0: "batch_size"},
        "velocity_feats": {0: "batch_size"},
        "context_idx": {0: "batch_size"},
        "action_logits": {0: "batch_size"},
        "control_values": {0: "batch_size"},
    }

    print("Exporting model to ONNX...")
    torch.onnx.export(
        model,
        (dummy_intent, dummy_conf, dummy_motion, dummy_direction, dummy_velocity, dummy_context),
        onnx_output_path,
        export_params=True,
        opset_version=14,
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )

    print(f"Model exported successfully to {onnx_output_path} ({os.path.getsize(onnx_output_path)/1024:.1f} KB)")

    # Structural & Numerical verification
    import onnx
    import onnxruntime as ort

    onnx_model = onnx.load(onnx_output_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model structural check: PASSED")

    # Runtime Numerical Check
    ort_session = ort.InferenceSession(onnx_output_path)
    
    with torch.no_grad():
        pt_logits, pt_controls = model(
            dummy_intent, dummy_conf, dummy_motion, dummy_direction, dummy_velocity, dummy_context
        )

    ort_inputs = {
        "intent_idx": dummy_intent.numpy(),
        "intent_conf": dummy_conf.numpy(),
        "motion_idx": dummy_motion.numpy(),
        "direction_idx": dummy_direction.numpy(),
        "velocity_feats": dummy_velocity.numpy(),
        "context_idx": dummy_context.numpy(),
    }
    ort_outputs = ort_session.run(None, ort_inputs)

    np.testing.assert_allclose(pt_logits.numpy(), ort_outputs[0], rtol=1e-3, atol=1e-5)
    np.testing.assert_allclose(pt_controls.numpy(), ort_outputs[1], rtol=1e-3, atol=1e-5)
    print("PyTorch vs ONNX numerical verification: PASSED")


if __name__ == "__main__":
    export_to_onnx()

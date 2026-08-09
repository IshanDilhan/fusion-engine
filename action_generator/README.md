# Action Generator — Multimodal Neural Policy Module for HRI

> **Adaptive Multimodal Human–Robot Interaction (HRI) Framework**  
> **Module**: Action Generator (`MultimodalActionGenerator`)  
> **Target Hardware**: NVIDIA Jetson Orin Nano (8GB)  
> **Latency**: < 1.8 ms  
> **Model Footprint**: 16,890 parameters (~72 KB ONNX)

---

## 1. Executive Summary & System Role

In the Adaptive Multimodal HRI Framework, perception modalities (Emotion, Gesture, Motion, Context) are processed by upstream cue runners and fused by the **Fusion Engine** into a human **Intent Code ($F01–F10$)**.

The **Action Generator** is the downstream neural policy module. It takes **four inputs**:
1. **Intent Code** ($F01–F10$) + Confidence (from Fusion Engine)
2. **Motion State** (`sitting`, `standing`, `walking`, `stepping_back`) (direct branch from Motion Runner)
3. **Motion Direction** (`toward_robot`, `away_from_robot`, `toward_object`, `toward_exit`, `lateral`, `stationary`)
4. **Context Scene** (`classroom`, `kitchen`, `offline`) (direct branch from Context Runner)

It predicts the optimal **Robot Action Code ($A01–A15$)** along with continuous **Motion Control Signals**:
- Linear Velocity $v$ (m/s)
- Angular Velocity $\omega$ (rad/s)
- Comfort Distance $d$ (meters)

---

## 2. Pipeline Integration Flow

```
                      [ PERCEPTION RUNNERS ]
           ┌──────────────┬──────────────┬──────────────┐
           ▼              ▼              ▼              ▼
       [Emotion]      [Gesture]      [Motion]       [Context]
     (MobileNetV2)      (TCN)      (LSTM+Attn)     (CLIP/VLM)
           │              │              │              │
           └──────────────┴──────┬───────┴──────────────┘
                                 ▼
                          [FUSION ENGINE]
                            (LightGBM)
                                 │
                                 ▼
                          Intent (F01–F10)
                                 │
 ┌───────────────────────────────┼───────────────────────────────┐
 │                               ▼                               │
 │                  ┌────────────────────────┐                   │
 │ Direct Branch ──▶│    ACTION GENERATOR    │◀── Direct Branch  │
 │ (Motion &        │ (Multi-Task Policy Net)│    (Context Scene │
 │  Direction)      └────────────┬───────────┘     Classroom/Kit)│
 │                               │                               │
 │                               ▼                               │
 │                    [SAFETY OVERRIDE FILTER]                   │
 │                               │                               │
 └───────────────────────────────┼───────────────────────────────┘
                                 ▼
                     ┌───────────────────────┐
                     │ Action Code (A01–A15) │
                     │ Control: [v, ω, d]    │
                     └───────────────────────┘
                                 │
                                 ▼
                        [ROBOT CONTROLLER]
```

---

## 3. File Map & Directory Structure

```
d:\fusion-engine\action_generator\
├── __init__.py                          # Package initialization
├── config.py                           # Vocabulary mappings, hyperparameters & control defaults
├── model.py                            # PyTorch MultimodalActionGenerator nn.Module (~17K params)
├── dataset.py                          # ActionDataset loader with Modality Dropout
├── build_dataset_csv.py                # Script to extract 62 scenarios to CSVs
├── train.py                            # Multi-task training script (Focal + Huber Loss)
├── inference.py                        # Python runtime inference engine with safety rules
├── safety_override.py                  # Hard-coded post-prediction emergency filter
├── export_onnx.py                      # ONNX export & verification script
├── run_all.py                          # Master automation runner script
├── index.html                          # Single-page ONNX Web App for Hugging Face Spaces
├── README.md                           # Main module documentation (This file)
├── MODEL_ARCHITECTURE.md               # Technical breakdown of model & math
├── checkpoints/
│   ├── best_action_generator.pt        # Trained PyTorch weights (72.2 KB)
│   └── action_generator.onnx           # Jetson deployment ONNX model (72.0 KB)
└── training_data/
    ├── action_generator_training_scenarios.csv  # 62 base scenario table
    └── action_generator_augmented_training.csv  # 431 augmented training samples
```

---

## 4. Quickstart Guide (Commands & Execution)

### 4.1 Activate Virtual Environment
```powershell
# From project root
.\.venv\Scripts\Activate.ps1
cd action_generator
```

### 4.2 Generate CSV Datasets
```bash
python build_dataset_csv.py
```
*Outputs*: `training_data/action_generator_training_scenarios.csv` (62 base rows) and `training_data/action_generator_augmented_training.csv` (431 augmented rows).

### 4.3 Train the Model
```bash
python train.py
```
*Results*: Achieves **95.1% Train Accuracy / 85.7% Validation Accuracy**. Saves `checkpoints/best_action_generator.pt`.

### 4.4 Test Runtime Inference & Safety Overrides
```bash
python run_all.py --step infer
```

### 4.5 Export ONNX Model
```bash
python export_onnx.py
```
*Outputs*: Verified ONNX checkpoint `checkpoints/action_generator.onnx`.

---

## 5. Python Integration Code

Integrate into your main robot loop:

```python
from action_generator.inference import ActionInference

# 1. Initialize Engine
action_engine = ActionInference('action_generator/checkpoints/best_action_generator.pt')

# 2. Predict inside perception loop
result = action_engine.predict(
    intent='F02',                   # From Fusion Engine
    intent_confidence=0.95,          # From Fusion Engine
    motion_state='stepping_back',        # Direct from Motion Runner
    direction='away_from_robot',     # Direct from Motion Runner
    velocity=0.8,                    # Human speed (m/s)
    context='kitchen'                # Direct from Context Runner
)

# 3. Send outputs to Robot Controller
print(f"Action        : {result.action}")                   # e.g., 'A02'
print(f"Description   : {result.action_description}")       # 'Halt all motion + fire/smoke alert'
print(f"Linear Speed  : {result.linear_velocity_m_s} m/s")  # 0.0 m/s
print(f"Comfort Dist  : {result.comfort_distance_m} m")     # 2.0 m
print(f"Safety Active : {result.safety_override_active}")   # True
```

---

## 6. Safety Override System

Safety is **never left to learned weights**. The `safety_override.py` module enforces hard post-prediction rules:

- **Trigger Conditions**:
  - `intent == 'F02'` (Emergency Intent), OR
  - $P(A02) \ge 0.15$ (Fire alert threshold), OR
  - $P(A03) \ge 0.15$ (Medical alert threshold), OR
  - $P(A14) \ge 0.15$ (Classroom hazard threshold)

- **Enforced Response**:
  - **Kitchen Context**: Action forced to `A02` (Fire Alert)
  - **Classroom Context**: Action forced to `A14` (Halt Risky Action / Notify Supervisor)
  - **Linear Speed**: Forced to `0.0 m/s` (Immediate Halt)
  - **Comfort Distance**: Forced to `2.0 m` (Maximum Safety Distance)

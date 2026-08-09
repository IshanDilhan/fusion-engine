# Action Generator — Multimodal Neural Policy Module for HRI

> **Adaptive Multimodal Human–Robot Interaction (HRI) Framework**  
> **Module**: Action Generator (`MultimodalActionGenerator`)  
> **Target Hardware**: NVIDIA Jetson Orin Nano (8GB)  
> **Latency**: < 1.8 ms  
> **Model Footprint**: 16,874 parameters (~72 KB ONNX)

---

## 1. Executive Summary & System Role

In the Adaptive Multimodal HRI Framework, perception modalities (Emotion, Gesture, Motion, Context) are processed by upstream cue runners and fused by the **Fusion Engine** into a human **Intent Code ($F01–F09$)**.

The **Action Generator** is the downstream neural policy module. It takes **four inputs**:
1. **Intent Code** ($F01–F09$) + Confidence (from Fusion Engine)
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
                          Intent (F01–F09)
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

## 3. Streamlined File Structure

```
action_generator/
├── __init__.py                          # Package initialization
├── config.py                           # Vocabulary mappings (F01-F09, A01-A15) & control defaults
├── model.py                            # PyTorch MultimodalActionGenerator nn.Module (~16.8K params)
├── dataset.py                          # ActionDataset loader with Modality Dropout
├── build_dataset_csv.py                # Script to extract 73 base scenarios to CSVs
├── train.py                            # Multi-task training script (Focal + Huber Loss)
├── inference.py                        # Python runtime inference engine with safety rules
├── safety_override.py                  # Hard-coded post-prediction emergency filter
├── export_onnx.py                      # ONNX export & verification script
├── live_video_demo.py                  # Real-time visual video tester with Expected vs. Got HUD
├── README.md                           # Core module documentation (This file)
├── checkpoints/
│   ├── best_action_generator.pt        # Trained PyTorch weights (72.0 KB)
│   └── action_generator.onnx           # Jetson deployment ONNX model (71.8 KB)
└── training_data/
    ├── action_generator_training_scenarios.csv  # 72 base scenario table
    └── action_generator_augmented_training.csv  # 318 augmented training samples
```

---

## 4. Quickstart Guide (Commands & Execution)

### 4.1 Generate CSV Datasets
```bash
python build_dataset_csv.py
```
*Outputs*: `training_data/action_generator_training_scenarios.csv` (72 base rows) and `training_data/action_generator_augmented_training.csv` (318 augmented rows).

### 4.2 Train the Model
```bash
python train.py
```
*Results*: Achieves **94.08% Train Accuracy / 70.97% Validation Accuracy**. Saves `checkpoints/best_action_generator.pt`.

### 4.3 Export ONNX Model
```bash
python export_onnx.py
```
*Outputs*: Verified ONNX checkpoint `checkpoints/action_generator.onnx`.

### 4.4 Live Video HUD Tester
```bash
python live_video_demo.py --source synthetic
```

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
    motion_state='stepping_back',    # Direct from Motion Runner
    direction='away_from_robot',     # Direct from Motion Runner
    velocity=0.8,                    # Human speed (m/s)
    context='kitchen'                # Direct from Context Runner
)

# 3. Send outputs to Robot Controller
print(f"Action        : {result.action}")                   # e.g., 'A02' (Emergency Halt!)
print(f"Description   : {result.action_description}")       # 'Halt all motion + fire/smoke alert'
print(f"Linear Speed  : {result.linear_velocity_m_s} m/s")  # 0.0 m/s (Emergency Stop)
print(f"Comfort Dist  : {result.comfort_distance_m} m")     # 2.0 m (Max Clearance)
print(f"Safety Active : {result.safety_override_active}")   # True
print(f"Safety Reason : {result.safety_reason}")            # Emergency hazard bypass triggered
```

---

## 6. Internal Physical Safety Gate

Safety is **handled internally** through a 2-tier priority policy inside `safety_override.py`:

- **Priority 1: Emergency Hazard Bypass**:
  - **Trigger**: `intent == 'F02'` (Emergency Intent), OR hazard action probability $\ge 0.15$.
  - **Behavior**: Forces emergency action (`A02` Kitchen / `A14` Classroom), halts linear motion ($v=0.0\text{ m/s}$), and sets maximum clearance ($d=2.0\text{ m}$).

- **Priority 2: Dynamic Proximity Yielding Gate**:
  - **Trigger**: Human moving `toward_robot` at speed $>0.5\text{ m/s}$ and distance $<1.0\text{ m}$.
  - **Behavior**: **Preserves predicted Action Code** (protects model accuracy), clamps linear velocity to **$v=-0.2\text{ m/s}$** (robot physically steps back to yield clearance space), and sets comfort clearance to **$d=1.5\text{ m}$**.

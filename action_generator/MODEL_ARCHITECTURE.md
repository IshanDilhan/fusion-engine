# MultimodalActionGenerator — Technical Architecture & Math Guide

> **Technical Architecture Specification & Mathematical Formulation**  
> **Model Name**: `MultimodalActionGenerator`  
> **Framework**: PyTorch (`nn.Module`) / ONNX Runtime  
> **Total Parameters**: 16,922 (~17K)  
> **Model Checkpoint Size**: 72.2 KB (.pt) / 72.0 KB (.onnx)

---

## 1. Network Architecture Diagram

```
 Input Tensors
 ┌──────────────────────────────────────────────┐
 │ 1. Intent Index    (0..9)  ──▶ Embed(10, 16) │──┐ (16D)
 │ 2. Intent Conf     (0..1)  ──▶ Scalar        │──┼──▶ (1D)
 │ 3. Motion Index    (0..5)  ──▶ Embed(6, 16)  │──┼──▶ (16D)
 │ 4. Direction Index (0..5)  ──▶ Embed(6, 8)   │──┼──▶ (8D)
 │ 5. Velocity Vector [v,a,m] ──▶ Continuous    │──┼──▶ (3D)
 │ 6. Context Index   (0..2)  ──▶ Embed(3, 8)   │──┘ (8D)
 └──────────────────────────────────────────────┘
                        │
                        ▼  Concatenate Tensors
            Input Vector x ∈ ℝ⁵²
                        │
                        ▼
 ┌──────────────────────────────────────────────┐
 │ Dense Fusion Core                            │
 │                                              │
 │  Layer 1: Linear(52 ──▶ 128)                 │
 │           LayerNorm(128)                     │
 │           GELU() Activation                  │
 │           Dropout(p = 0.20)                  │
 │                                              │
 │  Layer 2: Linear(128 ──▶ 64)                 │
 │           LayerNorm(64)                      │
 │           GELU() Activation                  │
 └──────────────────────┬───────────────────────┘
                        │
                        ▼  Bottleneck Features h₂ ∈ ℝ⁶⁴
             ┌──────────┴──────────┐
             ▼                     ▼
 ┌──────────────────────┐  ┌──────────────────────┐
 │ Head 1: Action Class │  │ Head 2: Motion Ctrl  │
 │ Linear(64 ──▶ 15)    │  │ Linear(64 ──▶ 3)     │
 │                      │  │                      │
 │ Output: Logits       │  │ Output: Continuous   │
 │   Softmax ──▶ P(Aᵢ)  │  │   [v, ω, d]          │
 └──────────────────────┘  └──────────────────────┘
```

---

## 2. Input Tensor Mathematical Formulation

The model converts discrete and continuous inputs into numerical tensors, passing them through 4 trainable embedding lookup tables and a continuous feature vector:

$$\mathbf{x} = [\mathbf{e}_{\text{intent}} \,\|\, c_{\text{intent}} \,\|\, \mathbf{e}_{\text{motion}} \,\|\, \mathbf{e}_{\text{direction}} \,\|\, \mathbf{v}_{\text{human}} \,\|\, \mathbf{e}_{\text{context}}] \in \mathbb{R}^{52}$$

Where:
- $\mathbf{e}_{\text{intent}} \in \mathbb{R}^{16}$: Learnable embedding for Intent $F01–F10$ ($\text{Vocabulary}=10$)
- $c_{\text{intent}} \in \mathbb{R}^{1}$: Continuous Intent confidence score $[0.0, 1.0]$ from Fusion Engine
- $\mathbf{e}_{\text{motion}} \in \mathbb{R}^{16}$: Learnable embedding for Motion state (`sit`, `stand`, `walk`, `run`, `step_back`, `lean_forward`) ($\text{Vocabulary}=6$)
- $\mathbf{e}_{\text{direction}} \in \mathbb{R}^{8}$: Learnable embedding for Motion Direction (`toward_robot`, `away_from_robot`, `toward_object`, `toward_exit`, `lateral`, `stationary`) ($\text{Vocabulary}=6$)
- $\mathbf{v}_{\text{human}} \in \mathbb{R}^{3}$: Continuous human movement vector $[\text{speed (m/s)}, \text{acceleration placeholder}, \text{is\_moving\_flag}]$
- $\mathbf{e}_{\text{context}} \in \mathbb{R}^{8}$: Learnable embedding for Context Scene (`classroom`, `kitchen`, `offline`) ($\text{Vocabulary}=3$)

---

## 3. Dense Fusion Core Formulation

The $52\text{-dimensional}$ feature vector $\mathbf{x}$ passes through two dense projection blocks with normalization, non-linear activation, and dropout regularization:

$$\mathbf{h}_1 = \text{GELU}(\text{LayerNorm}(\mathbf{W}_1 \mathbf{x} + \mathbf{b}_1)), \quad \mathbf{W}_1 \in \mathbb{R}^{128 \times 52}, \, \mathbf{b}_1 \in \mathbb{R}^{128}$$

$$\mathbf{h}_2 = \text{GELU}(\text{LayerNorm}(\mathbf{W}_2 \text{Dropout}(\mathbf{h}_1, p=0.20) + \mathbf{b}_2)), \quad \mathbf{W}_2 \in \mathbb{R}^{64 \times 128}, \, \mathbf{b}_2 \in \mathbb{R}^{64}$$

---

## 4. Multi-Task Output Heads

### Head 1: Action Classification ($A01–A15$)
Computes probability distribution across all 15 robot actions:

$$\mathbf{z}_{\text{action}} = \mathbf{W}_{\text{act}} \mathbf{h}_2 + \mathbf{b}_{\text{act}} \in \mathbb{R}^{15}, \quad \mathbf{W}_{\text{act}} \in \mathbb{R}^{15 \times 64}$$

$$P(A_i) = \text{Softmax}(\mathbf{z}_{\text{action}})_i = \frac{\exp(z_i)}{\sum_{j=1}^{15} \exp(z_j)}$$

### Head 2: Motion Controller ($v, \omega, d$)
Regresses continuous motor and safety parameters:

$$\begin{bmatrix} v \\ \omega \\ d \end{bmatrix} = \mathbf{W}_{\text{ctrl}} \mathbf{h}_2 + \mathbf{b}_{\text{ctrl}} \in \mathbb{R}^{3}, \quad \mathbf{W}_{\text{ctrl}} \in \mathbb{R}^{3 \times 64}$$

- $v \in [0.0, 1.2]$ m/s (Linear speed)
- $\omega \in [0.0, 0.5]$ rad/s (Turning rate)
- $d \in [0.5, 3.0]$ m (Comfort distance)

---

## 5. Multi-Task Loss Objective

Training minimizes a composite multi-task loss function combining **Focal Loss** (for class-imbalanced action classification) and **Huber Loss** (for continuous velocity regression):

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{Focal}}(\mathbf{y}_{\text{action}}, \hat{\mathbf{y}}_{\text{action}}) + 0.5 \cdot \mathcal{L}_{\text{Huber}}(\mathbf{u}_{\text{ctrl}}, \hat{\mathbf{u}}_{\text{ctrl}})$$

### Focal Loss Formulation ($\gamma = 2.0$):
$$\mathcal{L}_{\text{Focal}} = -\alpha_t (1 - p_t)^\gamma \log(p_t)$$

Where $p_t$ is the model's estimated probability for the true ground-truth class. Down-weights easy background samples and forces the model to learn rare safety actions ($A02, A03, A14$).

---

## 6. Training Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| `EMBEDDING_DIM` | 16 / 8 | Embedding sizes per categorical input |
| `HIDDEN_DIM` | 128 $\rightarrow$ 64 | Dense Fusion Core hidden dimensions |
| `DROPOUT` | 0.20 | Inter-layer regularization probability |
| `OPTIMIZER` | AdamW | `lr=1e-3`, `weight_decay=1e-4` |
| `SCHEDULER` | CosineAnnealingLR | `T_max=100`, `eta_min=1e-5` |
| `MODALITY_DROPOUT_P` | 0.15 | 15% probability of context sensor masking |
| `EPOCHS` | 100 | Training duration (~30 seconds execution time) |
| `BATCH_SIZE` | 32 | Mini-batch size |
| `PARAM_COUNT` | 16,922 | Total trainable parameter count |

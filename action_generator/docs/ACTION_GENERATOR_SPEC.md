# Action Generator — Module Technical Specification & Architecture

> **Adaptive Multimodal Human–Robot Interaction (HRI) Framework**  
> **Module**: Action Generator (`MultimodalActionGenerator`)  
> **Reference Document**: Intent ($F01–F09$) to Robot Action ($A01–A15$) Mapping & Safety Policy

---

## 1. Intent-to-Action Reference Matrix (Table A.1 & A.2)

| Intent Code & Name | Primary Classroom Action | Primary Kitchen Action | Action Description & Robot Behavior | Linear Velocity ($v$) | Comfort Clearance ($d$) |
|---|---|---|---|---|---|
| **F01**: Greeting / Positive Ack | **A01** | **A01** | Positive acknowledgment; prompt/prepare next task | `0.0 m/s` | `1.0 m` |
| **F02**: Emergency / Danger Hazard | **A14** | **A02** | **A14**: Halt risky action / notify teacher<br>**A02**: Halt all motion + fire/smoke alert | `0.0 m/s` (Stop) | `2.0 m` (Safety) |
| **F03**: Task Assistance Request | **A04** / **A13** | **A04** / **A13** | **A04**: Approach/follow to indicated spot<br>**A13**: Follow; offer to fetch/carry | `0.3 m/s` / `0.4 m/s` | `0.8 m` / `1.0 m` |
| **F04**: Help Request in Distress | **A05** / **A15** | **A05** | **A05**: Offer task guidance, supportive tone<br>**A15**: Ask clarifying question | `0.0 m/s` | `1.0 m` |
| **F05**: Engaged / Busy — Do Not Interrupt | **A06** | **A06** | Hold position; do not interrupt; stay aware | `0.0 m/s` | `1.5 m` |
| **F06**: Requests Passage / Clear Space | **A11** | **A11** | Move aside promptly; clear path | `0.3 m/s` | `0.5 m` |
| **F07**: Frustration / Agitation | **A08** | **A08** | Calm assistance, soft tone (de-escalation) | `0.0 m/s` | `1.5 m` |
| **F08**: Break / Relief Request | **A07** | **A07** | Acknowledge; suggest break/alternative activity | `0.0 m/s` | `1.2 m` |
| **F09**: Discouraged / Giving Up | **A12** | **A12** | Encourage; suggest alternative approach | `0.0 m/s` | `1.0 m` |

---

## 2. Model Architecture & Math

- **Multi-Task Dense Fusion Network**:
  - Input: $[E_{\text{intent}}, \, c_{\text{intent}}, \, E_{\text{motion}}, \, E_{\text{direction}}, \, v_{\text{human}}, \, E_{\text{context}}] \in \mathbb{R}^{52}$
  - Embeddings: Intent ($9 \to 16$), Motion ($4 \to 16$), Direction ($6 \to 8$), Context ($3 \to 8$).
  - Core: Dense 52 $\to$ 128 $\to$ 64 with LayerNorm + GELU + Dropout(0.20).
  - Head 1: Action Classifier ($64 \to 15$, Softmax).
  - Head 2: Continuous Motor Controller ($64 \to 3$, $[v, \omega, d]$).

---

## 3. Internal Physical Safety Gate

- **Priority 1: Emergency Hazard Bypass**:
  - `F02` Intent $\implies$ Forced `A02`/`A14` Action, $v = 0.0\text{ m/s}$, $d = 2.0\text{ m}$.
- **Priority 2: Dynamic Proximity Yielding Gate**:
  - Rapid human approach ($<1.0\text{ m}$) $\implies$ Action Code preserved, $v = -0.20\text{ m/s}$ reverse yield step, $d = 1.5\text{ m}$.

# HRI Intent-to-Action Master Reference Mapping Table

> **Adaptive Multimodal Human–Robot Interaction (HRI) Framework**  
> **Module**: Action Generator (`MultimodalActionGenerator`)  
> **Reference Document**: Intent ($F01–F10$) to Robot Action ($A01–A15$) Mapping Matrix

---

## 1. Executive Intent-to-Action Reference Matrix

| Intent Code & Name | Primary Classroom Action | Primary Kitchen Action | Action Description & Robot Behavior | Linear Velocity ($v$) | Comfort Dist ($d$) | Safety Override Status |
|---|---|---|---|---|---|---|
| **F01**: Greeting / Positive Ack | **A01** | **A01** | Positive acknowledgment; prompt/prepare next task | `0.0 m/s` | `1.0 m` | ● Normal |
| **F02**: Emergency / Danger Hazard | **A14** | **A02** | **A14**: Halt risky action / notify teacher<br>**A02**: Halt all motion + fire/smoke alert | `0.0 m/s`<br>(Forced Stop) | `2.0 m`<br>(Max Safety) | 🚨 **Emergency Stop** |
| **F03**: Task Assistance Request | **A04** / **A13** | **A04** / **A13** | **A04**: Approach/follow to indicated spot<br>**A13**: Follow; offer to fetch/carry | `0.3 m/s`<br>`0.4 m/s` | `0.8 m`<br>`1.0 m` | ● Normal |
| **F04**: Help Request in Distress | **A05** / **A15** | **A05** | **A05**: Offer task guidance, supportive tone<br>**A15**: Ask clarifying question | `0.0 m/s` | `1.0 m` | ● Normal |
| **F05**: Engaged / Busy — Do Not Interrupt | **A06** | **A06** | Hold position; do not interrupt; stay aware | `0.0 m/s` | `1.5 m` | ● Normal |
| **F06**: Requests Passage / Clear Space | **A11** | **A11** | Move aside promptly; clear path | `0.3 m/s` | `0.5 m` | ● Normal |
| **F07**: Frustration / Agitation | **A08** | **A08** | Calm assistance, soft tone (de-escalation) | `0.0 m/s` | `1.5 m` | ● Normal |
| **F08**: Break / Relief Request | **A07** | **A07** | Acknowledge; suggest break/alternative activity | `0.0 m/s` | `1.2 m` | ● Normal |
| **F09**: Discouraged / Giving Up | **A12** | **A12** | Encourage; suggest alternative approach | `0.0 m/s` | `1.0 m` | ● Normal |

---

## 2. Context & Motion Disambiguation Rules

### Rule 1: Emergency Hazard Context Selection ($F02$)
- **Context = Kitchen**: Hazard triggers **A02** (*Fire/smoke alert + halt*).
- **Context = Classroom**: Hazard triggers **A14** (*Halt risky action + notify supervisor*).
- **Enforcement**: Immediate hard-coded safety override ($v = 0.0\text{ m/s}, \, d = 2.0\text{ m}$).

### Rule 2: Assistance Disambiguation ($F03$)
- **Motion = Seated / Stationary**: Action routes to **A04** (*Approach indicated spot*).
- **Motion = Walk + Direction = toward_object**: Action routes to **A13** (*Follow / fetch / carry*).

### Rule 3: Passage Request Disambiguation ($F06$)
- **Motion = Walk + Direction = toward_robot**: Action routes to **A11** (*Move aside promptly*).

---

## 3. Unified Robot Action Legend ($A01–A15$)

| Code | Action Name | Default Linear Velocity ($v$) | Default Angular Velocity ($\omega$) | Default Comfort Distance ($d$) |
|---|---|---|---|---|
| **A01** | Positive acknowledgment; prompt/prepare next task | `0.0 m/s` | `0.0 rad/s` | `1.0 m` |
| **A02** | Halt all motion + fire/smoke alert | `0.0 m/s` | `0.0 rad/s` | `2.0 m` |
| **A03** | Halt all motion + medical alert | `0.0 m/s` | `0.0 rad/s` | `2.0 m` |
| **A04** | Approach/follow to indicated spot; await instruction | `0.3 m/s` | `0.2 rad/s` | `0.8 m` |
| **A05** | Offer task guidance / answer, supportive tone | `0.0 m/s` | `0.1 rad/s` | `1.0 m` |
| **A06** | Hold position; do not interrupt; stay aware | `0.0 m/s` | `0.0 rad/s` | `1.5 m` |
| **A07** | Acknowledge; suggest break/alternative activity | `0.0 m/s` | `0.0 rad/s` | `1.2 m` |
| **A08** | Calm assistance, soft tone (de-escalation) | `0.0 m/s` | `0.05 rad/s` | `1.5 m` |
| **A09** | Wave back; do not follow | `0.0 m/s` | `0.0 rad/s` | `1.5 m` |
| **A10** | Acknowledge farewell; hold position | `0.0 m/s` | `0.0 rad/s` | `1.5 m` |
| **A11** | Move aside promptly; clear path | `0.3 m/s` | `0.5 rad/s` | `0.5 m` |
| **A12** | Encourage; suggest alternative approach | `0.0 m/s` | `0.0 rad/s` | `1.0 m` |
| **A13** | Follow; offer to fetch/carry | `0.4 m/s` | `0.2 rad/s` | `1.0 m` |
| **A14** | Halt risky action; check surroundings; notify supervisor | `0.0 m/s` | `0.1 rad/s` | `1.8 m` |
| **A15** | Ask clarifying question | `0.0 m/s` | `0.0 rad/s` | `1.0 m` |

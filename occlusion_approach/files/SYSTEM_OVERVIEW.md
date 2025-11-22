# System Architecture Overview

## System Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                    POSE ESTIMATION SYSTEM                        │
└─────────────────────────────────────────────────────────────────┘

Input Stream                    Processing                   Output
━━━━━━━━━━━                    ━━━━━━━━━━                  ━━━━━━━
                                                                    
Camera Pose     ┌──────────┐   ┌──────────────┐   ┌──────────────┐
Observations ──>│ Sensor   │──>│   Bayesian   │──>│  Estimated   │
(6D: x,y,z,    │ Model    │   │    Filter    │   │  Pose + Cov  │
 qw,qx,qy,qz)  └──────────┘   │              │   └──────────────┘
                │              │  ┌────────┐  │           │
                │ Occlusion    │  │  EKF   │  │           │
                │ Detection    │  │   or   │  │           ▼
                │              │  │   PF   │  │   ┌──────────────┐
                └──────────────┘  └────────┘  │   │ Interactive  │
                                  │            │   │ Visualizer   │
                Ground Truth      │ Motion     │   └──────────────┘
                Trajectory        │ Model +    │           │
                                  │ Prediction │           │
                                  │            │           ▼
                Camera Intrinsics │ Measurement│   User Display
                (K matrix)        │ Update     │   - 3D Scene
                                  └────────────┘   - Uncertainty
                                                   - Statistics
```

## Data Flow

```
┌─────────────────────────────────────────────────────────────────┐
│                         PROCESSING CYCLE                         │
└─────────────────────────────────────────────────────────────────┘

Frame t-1                Frame t                  Frame t+1
─────────                ───────                  ─────────

State: x(t-1)           State: x(t)              State: x(t+1)
Covariance: P(t-1)      Covariance: P(t)         Covariance: P(t+1)
     │                       │                         │
     │                       │                         │
     ▼                       ▼                         ▼
┌─────────┐            ┌─────────┐              ┌─────────┐
│PREDICTION│           │PREDICTION│             │PREDICTION│
│         │           │         │             │         │
│ x̄ = f(x)│           │ x̄ = f(x)│             │ x̄ = f(x)│
│ P̄=FPF'+Q│           │ P̄=FPF'+Q│             │ P̄=FPF'+Q│
└────┬────┘           └────┬────┘             └────┬────┘
     │                     │                        │
     │ Observation z(t-1)  │ Observation z(t)       │ Obs z(t+1)
     │ (may be occluded)   │ (may be occluded)      │
     ▼                     ▼                        ▼
┌─────────┐            ┌─────────┐              ┌─────────┐
│ UPDATE  │            │ UPDATE  │              │ UPDATE  │
│         │            │         │              │         │
│If z≠null│            │If z≠null│              │If z≠null│
│ K=P̄H'S⁻¹│            │ K=P̄H'S⁻¹│              │ K=P̄H'S⁻¹│
│ x=x̄+Ky  │            │ x=x̄+Ky  │              │ x=x̄+Ky  │
│ P=(I-KH)│            │ P=(I-KH)│              │ P=(I-KH)│
│   ·P̄    │            │   ·P̄    │              │   ·P̄    │
└─────────┘            └─────────┘              └─────────┘

    Normal               Occluded                 Recovered
  (σ small)             (σ large)               (σ reduced)
```

## State Vector Components

```
┌──────────────────────────────────────────────────────────┐
│                   STATE VECTOR (13D)                     │
└──────────────────────────────────────────────────────────┘

Index   Component      Description           Range
─────   ─────────      ───────────           ─────
0-2     Position       [x, y, z]             ℝ³
        (Cartesian)    Location in space     [-∞, +∞]
                       
3-6     Orientation    [qw,qx,qy,qz]         S³
        (Quaternion)   Rotation as unit      ||q|| = 1
                       quaternion            
                       
7-9     Lin. Velocity  [vx, vy, vz]          ℝ³
        (Cartesian)    Rate of position      [-∞, +∞]
                       change                
                       
10-12   Ang. Velocity  [wx, wy, wz]          ℝ³
        (Axis-Angle)   Rate of rotation      [-∞, +∞]


Covariance Matrix P (13×13):
┌                                                        ┐
│ σ²ₓₓ  σ²ₓᵧ  σ²ₓᵧ  ...                                 │
│ σ²ᵧₓ  σ²ᵧᵧ  σ²ᵧᵧ  ...         Position-Position        │
│ σ²ᵧₓ  σ²ᵧᵧ  σ²ᵧᵧ  ...                                 │
│                                                        │
│                        Quaternion-Quaternion           │
│                                                        │
│                                    Velocity-Velocity   │
└                                                        ┘
                  (Symmetric 13×13 matrix)
```

## Algorithm Comparison

```
┌──────────────────────────────────────────────────────────┐
│           EXTENDED KALMAN FILTER (EKF)                   │
└──────────────────────────────────────────────────────────┘

Representation:     Single Gaussian
State:              Mean x̂ and Covariance P
Complexity:         O(n²) per frame

Prediction:         Update:
┌──────────┐       ┌──────────┐
│ x̄ = f(x) │       │ K=P̄H'S⁻¹ │
│ P̄=FPF'+Q │       │ x=x̄+K(y) │
└──────────┘       │ P=(I-KH)P│
                   └──────────┘

Strengths:                  Limitations:
✓ Fast computation         ✗ Gaussian assumption
✓ Low memory              ✗ Linearization errors
✓ Optimal (if linear)     ✗ Single mode only
✓ Real-time capable       ✗ May diverge if very nonlinear


┌──────────────────────────────────────────────────────────┐
│              PARTICLE FILTER (PF)                         │
└──────────────────────────────────────────────────────────┘

Representation:     N weighted samples
State:              {x⁽ⁱ⁾, w⁽ⁱ⁾} for i=1..N
Complexity:         O(N·n) per frame

Prediction:         Update:              Resample:
┌──────────┐       ┌──────────┐        ┌──────────┐
│ x⁽ⁱ⁾~p(x│)      │ w⁽ⁱ⁾∝p(z│x)│        │ Draw N   │
│  |x⁽ⁱ⁾   │       │ Normalize│        │ samples  │
└──────────┘       └──────────┘        │ by weight│
                                       └──────────┘

Strengths:                  Limitations:
✓ No assumptions          ✗ Computationally expensive
✓ Multimodal support      ✗ High memory
✓ Nonlinear handling      ✗ Particle degeneracy
✓ Flexible               ✗ Requires many particles
```

## Occlusion Handling

```
┌──────────────────────────────────────────────────────────┐
│                  OCCLUSION SCENARIOS                      │
└──────────────────────────────────────────────────────────┘

Scenario 1: NO OCCLUSION
──────────────────────────
         ╔══════╗
Camera   ║Object║  
   🎥────→║  🔵  ║  Clear line of sight
         ╚══════╝  
         
Measurement: Valid, low noise (σ = 0.02m)
Action: Normal update
Result: Uncertainty remains small


Scenario 2: PARTIAL OCCLUSION
───────────────────────────────
         ╔══════╗
Camera      🚧    ║Object║
   🎥───────▓▓────→║  🔵  ║  Occluder blocks view
         Occluder ╚══════╝
         
Measurement: Invalid or noisy (σ = 0.2m)
Action: Skip update OR use with high R
Result: Uncertainty grows


Scenario 3: FULL OCCLUSION
────────────────────────────
   🎥      ┌─────┐
Camera────▓▓▓▓▓▓▓│
   🔴     │🚧🚧🚧│      ╔══════╗
          │Occl.│      ║Object║
          └─────┘      ║  🔵  ║
                       ╚══════╝
                       
Measurement: None (null)
Action: Prediction only
Result: Uncertainty grows rapidly


FAILURE MODES:
──────────────
Mode 1: 'null'     → No observation (z = None)
Mode 2: 'random'   → Random pose (z = rand())
Mode 3: 'previous' → Last valid pose (z = z_prev)
Mode 4: 'none'     → Noisy observation (z = true + noise)
```

## Uncertainty Evolution

```
┌──────────────────────────────────────────────────────────┐
│              UNCERTAINTY OVER TIME                        │
└──────────────────────────────────────────────────────────┘

Time →

Frame:   0      10     20     30     40     50     60     70
         │      │      │      │      │      │      │      │
         
σ(pos)   │      │      │      │      │▲     │▲     │      │
0.2m     │      │      │      │      ││     ││     │      │
         │      │      │      │      ││     ││     │      │
0.15m    │      │      │      │   ╱─╲││  ╱─╲││     │      │
         │      │      │      │  │   │╲─╯   ││     │      │
0.1m     │      │      │   ╱──┤  │   │      ││  ╱──┤      │
         │      │   ╱──┤  │   │  │   │      │╲─╯   │   ╱──┤
0.05m    │   ╱──┤  │   │  │   │  │   │      │      │  │   │
         ├──╯   └──┘   └──┘   └──┘   └──────┘      └──┘   └──
0m       └─────────────────────────────────────────────────────

Legend:  ─── Normal operation (small σ)
         ╱─╲ Occlusion event (large σ)
         
Effects:
• Process noise: Gradual σ increase
• Good measurement: Sharp σ decrease  
• Occlusion: Rapid σ increase
• Recovery: Gradual σ decrease
```

## Visualization Layout

```
┌─────────────────────────────────────────────────────────────┐
│                  INTERACTIVE DISPLAY                         │
└─────────────────────────────────────────────────────────────┘

┌───────────────────────────────┬─────────────────────────────┐
│                               │                             │
│        3D SCENE VIEW          │     PARTICLE VIEW (PF)      │
│                               │      or 2D VIEW (EKF)       │
│    🟢 Camera                  │                             │
│       ╲                       │    ······                   │
│        ╲  🚧 Occluder        │    ·🟠···  Particles        │
│         ╲  ▓▓                │    ······  (1000)           │
│          ╲ ▓▓                │                             │
│           ▼                   │    Weighted mean: 🟠        │
│         🟠🔵                  │                             │
│     Estimate + True           │    True position: 🔵        │
│                               │                             │
│     Orange ellipsoid =        │                             │
│     Uncertainty (3σ)          │                             │
│                               │                             │
├───────────────────────────────┼─────────────────────────────┤
│                               │                             │
│     TOP-DOWN VIEW (XY)        │    STATISTICS PANEL         │
│                               │                             │
│         Y                     │  Frame: 42/100              │
│         ▲                     │  Occlusion: YES             │
│         │                     │  Factor: 10.0               │
│    🚧───┼───                 │                             │
│    ▓▓   │                     │  Estimated Position:        │
│         │🟠🔵                 │   [0.023, 0.156, 0.487]    │
│         │                     │                             │
│         └────────► X          │  Uncertainty (σ):           │
│                               │   [0.142, 0.156, 0.089]    │
│    Orange ellipse =           │                             │
│    2D uncertainty projection  │  Orientation (quat):        │
│                               │   [0.98, -0.12, 0.08, 0.14]│
└───────────────────────────────┴─────────────────────────────┘
                                
                ┌──────────────────────┐
                │   FRAME SLIDER       │
                │  ◄═══════●═══════►  │
                │  0  42  50      100  │
                └──────────────────────┘
```

## Processing Pipeline

```
┌──────────────────────────────────────────────────────────┐
│                  FRAME PROCESSING                         │
└──────────────────────────────────────────────────────────┘

1. INITIALIZATION (Frame 0)
   ↓
   ┌────────────────────────────────┐
   │ x₀ = first measurement         │
   │ P₀ = initial covariance        │
   └────────────────────────────────┘

2. FOR EACH FRAME t = 1..N:
   ↓
   ┌────────────────────────────────┐
   │ A. MOTION PREDICTION           │
   │    • Apply velocity model      │
   │    • Add process noise         │
   │    • Scale by drift factor     │
   │    Result: x̄(t), P̄(t)          │
   └────────────┬───────────────────┘
                ↓
   ┌────────────────────────────────┐
   │ B. OCCLUSION CHECK             │
   │    • Ray-box intersection      │
   │    • Line of sight test        │
   │    Result: is_occluded         │
   └────────────┬───────────────────┘
                ↓
           ┌────┴────┐
      Yes  │Occluded?│  No
      ┌────┘         └────┐
      ↓                   ↓
   ┌──────────┐      ┌──────────┐
   │C1. HANDLE│      │C2. UPDATE│
   │OCCLUSION │      │WITH MEAS │
   │          │      │          │
   │Based on: │      │• Compute │
   │• null    │      │  innov.  │
   │• random  │      │• Kalman  │
   │• prev.   │      │  gain    │
   │• noise   │      │• Update  │
   └────┬─────┘      └────┬─────┘
        │                 │
        └────────┬────────┘
                 ↓
   ┌────────────────────────────────┐
   │ D. NORMALIZE & STORE           │
   │    • Quaternion normalization  │
   │    • Covariance symmetry       │
   │    • Save x(t), P(t)           │
   └────────────┬───────────────────┘
                ↓
   ┌────────────────────────────────┐
   │ E. VISUALIZE                   │
   │    • Update 3D scene           │
   │    • Draw uncertainty          │
   │    • Show statistics           │
   └────────────────────────────────┘
```

## Performance Characteristics

```
┌──────────────────────────────────────────────────────────┐
│              COMPUTATIONAL COMPLEXITY                     │
└──────────────────────────────────────────────────────────┘

Operation            EKF             Particle Filter
─────────            ───             ───────────────

Prediction          O(n²)            O(N·n)
                    ~0.1 ms          ~5 ms

Update              O(n·m + m³)      O(N·m)
                    ~0.3 ms          ~4 ms

Resampling          N/A              O(N·log N)
                    -                ~1 ms

Total per frame     ~0.5 ms          ~10 ms
                    
Memory              ~10² KB          ~10¹ MB
                    O(n²)            O(N·n)

Real-time (30 FPS)  ✅ Yes           ⚠️ Marginal
                    (33 ms budget)   (33 ms budget)


where:  n = state dimension (13)
        m = measurement dimension (7)
        N = number of particles (1000)
```

---

This architecture provides a **robust, principled approach** to 6D pose estimation with comprehensive uncertainty quantification.

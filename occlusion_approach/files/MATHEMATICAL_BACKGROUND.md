# Mathematical Background

## State Estimation Framework

### State Space Model

**State vector (13D):**
```
x = [px, py, pz, qw, qx, qy, qz, vx, vy, vz, wx, wy, wz]ᵀ
```

Where:
- `p = [px, py, pz]`: Position in 3D space
- `q = [qw, qx, qy, qz]`: Orientation as unit quaternion
- `v = [vx, vy, vz]`: Linear velocity
- `ω = [wx, wy, wz]`: Angular velocity

### Motion Model

**Prediction equations:**
```
p(t+1) = p(t) + v(t)·Δt + w_p
q(t+1) = q(t) ⊗ exp(ω(t)·Δt/2) + w_q
v(t+1) = v(t) + w_v
ω(t+1) = ω(t) + w_ω
```

Where:
- `⊗` denotes quaternion multiplication
- `w_*` are process noise terms
- `Δt` is the time step

**Quaternion exponential:**
```
exp(ω·Δt/2) = [cos(θ/2), sin(θ/2)·ω̂]
```
Where `θ = ||ω||·Δt` and `ω̂ = ω/||ω||`

### Measurement Model

**Direct observation:**
```
z = h(x) + v = [p, q] + v
```

Where:
- `z`: Measurement vector (7D)
- `v`: Measurement noise (Gaussian)
- `R`: Measurement covariance matrix

## Extended Kalman Filter (EKF)

### Algorithm Steps

**1. Prediction Step:**
```
x̄(t) = f(x(t-1))                    [State prediction]
P̄(t) = F·P(t-1)·Fᵀ + Q               [Covariance prediction]
```

Where:
- `F`: Jacobian of motion model
- `Q`: Process noise covariance
- `P`: State covariance matrix

**2. Update Step:**
```
K(t) = P̄(t)·Hᵀ·(H·P̄(t)·Hᵀ + R)⁻¹   [Kalman gain]
x(t) = x̄(t) + K(t)·(z(t) - h(x̄(t)))  [State update]
P(t) = (I - K(t)·H)·P̄(t)              [Covariance update]
```

Where:
- `H`: Jacobian of measurement model
- `R`: Measurement noise covariance
- `K`: Kalman gain matrix

### Jacobian Matrices

**Motion model Jacobian F:**
```
F = ∂f/∂x = [
    I₃    0    Δt·I₃   0
    0     J_q   0       Δt·J_ω
    0     0     I₃      0
    0     0     0       I₃
]
```

Where:
- `I₃`: 3×3 identity matrix
- `J_q`: Quaternion update Jacobian
- `J_ω`: Angular velocity to quaternion Jacobian

**Measurement model Jacobian H:**
```
H = ∂h/∂x = [
    I₃  0  0  0    [Position observation]
    0   I₄  0  0    [Orientation observation]
]
```

### Covariance Matrices

**Process noise Q (13×13):**
```
Q = [
    σ²_p·I₃     0         0        0
    0        σ²_q·I₄      0        0
    0           0      σ²_v·I₃     0
    0           0         0     σ²_ω·I₃
]
```

**Measurement noise R (7×7):**
```
R = [
    σ²_z_p·I₃      0
    0          σ²_z_q·I₄
]
```

Where σ²_* represents variance terms.

## Particle Filter

### Algorithm Steps

**1. Initialization:**
```
For i = 1 to N:
    x⁽ⁱ⁾(0) ~ p(x₀)         [Sample from initial distribution]
    w⁽ⁱ⁾(0) = 1/N            [Uniform weights]
```

**2. Prediction:**
```
For i = 1 to N:
    x⁽ⁱ⁾(t) ~ p(x(t) | x⁽ⁱ⁾(t-1))   [Sample from motion model]
```

**3. Update:**
```
For i = 1 to N:
    w⁽ⁱ⁾(t) = w⁽ⁱ⁾(t-1) · p(z(t) | x⁽ⁱ⁾(t))   [Weight by likelihood]

w⁽ⁱ⁾(t) = w⁽ⁱ⁾(t) / Σⱼ w⁽ʲ⁾(t)                [Normalize]
```

**4. Resampling (when N_eff < N/2):**
```
N_eff = 1 / Σᵢ (w⁽ⁱ⁾)²                       [Effective sample size]

For i = 1 to N:
    Draw j with probability w⁽ʲ⁾
    x⁽ⁱ⁾_new = x⁽ʲ⁾
```

### Likelihood Function

**Position likelihood:**
```
p(z_p | x⁽ⁱ⁾) = N(z_p; p⁽ⁱ⁾, σ²_p·I₃)
              = exp(-||z_p - p⁽ⁱ⁾||²/(2σ²_p))
```

**Orientation likelihood:**
```
p(z_q | x⁽ⁱ⁾) = exp(-d_q(z_q, q⁽ⁱ⁾)²/(2σ²_q))
```

Where `d_q(q₁, q₂) = 2·arccos(|q₁·q₂|)` is the quaternion distance.

**Combined likelihood:**
```
p(z | x⁽ⁱ⁾) = p(z_p | x⁽ⁱ⁾) · p(z_q | x⁽ⁱ⁾)
```

### State Estimation

**Weighted mean:**
```
x̂(t) = Σᵢ w⁽ⁱ⁾(t) · x⁽ⁱ⁾(t)
```

**Covariance:**
```
P(t) = Σᵢ w⁽ⁱ⁾(t) · (x⁽ⁱ⁾(t) - x̂(t)) · (x⁽ⁱ⁾(t) - x̂(t))ᵀ
```

## Quaternion Mathematics

### Quaternion Representation
```
q = [qw, qx, qy, qz] = [cos(θ/2), sin(θ/2)·n̂]
```

Where:
- `θ`: Rotation angle
- `n̂`: Rotation axis (unit vector)

### Quaternion Multiplication
```
q₁ ⊗ q₂ = [
    w₁w₂ - x₁x₂ - y₁y₂ - z₁z₂,
    w₁x₂ + x₁w₂ + y₁z₂ - z₁y₂,
    w₁y₂ - x₁z₂ + y₁w₂ + z₁x₂,
    w₁z₂ + x₁y₂ - y₁x₂ + z₁w₂
]
```

### Quaternion Conjugate
```
q* = [qw, -qx, -qy, -qz]
```

### Quaternion Inverse
```
q⁻¹ = q* / ||q||²
```

For unit quaternions: `q⁻¹ = q*`

### Angular Distance
```
d(q₁, q₂) = 2·arccos(|q₁·q₂|)
```

## Uncertainty Modeling

### Drift Model

**Time-varying process noise:**
```
Q(t) = Q₀ · (1 + α·t/T)
```

Where:
- `Q₀`: Base process noise
- `α`: Drift coefficient (typically 0.5)
- `t/T`: Normalized time

### Occlusion Model

**Measurement noise during occlusion:**
```
R_occluded = β · R_normal
```

Where `β ≥ 10` is the occlusion factor.

**Prediction-only mode:**
When `z(t) = null`:
- Skip update step
- Uncertainty grows: `P(t) = P̄(t)`

## Uncertainty Quantification

### Confidence Ellipsoids

**3-sigma bound (99.7% confidence):**
```
E = {x : (x - x̂)ᵀ P⁻¹ (x - x̂) ≤ 9}
```

**Eigenvalue decomposition:**
```
P = V·Λ·Vᵀ
```

Where:
- `V`: Eigenvectors (ellipsoid axes)
- `Λ`: Eigenvalues (variance along axes)

**Ellipsoid parameters:**
```
Semi-axes lengths: aᵢ = 3·√λᵢ
Orientation: V
```

### Mahalanobis Distance

**Statistical distance:**
```
d_M(x, x̂) = √((x - x̂)ᵀ P⁻¹ (x - x̂))
```

Used for:
- Outlier detection
- Consistency checking
- Innovation validation

## Performance Metrics

### Root Mean Square Error (RMSE)
```
RMSE = √(1/N · Σᵢ ||x̂ᵢ - xᵢ||²)
```

### Maximum Error
```
Max Error = max_i ||x̂ᵢ - xᵢ||
```

### Normalized Estimation Error Squared (NEES)
```
NEES(t) = (x(t) - x̂(t))ᵀ P(t)⁻¹ (x(t) - x̂(t))
```

Expected value: `E[NEES] = n` (state dimension)

### Computational Complexity

**EKF:**
- Prediction: O(n²)
- Update: O(n²·m + m³)
- Total per frame: O(n²)

Where n = state dimension, m = measurement dimension

**Particle Filter:**
- Prediction: O(N·n)
- Update: O(N·m)
- Resampling: O(N·log N)
- Total per frame: O(N·n)

Where N = number of particles

## Comparison Table

| Property | EKF | Particle Filter |
|----------|-----|-----------------|
| **Noise assumption** | Gaussian | Any distribution |
| **Linearity** | Requires linearization | None required |
| **Multimodal** | No | Yes |
| **Complexity** | O(n²) | O(N·n) |
| **Memory** | O(n²) | O(N·n) |
| **Accuracy** | High (if Gaussian) | Very high |
| **Real-time** | Yes | Depends on N |

## Implementation Details

### Numerical Stability

**Quaternion normalization:**
```python
q = q / (||q|| + ε)
```
Where `ε = 1e-8` prevents division by zero.

**Covariance symmetry:**
```python
P = (P + Pᵀ) / 2
```
Ensures numerical symmetry.

**Positive definiteness:**
```python
P = P + ε·I
```
Adds small diagonal term for stability.

### Adaptive Techniques

**Dynamic noise scaling:**
```python
Q_adaptive = Q_base · (1 + σ_innovation / σ_threshold)
```

**Outlier rejection:**
```python
if ||z - h(x̂)|| > 3·√(H·P·Hᵀ + R):
    Skip measurement
```

## References

### Books
1. **Thrun, S., Burgard, W., & Fox, D. (2005)**
   "Probabilistic Robotics"
   MIT Press

2. **Särkkä, S. (2013)**
   "Bayesian Filtering and Smoothing"
   Cambridge University Press

3. **Bar-Shalom, Y., Li, X. R., & Kirubarajan, T. (2001)**
   "Estimation with Applications to Tracking and Navigation"
   Wiley

### Papers
1. **Julier, S. J., & Uhlmann, J. K. (2004)**
   "Unscented filtering and nonlinear estimation"
   Proceedings of the IEEE, 92(3), 401-422

2. **Arulampalam, M. S., et al. (2002)**
   "A tutorial on particle filters for online nonlinear/non-Gaussian Bayesian tracking"
   IEEE Transactions on Signal Processing, 50(2), 174-188

3. **Shuster, M. D. (1993)**
   "A survey of attitude representations"
   Journal of the Astronautical Sciences, 41(4), 439-517

### Implementation
- **NumPy**: Numerical computations
- **SciPy**: Rotation handling
- **Matplotlib**: Visualization

## Notation Summary

| Symbol | Description |
|--------|-------------|
| `x` | State vector |
| `z` | Measurement vector |
| `P` | Covariance matrix |
| `Q` | Process noise covariance |
| `R` | Measurement noise covariance |
| `F` | State transition Jacobian |
| `H` | Measurement Jacobian |
| `K` | Kalman gain |
| `N(μ, Σ)` | Gaussian distribution |
| `⊗` | Quaternion multiplication |
| `I_n` | n×n identity matrix |
| `||·||` | Euclidean norm |
| `·ᵀ` | Transpose |
| `·⁻¹` | Matrix inverse |

---

This mathematical framework ensures robust, principled state estimation with quantified uncertainty.

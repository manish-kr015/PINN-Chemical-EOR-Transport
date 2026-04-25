# Physics-Informed Neural Networks for Chemical EOR Transport
### Physics Informed Neural Network Approach to  Simulate Chemical Transport for EOR Applications.

---

## 📌 Project Overview

This project develops a **Physics-Informed Neural Network (PINN)** model to simulate 
how chemicals like **surfactants and polymers** move through porous media during 
**Enhanced Oil Recovery (EOR)** operations.

The model solves the **1D Advection-Dispersion Equation (ADE)** — the governing 
equation for chemical transport in porous media — using a neural network that learns 
directly from physics instead of experimental data.

> **Key Highlight:** The neural network is trained using the physics equation itself 
> as the training signal — no labeled experimental data is needed.

---

## 🛢️ Background — Why This Project?

### What is Enhanced Oil Recovery (EOR)?
In oil fields, after primary and secondary recovery, a large amount of oil still 
remains trapped in the reservoir. **Chemical EOR** involves injecting chemicals 
like surfactants and polymers to:
- Reduce interfacial tension between oil and water
- Improve sweep efficiency
- Mobilize trapped oil

### The Problem
When chemicals are injected into a reservoir, two things happen simultaneously:
- **Advection** — chemical is carried forward by flowing water
- **Dispersion** — chemical spreads and mixes as it moves

Understanding how concentration changes over **time and distance** is critical 
for designing effective EOR operations.

### Why PINNs?
Traditional analytical solutions exist only for simple ideal cases. In real 
reservoirs with complex geometry and heterogeneous properties — no analytical 
solution exists. PINNs solve this problem by:

| Feature | Traditional Methods | PINNs |
|---|---|---|
| Experimental data needed | Sometimes | ❌ Not needed |
| Mesh/Grid required | ✅ Yes | ❌ No |
| Fixed time step (Δt) | ✅ Required | ❌ Not required |
| Complex geometry | Difficult | ✅ Easy |
| Inverse problems | ❌ Cannot | ✅ Can |
| Real time prediction | ❌ Slow | ✅ Instant after training |

---

## 📐 Governing Equation — 1D Advection Dispersion Equation (ADE)

$$R \frac{\partial c}{\partial t} = D \frac{\partial^2 c}{\partial x^2} - v \frac{\partial c}{\partial x}$$

| Symbol | Parameter | Description |
|---|---|---|
| c(x,t) | Concentration | Chemical concentration at position x and time t |
| R | Retardation factor | How much soil slows the chemical |
| D | Dispersion coefficient | How much chemical spreads |
| v | Velocity | How fast water carries chemical |
| x | Distance | Along column from 0 to L |
| t | Time | From 0 to t₀ |

---

## 🔲 Initial and Boundary Conditions

### Initial Condition
$$c(x, 0) = C_i$$
At time zero, the column has uniform background concentration Cᵢ (clean column = 0)

### Inlet Boundary Condition — Dirichlet
$$c(0, t) = C_0 \quad \text{for } 0 < t \leq t_0$$
At inlet x=0, chemical is injected at concentration C₀ for duration t₀

### Outlet Boundary Condition — Neumann
$$\frac{\partial c}{\partial x}(L, t) = 0$$
At outlet x=L, zero concentration gradient — chemical exits freely by flow

---

## 📊 Analytical Solution — Cleary and Adrian (1973)

For **0 < t ≤ t₀** (injection phase):

$$c(x,t) = C_i + (C_0 - C_i) \cdot A(x,t)$$

For **t > t₀** (post injection phase):

$$c(x,t) = C_i + (C_0 - C_i) \cdot A(x,t) - C_0 \cdot A(x, t-t_0)$$

Where A(x,t) is:

$$A(x,t) = \frac{1}{2} \text{erfc}\left[\frac{Rx-vt}{2\sqrt{DRt}}\right] + \frac{1}{2}e^{vx/D} \text{erfc}\left[\frac{Rx+vt}{2\sqrt{DRt}}\right]$$

$$+ \frac{1}{2}\left[2 + \frac{v(2L-x)}{D} + \frac{v^2t}{DR}\right] e^{vL/D} \text{erfc}\left[\frac{R(2L-x)+vt}{2\sqrt{DRt}}\right]$$

$$- \left(\frac{v^2t}{\pi DR}\right)^{1/2} \exp\left[\frac{vL}{D} - \frac{R}{4Dt}\left(2L-x+\frac{vt}{R}\right)^2\right]$$

> This analytical solution is used **only for validation** of the PINN model.
> In real EOR problems, no analytical solution exists — that is exactly why we use PINNs.

---

## 🧠 PINN Methodology

### What is a PINN?
A Physics-Informed Neural Network embeds physical laws directly into the 
neural network training process through a custom loss function.

```
Total Loss = PDE Loss + IC Loss + BC Loss

PDE Loss  = mean( R·∂c/∂t - D·∂²c/∂x² + v·∂c/∂x )²
IC  Loss  = mean( c(x,0) - Cᵢ )²
BC1 Loss  = mean( c(0,t) - C₀ )²
BC2 Loss  = mean( ∂c/∂x(L,t) )²

When Total Loss → 0 :
  ✓ PDE satisfied everywhere
  ✓ Initial condition satisfied
  ✓ Boundary conditions satisfied
```

### Training Flow
```
Step 1 → Sample random (x,t) collocation points in domain
Step 2 → Pass through Neural Network → predict c(x,t)
Step 3 → Use autograd to compute ∂c/∂t, ∂c/∂x, ∂²c/∂x²
Step 4 → Compute PDE residual
Step 5 → Compute IC and BC losses
Step 6 → Total Loss = PDE + IC + BC1 + BC2
Step 7 → Backpropagate → update network weights
Step 8 → Repeat until loss converges to zero
```

---

## 🏗️ Neural Network Architecture

```
Input Layer  : [x, t]          → 2 neurons
Hidden Layer1: 64 neurons      → tanh activation
Hidden Layer2: 64 neurons      → tanh activation
Hidden Layer3: 64 neurons      → tanh activation
Hidden Layer4: 64 neurons      → tanh activation
Output Layer : [c(x,t)]        → 1 neuron
```

**Why tanh?**
PINNs require computing second derivatives of the network output.
ReLU has zero second derivative — useless for the ∂²c/∂x² term.
Tanh is smooth with non-zero higher derivatives — ideal for PDEs.

---

## ⚙️ Training Strategy

| Phase | Optimizer | Iterations | Purpose |
|---|---|---|---|
| Phase 1 | Adam | 8000 | Fast convergence |
| Phase 2 | L-BFGS | Auto | High precision fine-tuning |

**Why two optimizers?**
Adam is fast but gets stuck near the solution without converging precisely.
L-BFGS uses second-order curvature information to find the exact minimum.

---

## 📋 Parameters Used

| Parameter | Symbol | Value |
|---|---|---|
| Retardation factor | R | 1.0 |
| Dispersion coefficient | D | 0.1 |
| Velocity | v | 1.0 |
| Inlet concentration | C₀ | 1.0 |
| Initial concentration | Cᵢ | 0.0 |
| Column length | L | 1.0 |
| Pulse end time | t₀ | 1.0 |

---

## 📊 Collocation Points — 70:30 Split

| Type | Points | Purpose |
|---|---|---|
| Interior (PDE) | 3000 | Enforce physics equation |
| Boundary | 300 | Enforce BC conditions |
| Initial | 300 | Enforce IC condition |
| **Training Total** | **3600** | **~70%** |
| Test points | 1000 | Verify generalization (~30%) |

---

## ✅ Model Verification

### Verification 1 — Error Metrics (MAE and RMSE)
Compare PINN predictions against Cleary & Adrian (1973) analytical solution

### Verification 2 — PDE Residual Check
*(Works even when NO analytical solution exists)*

Sample 2000 random points in domain and check:
```
Mean |PDE Residual| < 0.01  →  Physics satisfied ✓
```
If residual is near zero everywhere — model is correct
without needing any analytical formula.

### Verification 3 — Mass Conservation Check
*(Works even when NO analytical solution exists)*

Integrate concentration over column at each time step:
```
Mass = ∫ c(x,t) dx  from 0 to L
```
Mass must increase smoothly during injection — 
confirms physical consistency of the solution.

---

## 📈 Results

| Time | MAE | RMSE |
|---|---|---|
| t = 0.2 | < 0.005 | < 0.005 |
| t = 0.4 | < 0.005 | < 0.005 |
| t = 0.6 | < 0.005 | < 0.005 |
| t = 0.8 | < 0.005 | < 0.005 |
| t = 1.0 | < 0.005 | < 0.005 |

> PINN predictions closely match analytical solution at all time snapshots.

---

## 📁 Repository Structure

```
📦 PINN-Chemical-EOR-Transport
 ┣ 📂 code
 ┃ ┣ 📜 ADE_PINN.py              → Main PINN model
 ┃ ┣ 📜 analytical_solution.py   → Cleary & Adrian 1973
 ┃ ┗ 📜 verification.py          → All verification methods
 ┣ 📂 results
 ┃ ┣ 🖼️ PINN_vs_Analytical.png   → Comparison plots
 ┃ ┣ 🖼️ PDE_Residual.png         → Residual check plot
 ┃ ┣ 🖼️ Mass_Conservation.png    → Mass conservation plot
 ┣ 📂 docs
 ┃ ┗ 📜 equations_reference.pdf  → Governing equations reference
 ┣ 📜 README.md
 ┗ 📜 requirements.txt
```

---

## 🚀 How to Run

### Install Dependencies
```bash
pip install deepxde torch numpy matplotlib scipy
```

### Run the Model
```bash
python code/ADE_PINN.py
```

### Requirements File (requirements.txt)
```
deepxde>=1.9.0
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.4.0
scipy>=1.7.0
```

---

## 🔮 Future Work

- [ ] Extend model to **t > t₀** (post injection phase)
- [ ] Add **breakthrough curve** analysis
- [ ] Extend to **2D heterogeneous reservoir**
- [ ] Solve **coupled nonlinear water flooding equations**
- [ ] **Inverse modeling** — identify unknown D, v, R from measurements

---

## 📚 References

1. Cleary, R.W. and Adrian, D.D. (1973). *Analytical solution of the 
   convective-dispersive equation for cation adsorption in soils.* 
   Soil Science Society of America Journal.

2. Raissi, M., Perdikaris, P., & Karniadakis, G.E. (2019). 
   *Physics-informed neural networks: A deep learning framework for 
   solving forward and inverse problems involving nonlinear partial 
   differential equations.* Journal of Computational Physics.

3. Lu, L., et al. (2021). *DeepXDE: A deep learning library for 
   solving differential equations.* SIAM Review.

4. Green, D.W. and Willhite, G.P. (2018). 
   *Enhanced Oil Recovery.* SPE Textbook Series.

---

## 👨‍💻 Author

**Manish Kumar**
Final Year, Petroleum Engineering
Indian Institute of Petroleum and Energy, Visakhapatnam

---

## 📄 License

This project is developed as part of final year academic project at IIPE.

---

> *"PINNs are not competing with analytical solutions —
>  they are built for the cases where analytical solutions do not exist."*

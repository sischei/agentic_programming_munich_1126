# Session log, `ode_solver/`

## Goal

A self-contained PyTorch PINN sandbox for the "Agentic Programming"
lecture, with two parallel examples, a 1D ODE and a 2D PDE, that share the
same `MLP` + `torch.compile` + autograd-grad pattern so readers can diff the
two scripts and see what changes from ODE → PDE.

## Problems

**ODE** (`solver.py`, `plot.py`):

```
y''(x) + y(x) = 0,   x ∈ [0, 2π],   y(0) = 1,  y'(0) = 0
y_exact(x)    = cos(x)
```

**PDE** (`solver_pde.py`, `plot_pde.py`): 2D Poisson on `Ω = (0,1)²` with the
manufactured solution

```
-Δu(x, y) = f(x, y),   u = 0 on ∂Ω
u_exact(x, y) = sin(π x) sin(π y)
f(x, y)       = 2 π²   sin(π x) sin(π y)
```

Two BC strategies are wired into `solver_pde.py`:

- **soft BC (default)**: penalty `(u_bd)²` with weight `--lam-bc 1.0`. Plateau
  near `Linf ≈ 2.5e-2` is the well-known soft-BC failure mode and is the point
  the lecture demonstrates.
- **hard BC** (`--hard-bc`): ansatz `u(x,y) = x(1−x)y(1−y)·NN(x,y)`. Boundary
  values are zero by construction, so the BC term vanishes and `Linf` drops
  by about two orders of magnitude (2.5e-2 to 2.3e-4).

## Files in this folder

| File             | Purpose                                                       |
|------------------|---------------------------------------------------------------|
| `solver.py`      | Train the 1D ODE PINN; save `model.pt`                        |
| `plot.py`        | Load `model.pt`; overlay PINN vs `cos(x)`; save `solution.png`|
| `solver_pde.py`  | Train the 2D Poisson PINN (soft or hard BC); save `model_pde.pt` |
| `plot_pde.py`    | Load `model_pde.pt`; render exact / pred / error heatmaps; save `solution_pde.png` |
| `model.pt`       | Trained ODE weights (15k epochs, seed 0)                      |
| `model_pde.pt`   | Trained PDE weights, *inner MLP only*, so the same file works for both BC modes |
| `solution.png`   | ODE plot                                                      |
| `solution_pde.png` | PDE three-panel plot                                        |
| `solution_pde_hardbc.png` | The same plot from the hard-BC weights: no boundary halo |
| `train_ode.log`, `train_pde.log` | Full training logs (not shipped in this repo) |
| `plan_pde.md`    | Brief and reconciled targets for the PDE extension            |
| `.gitignore`     | Excludes `__pycache__/`, `*.pyc`, `*.log` (not shipped)       |

## Architecture

Shared components:
- `MLP`: `tanh` hidden layers, scalar output (ODE: 32×3, PDE: 64×4).
- Per-step JIT: `torch.compile(step, backend="eager", fullgraph=False)`. The
  default `aot_autograd` backend in PyTorch 2.x does not support the
  double-backward needed for PINN losses. (Hand-written PINN
  notebooks that do not compile the step never meet this.)
- Xavier uniform init by default (`--init xavier`); pass `--init default` for
  the framework default.
- Optional cosine LR schedule (`--lr-schedule cosine`), gradient clipping
  (`--clip-grad`), and an L-BFGS finetune stage (`--lbfgs-steps`).

ODE-specific:
- 200 collocation points uniform in `[0, 2π]` (fixed grid).
- Loss = MSE residual + `(y(0) − 1)² + (y'(0))²`.

PDE-specific:
- Interior collocation: 2048 uniform random samples in `(0,1)²`, redrawn each
  epoch (good teaching point on stochastic collocation).
- Boundary: 128 points per side, redrawn each epoch.
- Loss = `((-Δu) − f)²` + (soft-BC only) `λ_bc · (u_bd)²`.
- Eval grid: fixed `128 × 128`; reports `Linf` and discrete `L²`. The `L²`
  formula `√mean(err²)` already matches `‖·‖_{L²(Ω)}` because `|Ω| = 1`; on a
  non-unit domain you would multiply by `√area`.

The ODE script defaults to 15k epochs and the PDE to 20k; the PDE problem is
the harder of the two (2D, non-trivial BCs) so it gets the longer schedule.

## CLI

```bash
# 1D ODE
python3 solver.py [--epochs 15000] [--n-collocation 200] [--seed 0] \
                  [--lr 1e-3] [--lr-schedule none|cosine] [--clip-grad 0] \
                  [--init xavier|default] [--lbfgs-steps 0] \
                  [--device cpu] [--dtype float32|float64] [--eval-freq 500]
python3 plot.py

# 2D Poisson (soft BC, default)
python3 solver_pde.py [--epochs 20000] [--n-int 2048] [--n-bd 512] [--seed 0] \
                      [--lam-bc 1.0] [--lr 1e-3] [--lr-schedule none|cosine] \
                      [--clip-grad 0] [--init xavier|default] [--lbfgs-steps 0] \
                      [--device cpu] [--dtype float32|float64] [--eval-freq 500]
python3 plot_pde.py

# 2D Poisson with hard-BC ansatz upgrade (boundary exact by construction)
python3 solver_pde.py --hard-bc
python3 plot_pde.py   --hard-bc
```

`plot_pde.py` mirrors the `--hard-bc` flag because the saved `model_pde.pt`
contains only the inner-MLP weights, the wrapper is reapplied at load time.

## Verified results (seed 0)

ODE, default `solver.py`:
- final ODE residual ≈ `5.7e-7`
- final IC loss ≈ `1e-10`
- final `Linf` vs `cos(x)` ≈ `2.0e-4`

PDE, default `solver_pde.py` (soft BC, 20k epochs):
- final `Linf ≈ 2.5e-2`,  `L² ≈ 6.6e-3`  (the soft-BC plateau)

PDE, `solver_pde.py --hard-bc` (20k epochs, run 2026-09-09):
- final `Linf ≈ 2.3e-4`,  `L² ≈ 9.3e-5`; the hard-BC ansatz removes the
  boundary halo entirely (see `solution_pde_hardbc.png`).
- `model_pde.pt` now holds these hard-BC weights; `plot_pde.py --hard-bc`
  reproduces the figure from them. `solution_pde.png` is the soft-BC figure
  from the original session, kept for the comparison.

## How to restore / resume

1. `cd code/extras/ode_solver`
2. `python3 plot.py` and `python3 plot_pde.py`, regenerate figures from the
   cached weights (no retraining).
3. `python3 solver.py` and `python3 solver_pde.py`, retrain from scratch
   (≈ 1 min ODE, ≈ 5 min PDE on a laptop CPU).

## Tweak ideas

- swap activation (`tanh` → `sin` / SIREN) and watch convergence speed
- hard-encode ICs in the ODE: `y_hat(x) = 1 + x² · NN(x)`, IC loss vanishes
  (analogous to `--hard-bc` for the PDE)
- extend the ODE domain to `[0, 4π]` or `[0, 8π]` and watch the error grow
  with frequency
- swap ODE: `y'' + 4y = 0, y(0)=1, y'(0)=0` → exact `cos(2x)`
- swap PDE forcing for a non-symmetric manufactured solution
  (e.g. `u = sin(πx) · sin(2π y)`) and update `f` accordingly
- compare soft BC vs hard BC vs Lagrangian-augmented BC on the same problem

## Gotchas hit during the session

1. **`requires_grad_(True)` inside the compiled step** raised
   `RuntimeError: element 0 of tensors does not require grad`. Fix: set
   `requires_grad` once on the persistent leaf tensor outside `step()` (ODE).
2. **Default `torch.compile` backend cannot double-backward.** Fix:
   `backend="eager", fullgraph=False`.
3. **Soft-BC plateau** at `Linf ≈ 2.5e-2` (PDE), not a bug, the textbook
   failure mode. Use `--hard-bc` to remove it.

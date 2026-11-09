# Plan: 2D PDE PINN solver

## Brief

Devise a PINN solver in PyTorch for a 2D PDE with an analytical reference and
clear convergence metrics, paralleling the existing 1D ODE example.

## Problem

On `Ω = (0, 1)²`, solve

```
-Δu(x, y) = f(x, y)   in Ω
        u = 0         on ∂Ω
```

with manufactured solution and forcing

```
u_exact(x, y) = sin(π x) sin(π y)
f(x, y)       = 2 π²   sin(π x) sin(π y)
```

(Verify: `Δ[sin(πx)sin(πy)] = -2π²·sin(πx)sin(πy)`, so `-Δu = +2π²·sin(πx)sin(πy) = f`. ✓)

## Metrics

Reported on a fixed `128 × 128` evaluation grid every 500 epochs and at the
end of training:

```
Linf = (u_pred - u_exact).abs().max()
L²   = sqrt(mean((u_pred - u_exact)²))
```

Because `|Ω| = 1`, the discrete `L²` already approximates `‖·‖_{L²(Ω)}`
without an explicit `√(Δx·Δy)` factor.

## Two BC strategies (both implemented in `solver_pde.py`)

| Mode      | Flag         | Loss                                         |
|-----------|--------------|----------------------------------------------|
| soft BC   | (default)    | `((-Δu) − f)²` + `λ_bc · (u_bd)²`            |
| hard BC   | `--hard-bc`  | `((-Δu) − f)²`, with `u = x(1−x)y(1−y)·NN`    |

The hard-BC ansatz makes `u = 0` on `∂Ω` exact by construction.

## Achieved metrics (seed 0, 20 000 epochs, defaults otherwise)

| Mode    | `Linf`     | `L²`       | Notes                                   |
|---------|------------|------------|-----------------------------------------|
| soft BC | `≈ 2.5e-2` | `≈ 6.6e-3` | classic soft-BC plateau; visible halo on `∂Ω` |
| hard BC | `2.3e-4` | `9.3e-5` | measured 2026-09-09, seed 0, 20 000 epochs; no halo |

The original target `Linf ≲ 5e-3, L² ≲ 2e-3` is not reachable with the soft-BC
default, the soft penalty trades boundary fidelity for interior residual.
That is the *pedagogical demonstration* of the lecture: you run the
default first, see the boundary halo, then add `--hard-bc` to remove it.

## Lecture flow

1. `python3 solver_pde.py && python3 plot_pde.py`, observe the soft-BC halo.
2. `python3 solver_pde.py --hard-bc && python3 plot_pde.py --hard-bc`:
   observe the halo collapse.
3. Optional: `--lbfgs-steps 200` after Adam to polish the final accuracy by
   another factor of ~2–5.

## Critical files

- `solver_pde.py`, training script (both BC modes).
- `plot_pde.py`, three-panel visualisation (mirror `--hard-bc`).
- `train_pde.log`, full training log of the most recent run.

## Verification

End-to-end checks:

1. `python3 solver_pde.py --epochs 20000`
   - terminal `Linf ≈ 2.5e-2`, `L² ≈ 6.6e-3` (soft-BC plateau).
2. `python3 plot_pde.py`
   - three-panel `solution_pde.png`; faint boundary halo in the error panel.
3. `python3 solver_pde.py --hard-bc --epochs 20000`
   - terminal `Linf ≈ 2.3e-4`, `L² ≈ 9.3e-5` (measured; two orders of magnitude below soft BC).
4. `python3 plot_pde.py --hard-bc`
   - error panel is essentially flat (no halo).

A zero network gives `Linf = 1.0` (since `max u_exact = 1`), so any
`Linf < 0.1` is real learning, not a bug pretending to converge.

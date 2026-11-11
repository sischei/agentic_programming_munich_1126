# Extras: material from the previous version of this lecture

Not referred to by any slide. Kept for readers who want the earlier `/goal` demo and
plan-reconciliation example, which used physics-informed neural networks (PINNs) on
differential equations with known solutions instead of the DEQN lead example.

| Folder | What it is | Run |
|---|---|---|
| `pinn_loop/` | The former `/goal` task: a PINN for u'' + u = 0 on [0, pi/2] with hard boundary conditions, three tests, the stub, the baseline that set the threshold | `cd pinn_loop && python3 -m pytest -q` (fails with NotImplementedError until `pinn_hard_bc.py` is written) |
| `pinn_loop_solution/` | A solution that passes the three tests (max error 6.7e-5, about 40 s) | copy `pinn_hard_bc.py` into `pinn_loop/` |
| `ode_solver/` | The agent-written plan (`plan_pde.md`), session notes (`session.md`), solver and plots for a 1-D ODE and a 2-D Poisson equation with soft and hard boundary conditions | `python3 plot_pde.py`, `python3 plot_pde.py --hard-bc` |
| `figures/` | The three figures those slides used | |

All of it needs only numpy, matplotlib, torch and pytest, as listed in `requirements.txt`.

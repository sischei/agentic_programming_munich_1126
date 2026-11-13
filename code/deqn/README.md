# The DEQN lead example

A deep equilibrium net (DEQN; Azinovic, Gaegauf and Scheidegger, 2022) replaces the
spline of the reference solver `brock_mirman_reference.py` by a small neural network
for the savings rate, and the grid iteration by gradient descent on the mean squared
Euler residual at sampled states. The lecture builds one step by step with the agent,
checking every step against a known answer. This folder holds our reference versions
of what the agent is asked to produce, the tests the `/goal` demo works against, and
the artifacts the slides show. Source material: the course notebooks in `source/`.

| Step | Lecture | File | Checked against | Run time |
|---|---|---|---|---|
| 1 | Part III | `deterministic.py`: the TF notebook ported to PyTorch, delta = 1 | closed form s* = alpha beta = 0.3564 | 3 s |
| 2-4 | Parts IV, VII | `stochastic.py` (the stub the `/goal` fills), `solution/deqn_stochastic.py` (our answer) | reference policy `solve(0.96, 0.9)` | 11 s |
| 3 | Part VI | `accuracy.py`: the on-path check (also the `/deqn-accuracy` skill) | closed form for the Step-1 policy (the live demo); reference policy otherwise | 5 s |
| 4 | Part VII | `tests/`, `reference/`: five tests and where their thresholds come from | as above | 14 s |
| 5-6 | Part VIII | `plan_deqn.md`, `stochastic_box.py`, `silent_failure.md`, `plot_silent_failure.py` | as above | 11 s |
| 7 | Part VIII | `loss_kernels.py`: four loss kernels, one table and one figure | as above | 45 s |

Everything runs on a laptop CPU in float64 with fixed seeds (`common.py` pins 4 threads).

## Measured numbers (this machine, 2026-11-11, torch 2.2.1)

| Script | Training loss (last 100 steps) | On-path mean \|r\| | max \|s − s_ref\| | E[s] | Simulated log K |
|---|---|---|---|---|---|
| `deterministic.py` | 3.1e-7 | 4.7e-4 (grid) | 4.9e-4 vs alpha beta | 0.3564 | path from K = 0.5 stays in [0.1, 1] |
| `solution/deqn_stochastic.py` | 6.0e-7 | 1.4e-4 | 2.3e-3 vs reference | 0.2523 (reference 0.2518) | [1.08, 1.81], window [0.96, 1.96] |
| `stochastic_box.py` (no bias init) | **2.9e-7** | **9.2e-2** | 0.60 (on the reference solver's path) | **0.937** | **[3.14, 3.72]: outside** |

The last row is the silent failure of Part VIII: the lower training loss belongs to the
wrong solution, the over-saving branch of the Euler equation (`silent_failure.md`).
`accuracy.py` says ACCURATE / ACCURATE / WRONG for the three. Step 1 uses the default
initialization: the closed form enters only the check.

`loss_kernels.py` (MSE, MAE, Huber, log-cosh, same recipe): all four give E[s] within
3e-4 of each other and on-path mean |r| between 1.0e-4 and 1.7e-4. The training-loss
column cannot rank them: the four losses are in different units (MAE in |r|, the others
in r^2; at these residual sizes Huber and log-cosh are rescaled MSE). Only the on-path
columns compare them.

## Layout

```
common.py            shared pieces, shipped to the demo project as deqn/common.py: parameters,
                     steady state, Gauss-Hermite nodes, make_net(s_init=...), encode_state,
                     save/load a policy with a JSON sidecar, simulate_path, euler_residual_np.
                     The numpy residual is for checking; the torch residual and the training
                     loop are what the agent writes.
stochastic.py        the stub (raises NotImplementedError); the /goal target; copied to deqn/stochastic.py
source/              the two course notebooks flattened to scripts (TensorFlow; not run here)
deterministic.py     Step 1; writes outputs/deqn_deterministic.{pt,json} and deqn_step1_figure.pdf
solution/            our stochastic solver; the residual function is the code on the Part VII slides
stochastic_box.py    the same solver with s_init=None; writes outputs/deqn_stochastic_box.{pt,json}
accuracy.py          the on-path check and, with --plot, its three-panel figure (slides/figures/
                     agentic_deqn_verify_*.pdf); copied into .claude/skills/deqn-accuracy/ by the setup script
tests/               conftest.py (an independent numpy oracle) and the five tests
reference/           make_reference.py, policy_rho090.csv, baseline.md
loss_kernels.py      writes outputs/deqn_kernel_table.tex and deqn_kernel_figure.pdf
plot_silent_failure.py   writes slides/figures/agentic_deqn_silent_{bad,good}.pdf
plan_deqn.md         the plan of Step 4 as shown on the Part VIII slide, with the reconciliation
silent_failure.md    what went wrong in stochastic_box.py and how it was found
outputs/             the committed artifacts the slides show
```

Run from `code/` (the project root of the repository copy): `python3 deqn/deterministic.py`,
`python3 deqn/solution/deqn_stochastic.py`, `python3 -m pytest -q deqn/tests` (fails with
NotImplementedError until `deqn/stochastic.py` is implemented; copy the solution over it to
see 5 passed).

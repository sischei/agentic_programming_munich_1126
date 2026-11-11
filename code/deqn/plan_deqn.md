# Plan: a DEQN for the stochastic Brock-Mirman model (Step 4)

*Written for the lecture in the form plan mode produces; the numbers under "Achieved"
are measured, see `reference/baseline.md`. The "First implementation" paragraph is the
reconciliation: what the plan said, what the run showed, what changed.*

## Brief
Implement `deqn/stochastic.py` so that the five tests in `tests/` pass. The tests are
the specification: accuracy on the path the network's own policy generates, and
against the reference solver's policy. The training loss is not consulted.

## Problem
Household chooses the savings rate s in the stochastic growth model
Y = z K^alpha, C = (1 - s) Y, K' = (1 - delta) K + s Y, log z' = rho log z + sigma eps,
alpha 0.36, delta 0.10, sigma 0.04, beta 0.96, rho 0.9 (`deqn.common.PARAMS["stochastic"]`).
Equilibrium condition (log utility): 1/C = beta E[(1 - delta + alpha z' K'^(alpha-1)) / C'].
Residual used as the loss: r = beta C E[(1 - delta + R') / C'] - 1, a fraction of
consumption. Known answer: the reference policy, `brock_mirman_reference.solve(0.96, 0.9)`.

## Recipe
- State transform x = (log z / sd_z, log K - log K*), `common.encode_state`.
- Network 2-32-32-1 tanh, sigmoid output, `common.make_net(2)` with the course
  notebook's initialisation (the untrained policy saves about half of output).
- Expectation: 5 Gauss-Hermite nodes, `common.gh_nodes(5)`.
- Training states: 128 fresh draws per step, log z ~ N(0, sd_z), log K ~ U(log K* +- 0.5).
- Adam 1e-3, 2000 steps, seed 42 (`torch.manual_seed` before the network), float64.
- `solve(seed, steps)` returns {"s", "net", "wall"} and saves results/deqn_stochastic.pt
  with its sidecar via `common.save_policy`.

## Metrics and targets (from `reference/baseline.md`)
| Metric | Target |
|---|---|
| mean \|r\| on the network's own path (2,000 periods, seed 7) | < 1e-3 |
| max \|s - s_ref\| on the reference states | < 1e-2 |
| \|E[s] - 0.2521\| | < 5e-3 |
| own-path log K inside [0.96, 1.96] | always |
| two runs, same seed | identical |

## First implementation, and what the plan did not foresee
The recipe as planned. Training loss after 2000 steps: 2.9e-7, below the correct run's 6.0e-7.
On-path mean |r|: 9.2e-2. E[s] = 0.94 instead of 0.25; the simulated log K drifted to
[3.14, 3.72], outside the box [0.96, 1.96] the loss was evaluated on. WRONG on four of
the five tests. Cause: the Euler equation is necessary, not sufficient; on the box it
also has an over-saving branch (s rising in K, consumption shrinking, capital past the
golden rule), which transversality rules out and the loss does not. Inside the box that
branch has a residual as small as the equilibrium's (mean |r| 3.8e-4). From s ~ 0.5,
descent landed on it.

## Revised recipe (achieved)
One change: `s_init=s*` in `make_net`, so descent starts on the equilibrium branch.
Training loss 6.0e-7 (higher than before);
on-path mean |r| 1.4e-4, max 9.2e-4; max |s - s_ref| 2.3e-3; E[s] 0.2523 (the reference on the
same path 0.2518; reference file 0.2521); log K in [1.08, 1.81]. 11 s. All five tests pass.

## Verification
`python3 -m pytest -q tests/` (14 s), then `/deqn-accuracy results/deqn_stochastic.pt`.
Sanity rule: the constant policy s = s* gives on-path mean |r| of about 1.2e-2 (measured). A run
whose root-mean-square residual is below that but whose E[s] is far from s* has found a wrong
branch, not a solution.

## Critical files
`deqn/stochastic.py` (to write), `deqn/common.py` (read only), `tests/` and
`reference/` (never edit), `results/deqn_stochastic.pt` (output).

# CLAUDE.md -- Growth-model sandbox (lecture demo project)

## Project Overview
- **Model:** stochastic Brock-Mirman growth model: log utility, Y = z K^alpha,
  K' = (1 - delta) K + s Y, log z AR(1) with persistence rho; alpha 0.36, delta 0.10,
  sigma 0.04, beta 0.96, rho 0.9 for the DEQN (hidden in the estimation data). The
  household chooses the savings rate s. Step 1 of the DEQN build uses the course
  notebook's calibration, delta = 1 and beta = 0.99, where the closed form is s* = alpha beta.
- **Purpose:** a sandbox with known answers for working with an AI coding agent. Two
  threads: (1) a deep equilibrium net (DEQN) for the model is built step by step in
  `deqn/`, checked against a closed form and against the reference solver; (2) five
  simulated economies differ only in rho, which is hidden, and rho is estimated from them.
- **Stage:** exploration; nothing here is a paper.

## Repository Structure
```
brock_mirman_reference.py  <- the reference solver (a conventional grid-based solver). READ ONLY.
deqn/
  common.py                <- shared DEQN pieces: parameters, steady state, quadrature nodes,
                              make_net, encode_state, save/load a policy, simulate_path, and the
                              numpy residual used for CHECKING. The torch residual and the training
                              loop are what you write. READ ONLY.
  deterministic.py         <- Step 1 (written in a session): the ported deterministic DEQN
  stochastic.py            <- Step 4: solve(seed, steps); the tests below are its specification
  source/                  <- the original TensorFlow notebooks as scripts; legacy input, never run
tests/                     <- the five acceptance tests of deqn/stochastic.py. THE EXAM: never edit.
reference/                 <- the reference policy on simulated states + baseline.md. THE EXAM: never edit.
data/
  bm_panel.csv             <- observed panel: economy, t, Y, C, K, s (5 x 500 rows)
  bm_panel_truth.csv       <- VERIFICATION FILE. See the first rule.
  raw/                     <- stands in for confidential raw data: never read, never edit
results/                   <- saved policies (*.pt with a *.json sidecar), JSON and CSV outputs
scripts/  figures/  notes/ <- analysis scripts, PDF figures, session notes and reports
```

## The two rules of this project
1. `data/bm_panel_truth.csv` holds the hidden rho of each economy. Do not open it, print it,
   grep it or load it, in any step, unless the prompt explicitly says the verification step
   has been reached. Estimates are produced without it; it is read once, at the end, to grade them.
2. `reference/` holds the reference solver's policy on simulated states and `tests/` the
   acceptance tests. Never load the reference into a loss, an initialisation or a fit; it
   is compared with, once, in the verification step. Never edit `tests/` or `reference/`
   (a hook refuses; the permission rules deny reading `reference/`, pytest may read it).

## DEQN notation and standards
- Policy: a network s_theta(x) for the savings rate, sigmoid output; state transform
  x = (log z / sd_z, log K - log K*) from `common.encode_state`; `common.make_net(2, s_init=s*)`.
- Residual (log utility): r = beta C E[(1 - delta + alpha z' K'^(alpha-1)) / C'] - 1, a fraction
  of consumption; the expectation uses the 5 Gauss-Hermite nodes of `common.gh_nodes`.
- Training states: fresh draws every step, log z ~ N(0, sd_z), log K ~ U(log K* +- 0.5).
- Accuracy means the residual on the path the network's own policy generates
  (`common.simulate_path`) and the distance to the reference policy. The training loss is
  not an accuracy measure and is never reported as one.
- torch on CPU, float64, `torch.manual_seed(seed)` before the network is built; torch only
  under `deqn/`. Every saved policy gets its JSON sidecar via `common.save_policy`.

## Reference solver API (brock_mirman_reference.py)
- `steady_state(beta)`, `solve(beta, rho)`, `simulate(policy, beta, rho, T, seed)`,
  `euler_error(policy, beta, rho, z, K)`, `moments(Y, C, K)`. Read the docstring first.
  Use `moments` instead of re-implementing moments; `solve(beta, rho)(z, K)` is the savings rate.

## Coding Standards
- Python 3.10+, numpy/scipy/pandas/matplotlib, torch as above; type hints on signatures.
- `pathlib.Path` for all file I/O; paths relative to the project root; seeds are arguments
  with a default, never hard-coded inside functions.
- Figures: PDF, serif font, no gridlines. Tables: raw LaTeX with booktabs, no preamble.

## Notation
- K capital, z productivity, s = 1 - C/Y savings rate, rho persistence of log z, beta discount
  factor, K* and s* the deterministic steady state. Moments: Std(dlog C), Autocorr(dlog C),
  Autocorr(log Y), mean s.

## Current Status
**UPDATE THIS EVERY SESSION**
- Current task: (fill in)
- Last thing done: project seeded by setup_demo_project.sh; `pytest -q tests/` fails with NotImplementedError
- Next step: (fill in)

## Do Not Touch
- `brock_mirman_reference.py`, `deqn/common.py`, `deqn/source/`, `tests/`, `reference/`,
  `data/bm_panel.csv`, `data/raw/`

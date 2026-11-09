# CLAUDE.md -- Growth-model sandbox (lecture demo project)

## Project Overview
- **Model:** stochastic Brock-Mirman growth model: log utility, Y = z K^alpha,
  K' = (1 - delta) K + s Y, log z AR(1) with persistence rho; alpha 0.36, delta 0.10,
  sigma 0.04, beta 0.96.
- **Purpose:** a sandbox with a known answer for working with an AI coding agent.
  Five simulated economies differ only in rho, which is hidden from the agent.
- **Stage:** exploration; nothing here is a paper.

## Repository Structure
```
brock_mirman_reference.py  <- the reference solver. READ ONLY. Never modify.
data/
  bm_panel.csv             <- observed panel: economy, t, Y, C, K, s (5 x 500 rows)
  bm_panel_truth.csv       <- VERIFICATION FILE. See the rule below.
  raw/                     <- stands in for confidential raw data: never read, never edit
scripts/                   <- analysis scripts written in the sessions
results/                   <- JSON and CSV outputs
figures/                   <- PDF figures
notes/                     <- session notes, reports, progress files
```

## The one rule of this project
- `data/bm_panel_truth.csv` holds the hidden rho of each economy. Do not open it, print it,
  grep it or load it, in any step, unless the prompt explicitly says the verification step has
  been reached. Estimates are produced without it; it is read once, at the end, to grade them.

## Reference solver API (brock_mirman_reference.py)
- `steady_state(beta)`, `solve(beta, rho)`, `simulate(policy, beta, rho, T, seed)`,
  `euler_error(policy, beta, rho, z, K)`, `moments(Y, C, K)`. Read the docstring first.
  Use `moments` instead of re-implementing moments.

## Coding Standards
- Python 3.10+, numpy/scipy/pandas/matplotlib only unless asked; type hints on signatures.
- `pathlib.Path` for all file I/O; paths relative to the project root; no hard-coded seeds
  inside functions, pass them as arguments with a default.
- Figures: PDF, serif font, no gridlines. Tables: raw LaTeX with booktabs, no preamble.

## Notation
- K capital, z productivity, s = 1 - C/Y savings rate, rho persistence of log z, beta discount
  factor. Moments: Std(dlog C), Autocorr(dlog C), Autocorr(log Y), mean s.

## Current Status
**UPDATE THIS EVERY SESSION**
- Current task: (fill in)
- Last thing done: project seeded by setup_demo_project.sh
- Next step: (fill in)

## Do Not Touch
- `brock_mirman_reference.py`, `data/bm_panel.csv`, `data/raw/`

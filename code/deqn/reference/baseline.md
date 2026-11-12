# Where the test thresholds come from

The five thresholds in `tests/test_deqn_stochastic.py` are set at 4-25x the numbers
the reference solution (`solution/deqn_stochastic.py`, kept outside the folder the
`/goal` works in) reaches, so that a correct solver passes on any laptop and the
wrong one of Part VIII (`stochastic_box.py`) does not.

Reference run, 2026-11-11, PyTorch 2.2.1, float64, 4 CPU threads, seed 42, 2000 steps:

| Quantity (on the path the network's own policy generates, 2,000 periods, seed 7) | Measured | Threshold | Margin |
|---|---|---|---|
| mean \|Euler residual\| | 1.4e-4 | < 1e-3 | 7x |
| max \|s − s_ref\| on the 2,000 states of `policy_rho090.csv` | 2.3e-3 | < 1e-2 | 4x |
| \|E[s] − mean s_ref of the reference file (0.2521)\| | 2e-4 | < 5e-3 | 25x |
| simulated log K inside the training window [log K* − 0.5, log K* + 0.5] = [0.96, 1.96] | [1.08, 1.81] | hard bound | the wrong run gives [3.14, 3.72] |
| two runs with `solve(42, steps=200)` | bit-identical | equality | same machine only |

Training took 11 s; `pytest -q` (one training of 2000 steps plus two of 200) 14 s.
The training loss over the last 100 steps was 6.0e-7. It is not a threshold: the wrong
run of Part VIII reaches 2.9e-7 and fails four of the five tests.

`policy_rho090.csv` is written by `make_reference.py` (the reference policy at beta 0.96,
rho 0.9, on its own simulated path, seed 7). Re-run it: the file must not change.

To re-derive the thresholds on another machine: `python3 solution/deqn_stochastic.py`
and compare with the first column. Only the wall-clock should differ.

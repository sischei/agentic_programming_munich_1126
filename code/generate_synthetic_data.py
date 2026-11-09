"""
Generate the simulated panel used in the live demos of the lecture.

Five economies, each a stochastic Brock-Mirman growth model with its own
persistence rho of log productivity, solved with the reference solver in this
folder (brock_mirman_reference.py) and simulated for T = 500 periods after
a burn-in. The observed panel hides rho; the truth is written to a separate
file that the prompts tell the agent not to read until verification.

    python3 generate_synthetic_data.py

writes

    data/bm_panel.csv        economy, t, Y, C, K, s        (2500 rows)
    data/bm_panel_truth.csv  economy, rho, beta, seed      (5 rows)

Runtime: about 10 s on a laptop CPU (five reference solves).
"""
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brock_mirman_reference as bm  # noqa: E402

BETA = 0.96
RHOS = [0.50, 0.70, 0.80, 0.90, 0.95]
T = 500
SEEDS = [11, 12, 13, 14, 15]

OUT = HERE / "data"
OUT.mkdir(exist_ok=True)

rows, truth = [], []
print(f"{'economy':>7s} {'rho':>5s} {'solve s':>8s}  " + "  ".join(f"{n:>17s}" for n in bm.MOMENT_NAMES))
for i, (rho, seed) in enumerate(zip(RHOS, SEEDS), start=1):
    pol = bm.solve(BETA, rho)
    Y, C, K = bm.simulate(pol, BETA, rho, T=T, seed=seed)
    s = 1.0 - C / Y
    for t in range(T):
        rows.append((i, t + 1, Y[t], C[t], K[t], s[t]))
    truth.append((i, rho, BETA, seed))
    m = bm.moments(Y, C, K)
    print(f"{i:>7d} {rho:>5.2f} {pol.wall:>8.2f}  " + "  ".join(f"{v:>17.5f}" for v in m))

np.savetxt(OUT / "bm_panel.csv", np.array(rows), delimiter=",",
           header="economy,t,Y,C,K,s", comments="", fmt=["%d", "%d", "%.10f", "%.10f", "%.10f", "%.10f"])
np.savetxt(OUT / "bm_panel_truth.csv", np.array(truth), delimiter=",",
           header="economy,rho,beta,seed", comments="", fmt=["%d", "%.2f", "%.2f", "%d"])

print(f"\nwrote {OUT / 'bm_panel.csv'} ({len(rows)} rows) and {OUT / 'bm_panel_truth.csv'}")
print("The truth file is for verification only: the prompts tell the agent not to open it.")

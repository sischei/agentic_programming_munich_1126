"""Writes reference/policy_rho090.csv: the reference solver's savings rate on 2,000 simulated states.

The states are a path of the reference solver's own economy (beta 0.96, rho 0.9, seed 7,
200 periods burn-in). The tests evaluate the network at these states and compare.
Rerun: python3 make_reference.py  (the file must come out unchanged).
"""
import sys
from pathlib import Path

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents if (p / "brock_mirman_reference.py").exists())
sys.path.insert(0, str(ROOT))          # the project root: code/ in the repository, demo/ in the demo project
import brock_mirman_reference as bm  # noqa: E402

BETA, RHO = 0.96, 0.9
policy = bm.solve(BETA, RHO)
Y, C, K = bm.simulate(policy, BETA, RHO, T=2000, seed=7)
z = Y / K ** bm.ALPHA
s = 1.0 - C / Y
out = np.column_stack([np.log(z), np.log(K), s])
path = Path(__file__).resolve().parent / "policy_rho090.csv"
np.savetxt(path, out, delimiter=",", header="log_z,log_K,s_ref", comments="", fmt="%.10f")
print(f"wrote {path.name}: {len(out)} rows; mean s_ref = {s.mean():.4f}; log K in [{np.log(K).min():.2f}, {np.log(K).max():.2f}]")

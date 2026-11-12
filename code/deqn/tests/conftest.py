"""The tests' own oracle: Euler residuals and simulation in plain numpy, independent of
the solver under test and of deqn/common.py. Shared by the five tests."""
import math
import sys
from pathlib import Path

import numpy as np

# project root (holds brock_mirman_reference.py, deqn/, reference/): demo/ in the demo project, code/ in the repository
ROOT = next(p for p in Path(__file__).resolve().parents if (p / "brock_mirman_reference.py").exists())
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# the reference policy: reference/ in the demo project, deqn/reference/ in the repository
REFERENCE_CSV = next(c for c in (ROOT / "reference" / "policy_rho090.csv", ROOT / "deqn" / "reference" / "policy_rho090.csv") if c.exists())

ALPHA, BETA, DELTA, SIGMA, RHO = 0.36, 0.96, 0.10, 0.04, 0.9
K_SS = (ALPHA * BETA / (1.0 - BETA * (1.0 - DELTA))) ** (1.0 / (1.0 - ALPHA))   # 4.294
LOG_K_SS = math.log(K_SS)
_x, _w = np.polynomial.hermite.hermgauss(5)
GH_NODES, GH_WEIGHTS = _x * math.sqrt(2.0), _w / math.sqrt(math.pi)


def euler_residual(s_fn, log_z, log_K):
    """beta C E[(1 - delta + r') / C'] - 1 with a 5-node Gauss-Hermite expectation."""
    z, K = np.exp(log_z), np.exp(log_K)
    s = s_fn(log_z, log_K)
    Y = z * K ** ALPHA
    C = (1.0 - s) * Y
    Kn = (1.0 - DELTA) * K + s * Y
    E = np.zeros_like(C)
    for x, w in zip(GH_NODES, GH_WEIGHTS):
        lzn = RHO * log_z + SIGMA * x
        zn = np.exp(lzn)
        Cn = (1.0 - s_fn(lzn, np.log(Kn))) * zn * Kn ** ALPHA
        E += w * (1.0 - DELTA + ALPHA * zn * Kn ** (ALPHA - 1.0)) / Cn
    return BETA * C * E - 1.0


def simulate_own_path(s_fn, T=2000, seed=7, burn=200):
    """The economy the network generates: (log z, log K) for T periods after a burn-in."""
    rng = np.random.default_rng(seed)
    eps = rng.standard_normal(T + burn)
    lz, lk = 0.0, LOG_K_SS
    LZ, LK = np.empty(T), np.empty(T)
    for t in range(T + burn):
        s = float(s_fn(np.array([lz]), np.array([lk]))[0])
        Kn = (1.0 - DELTA) * math.exp(lk) + s * math.exp(lz) * math.exp(lk) ** ALPHA
        if t >= burn:
            LZ[t - burn], LK[t - burn] = lz, lk
        lz, lk = RHO * lz + SIGMA * eps[t], math.log(Kn)
    return LZ, LK

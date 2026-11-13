"""Shared pieces of the DEQN lead example (shipped to the demo project as deqn/common.py).

A deep equilibrium net (DEQN) replaces the spline of the reference solver by a small
neural network s_theta(x) for the savings rate, and the grid iteration by gradient
descent on the mean squared Euler residual at sampled states. This module holds what
every step shares and what the agent is told NOT to re-implement: the parameters,
the steady state, the quadrature nodes, the network constructor, the state transform,
saving and loading a policy with its sidecar, and simulation with a given policy.

It holds no training loop and no torch residual: those are what the agent writes in
each step. The numpy residual euler_residual_np is here for CHECKING a saved policy
(accuracy.py); the tests carry their own copy in conftest.py.

Both models of the lecture:
    "deterministic": delta = 1, beta = 0.99 (the course notebook); closed form s* = alpha beta
    "stochastic":    the repository's growth model, beta = 0.96, rho = 0.9; known answer = the reference solver

The project root (where brock_mirman_reference.py lives) is put on sys.path below.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
import brock_mirman_reference as bm  # noqa: E402

torch.set_default_dtype(torch.float64)
torch.set_num_threads(4)

PARAMS = {
    "deterministic": dict(alpha=0.36, beta=0.99, delta=1.0, sigma=0.0, rho=0.0),
    "stochastic": dict(alpha=bm.ALPHA, beta=0.96, delta=bm.DELTA, sigma=bm.SIGMA, rho=0.9),
}
WINDOW_HALF_WIDTH = 0.5   # training states: log K in [log K* - 0.5, log K* + 0.5]


def steady_state(p: dict) -> tuple[float, float]:
    """Deterministic steady state (K*, s*) at z = 1. With delta = 1, s* = alpha beta."""
    a, b, d = p["alpha"], p["beta"], p["delta"]
    k = (a * b / (1.0 - b * (1.0 - d))) ** (1.0 / (1.0 - a))
    return k, d * k ** (1.0 - a)


def sd_logz(p: dict) -> float:
    """Unconditional standard deviation of log z."""
    return p["sigma"] / math.sqrt(1.0 - p["rho"] ** 2) if p["sigma"] > 0 else 1.0


def gh_nodes(n: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Hermite nodes and weights for E[f(eps)], eps ~ N(0, 1); weights sum to 1."""
    if n == 5:
        return bm.GH_NODES, bm.GH_WEIGHTS
    x, w = np.polynomial.hermite.hermgauss(n)
    return x * math.sqrt(2.0), w / math.sqrt(math.pi)


def make_net(n_in: int, width: int = 32, act: str = "tanh", s_init: float | None = None) -> torch.nn.Sequential:
    """n_in-width-width-1 network with a sigmoid output (the savings rate is in (0, 1)).

    s_init: if given, the output bias is set so that the untrained network returns
    s_init on average. Step 1 (delta = 1) does not need it. Step 4 (delta < 1) does:
    the Euler equation also has an over-saving branch, and the initialization decides
    which branch gradient descent lands on (see silent_failure.md).
    """
    acts = {"tanh": torch.nn.Tanh, "silu": torch.nn.SiLU, "relu": torch.nn.ReLU}
    net = torch.nn.Sequential(
        torch.nn.Linear(n_in, width), acts[act](),
        torch.nn.Linear(width, width), acts[act](),
        torch.nn.Linear(width, 1), torch.nn.Sigmoid(),
    )
    if s_init is not None:
        with torch.no_grad():
            net[-2].bias.fill_(math.log(s_init / (1.0 - s_init)))
    return net


def encode_state(logz: torch.Tensor, logK: torch.Tensor, p: dict) -> torch.Tensor:
    """Network input for the stochastic model: (log z / sd, log K - log K*), shape (n, 2)."""
    kss, _ = steady_state(p)
    return torch.stack([logz / sd_logz(p), logK - math.log(kss)], dim=1)


def save_policy(net: torch.nn.Sequential, path: str | Path, meta: dict) -> Path:
    """Write the weights to <path> and a JSON sidecar next to it (model, params, arch, seed, ...)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), path)
    n_in = net[0].in_features
    width = net[0].out_features
    act = type(net[1]).__name__.lower()
    side = dict(meta, arch=dict(n_in=n_in, width=width, act=act), torch=torch.__version__)
    path.with_suffix(".json").write_text(json.dumps(side, indent=2) + "\n")
    return path


def load_policy(path: str | Path) -> tuple[torch.nn.Sequential, dict]:
    """Rebuild the network from the sidecar and load the weights."""
    path = Path(path)
    meta = json.loads(path.with_suffix(".json").read_text())
    a = meta["arch"]
    net = make_net(a["n_in"], a["width"], a["act"])
    net.load_state_dict(torch.load(path))
    net.eval()
    return net, meta


def params_of(meta: dict) -> dict:
    """The parameters a policy was trained with: the sidecar's, else the defaults of its model.
    A network trained at another calibration must be encoded and simulated at that calibration."""
    return dict(meta.get("params") or PARAMS[meta["model"]])


def policy_fn(net: torch.nn.Sequential, meta: dict):
    """Numpy-in, numpy-out savings-rate function: s(log_z, log_K) (stochastic) or s(K) (deterministic)."""
    p = params_of(meta)
    if meta["model"] == "stochastic":
        def s(log_z, log_K):
            with torch.no_grad():
                x = encode_state(torch.as_tensor(np.asarray(log_z, dtype=float)),
                                 torch.as_tensor(np.asarray(log_K, dtype=float)), p)
                return net(x).squeeze(-1).numpy()
    else:
        def s(K):
            with torch.no_grad():
                x = torch.as_tensor(np.asarray(K, dtype=float)).reshape(-1, 1)
                return net(x).squeeze(-1).numpy()
    return s


def simulate_path(s_fn, p: dict, T: int = 2000, seed: int = 7, burn: int = 200):
    """Simulate (log z, log K) for T periods with the policy s_fn(log_z, log_K), after a burn-in.

    Starts at the steady state. Returns two arrays of length T. This is the economy the
    network itself generates; accuracy is measured here, not on the training box.
    """
    rng = np.random.default_rng(seed)
    kss, _ = steady_state(p)
    eps = rng.standard_normal(T + burn)
    lz, lk = 0.0, math.log(kss)
    LZ, LK = np.empty(T), np.empty(T)
    for t in range(T + burn):
        s = float(s_fn(np.array([lz]), np.array([lk]))[0])
        z, K = math.exp(lz), math.exp(lk)
        Kn = (1.0 - p["delta"]) * K + s * z * K ** p["alpha"]
        if t >= burn:
            LZ[t - burn], LK[t - burn] = lz, lk
        lz = p["rho"] * lz + p["sigma"] * eps[t]
        lk = math.log(Kn)
    return LZ, LK


def euler_residual_np(s_fn, log_z: np.ndarray, log_K: np.ndarray, p: dict, n_gh: int = 5) -> np.ndarray:
    """Relative Euler residual beta C E[(1 - delta + r') / C'] - 1 at given states, in numpy."""
    a, b, d = p["alpha"], p["beta"], p["delta"]
    nodes, wts = gh_nodes(n_gh)
    z, K = np.exp(log_z), np.exp(log_K)
    s = s_fn(log_z, log_K)
    Y = z * K ** a
    C = (1.0 - s) * Y
    Kn = (1.0 - d) * K + s * Y
    E = np.zeros_like(C)
    for x, w in zip(nodes, wts):
        lzn = p["rho"] * log_z + p["sigma"] * x
        zn = np.exp(lzn)
        sn = s_fn(lzn, np.log(Kn))
        Cn = (1.0 - sn) * zn * Kn ** a
        E += w * (1.0 - d + a * zn * Kn ** (a - 1.0)) / Cn
    return b * C * E - 1.0

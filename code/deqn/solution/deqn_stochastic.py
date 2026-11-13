"""Reference solution of Step 4: a DEQN for the stochastic Brock-Mirman model.

Kept outside the folder the /goal works in. The residual function below is the code
shown on the slides of Part VII; stochastic_box.py calls train() with s_init=None and
is the silent failure of Part VIII.

Recipe: state x = (log z / sd_z, log K - log K*); network 2-32-32-1 tanh with a sigmoid
output for the savings rate, output bias initialized at logit(s*); loss = mean squared
Euler residual at 128 fresh training states per step, log z ~ N(0, sd_z),
log K ~ U(log K* +- 0.5); 5 Gauss-Hermite nodes; Adam 1e-3; 2000 steps; seed 42; float64.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from deqn import common  # noqa: E402

P = common.PARAMS["stochastic"]
ALPHA, BETA, DELTA, SIGMA, RHO = (P[k] for k in ("alpha", "beta", "delta", "sigma", "rho"))
K_SS, S_SS = common.steady_state(P)
SD_LOGZ = common.sd_logz(P)
NODES, WEIGHTS = (torch.as_tensor(v) for v in common.gh_nodes(5))


def residual(net: torch.nn.Module, logz: torch.Tensor, logK: torch.Tensor) -> torch.Tensor:
    """Relative Euler residual  beta C E[(1 - delta + r') / C'] - 1  at states (log z, log K)."""
    z, K = torch.exp(logz), torch.exp(logK)
    Y = z * K ** ALPHA
    s = net(common.encode_state(logz, logK, P)).squeeze(1)      # savings rate today
    C = (1.0 - s) * Y
    Kn = (1.0 - DELTA) * K + s * Y                              # capital tomorrow
    logKn = torch.log(Kn)
    E = torch.zeros_like(C)
    for x, w in zip(NODES, WEIGHTS):                            # expectation over eps'
        logzn = RHO * logz + SIGMA * x
        zn = torch.exp(logzn)
        sn = net(common.encode_state(logzn, logKn, P)).squeeze(1)
        Cn = (1.0 - sn) * zn * Kn ** ALPHA
        E = E + w * (1.0 - DELTA + ALPHA * zn * Kn ** (ALPHA - 1.0)) / Cn
    return BETA * C * E - 1.0


def train(seed: int = 42, steps: int = 2000, batch: int = 128, lr: float = 1e-3,
          s_init: float | None = S_SS, window: float = common.WINDOW_HALF_WIDTH):
    """Train the network; returns (net, wall seconds, mean training loss over the last 100 steps)."""
    torch.manual_seed(seed)
    net = common.make_net(2, width=32, act="tanh", s_init=s_init)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        logz = torch.randn(batch) * SD_LOGZ
        logK = math.log(K_SS) + (torch.rand(batch) * 2.0 - 1.0) * window
        loss = (residual(net, logz, logK) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return net, time.perf_counter() - t0, float(np.mean(losses[-100:]))


def solve(seed: int = 42, steps: int = 2000, s_init: float | None = S_SS,
          out: str | Path = "results/deqn_stochastic.pt") -> dict:
    net, wall, train_loss = train(seed=seed, steps=steps, s_init=s_init)
    meta = dict(model="stochastic", params=P, seed=seed, steps=steps, wall=round(wall, 2),
                train_loss_last100=train_loss, s_init=s_init, window=common.WINDOW_HALF_WIDTH)
    common.save_policy(net, out, meta)
    return dict(s=common.policy_fn(net, meta), net=net, wall=wall, train_loss=train_loss)


def report(out: dict) -> dict:
    """The on-path numbers of the lecture: residuals, distance to the reference policy, E[s], window."""
    import brock_mirman_reference as bm
    LZ, LK = common.simulate_path(out["s"], P)
    r = np.abs(common.euler_residual_np(out["s"], LZ, LK, P))
    s_net = out["s"](LZ, LK)
    ref_policy = bm.solve(BETA, RHO)
    window = (math.log(K_SS) - 0.5, math.log(K_SS) + 0.5)
    inside = window[0] < LK.min() and LK.max() < window[1]
    if not inside:                     # the reference spline is clipped beyond its grid: compare on the reference solver's own path
        _, _, K_ref = bm.simulate(ref_policy, BETA, RHO, len(LK), seed=7)
        LZ, LK_cmp = LZ, np.log(K_ref)
        s_net = out["s"](LZ, LK_cmp)
    else:
        LK_cmp = LK
    s_ref = ref_policy(np.exp(LZ), np.exp(LK_cmp))
    return dict(wall=out["wall"], train_loss=out["train_loss"], mean_abs_r=r.mean(), max_abs_r=r.max(),
                max_abs_ds=np.abs(s_net - s_ref).max(), mean_s=out["s"](*common.simulate_path(out["s"], P)).mean(),
                mean_s_ref=s_ref.mean(), gap_where="the own path" if inside else "the reference solver's path",
                logK_min=LK.min(), logK_max=LK.max(), window=window)


if __name__ == "__main__":
    out = solve(out=Path(__file__).resolve().parents[1] / "outputs" / "deqn_stochastic.pt")
    R = report(out)
    print(f"trained in {R['wall']:.1f} s; training loss (last 100 steps) {R['train_loss']:.2e}")
    print(f"on-path |Euler residual|: mean {R['mean_abs_r']:.2e}, max {R['max_abs_r']:.2e}")
    print(f"max |s - s_ref| on {R['gap_where']} {R['max_abs_ds']:.2e}; E[s] {R['mean_s']:.4f} vs reference {R['mean_s_ref']:.4f}")
    print(f"simulated log K in [{R['logK_min']:.2f}, {R['logK_max']:.2f}], training window [{R['window'][0]:.2f}, {R['window'][1]:.2f}]")

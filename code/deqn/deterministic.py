"""Step 1 of the lecture: the deterministic Brock-Mirman DEQN, ported from TensorFlow to PyTorch.

The port of source/deqn_tf_deterministic.py, same recipe: delta = 1, beta = 0.99, the
network maps K to the savings rate, K ~ U[0.1, 1], 64 states per step, Adam 1e-3,
2000 steps. Two changes the Level-3 prompt asks for: float64 with a fixed seed, and a
tanh network of width 32 built with common.make_net (default initialization: the closed form
is used in the check only, nowhere else).

What the notebook never printed: the error against the closed form, s* = alpha beta.
That number is the point of the step.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deqn import common  # noqa: E402

P = common.PARAMS["deterministic"]
ALPHA, BETA, DELTA = P["alpha"], P["beta"], P["delta"]
K_SS, S_SS = common.steady_state(P)          # K* = 0.1995, s* = alpha beta = 0.3564
K_LB, K_UB = 0.10, 1.0


def residual(net: torch.nn.Module, K: torch.Tensor) -> torch.Tensor:
    """Relative Euler residual 1 - C' / (beta C (1 - delta + r')) at capital K, as in the notebook."""
    Y = K ** ALPHA
    s = net(K.unsqueeze(1)).squeeze(1)
    C = (1.0 - s) * Y
    Kn = (1.0 - DELTA) * K + s * Y
    Yn = Kn ** ALPHA
    sn = net(Kn.unsqueeze(1)).squeeze(1)
    Cn = (1.0 - sn) * Yn
    Rn = ALPHA * Kn ** (ALPHA - 1.0)
    return 1.0 - Cn / (BETA * C * (Rn + 1.0 - DELTA))


def solve(seed: int = 42, steps: int = 2000, batch: int = 64, lr: float = 1e-3, s_init: float | None = None):
    torch.manual_seed(seed)
    net = common.make_net(1, width=32, act="tanh", s_init=s_init)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    t0 = time.perf_counter()
    losses = []
    for _ in range(steps):
        K = K_LB + (K_UB - K_LB) * torch.rand(batch)
        loss = (residual(net, K) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())
    wall = time.perf_counter() - t0
    meta = dict(model="deterministic", params=P, seed=seed, steps=steps, wall=round(wall, 2),
                train_loss_last100=float(np.mean(losses[-100:])), s_init=s_init, K_range=[K_LB, K_UB])
    return net, meta


def check(net: torch.nn.Module) -> dict:
    """The closed-form check on 1,001 points of [0.1, 1]: max |s(K) - alpha beta|, and the residual."""
    K = torch.linspace(K_LB, K_UB, 1001)
    with torch.no_grad():
        s = net(K.unsqueeze(1)).squeeze(1).numpy()
        r = residual(net, K).numpy()
    return dict(max_abs_ds=float(np.abs(s - S_SS).max()), mean_abs_r=float(np.abs(r).mean()),
                max_abs_r=float(np.abs(r).max()), K=K.numpy(), s=s, r=r)


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent / "outputs"
    net, meta = solve()
    c = check(net)
    meta["max_abs_ds"] = c["max_abs_ds"]
    common.save_policy(net, out_dir / "deqn_deterministic.pt", meta)
    print(f"trained in {meta['wall']:.1f} s; training loss (last 100 steps) {meta['train_loss_last100']:.2e}")
    print(f"max |s(K) - alpha*beta| on [0.1, 1]: {c['max_abs_ds']:.2e}  (alpha*beta = {S_SS:.4f})")
    print(f"|Euler residual| on the grid: mean {c['mean_abs_r']:.2e}, max {c['max_abs_r']:.2e}")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(1, 2, figsize=(6.0, 2.4))
    ax[0].plot(c["K"], c["s"], label="network $s(K)$")
    ax[0].axhline(S_SS, color="k", ls="--", lw=0.8, label=r"closed form $\alpha\beta$")
    ax[0].set_ylim(0.30, 0.40); ax[0].set_xlabel("$K$"); ax[0].set_ylabel("savings rate"); ax[0].legend(frameon=False)
    ax[1].semilogy(c["K"], np.abs(c["r"]) + 1e-16)
    ax[1].set_xlabel("$K$"); ax[1].set_ylabel("|Euler residual|")
    fig.tight_layout()
    fig.savefig(out_dir / "deqn_step1_figure.pdf")
    print(f"figure written to {out_dir / 'deqn_step1_figure.pdf'}")

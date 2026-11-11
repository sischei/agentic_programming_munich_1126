"""Step 5 of the lecture, one prompt -> table + figure: four loss kernels on the same recipe.

Trains the stochastic DEQN four times (MSE, MAE, Huber, log-cosh on the Euler residual),
everything else as in solution/deqn_stochastic.py, and reports each on the path the
trained network generates: training loss, |Euler residual| mean / p99 / max, E[s],
max |s - s_egm|, seconds. Writes outputs/deqn_kernel_table.tex (booktabs, no preamble)
and outputs/deqn_kernel_figure.pdf. Asserts that every E[s] is within 5e-3 of the mean
reference savings rate of reference/egm_policy_rho090.csv (the constant the tests use): the
kernels differ in accuracy, not in the answer.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import brock_mirman_reference as bm  # noqa: E402
from deqn import common  # noqa: E402
from deqn.solution import deqn_stochastic as sol  # noqa: E402

KERNELS = {
    "MSE": lambda r: (r ** 2).mean(),
    "MAE": lambda r: r.abs().mean(),
    "Huber": lambda r: torch.nn.functional.huber_loss(r, torch.zeros_like(r), delta=1e-2),
    "log-cosh": lambda r: torch.log(torch.cosh(r)).mean(),
}
OUT = Path(__file__).resolve().parent / "outputs"


def train_with(kernel, seed=42, steps=2000, batch=128, lr=1e-3):
    torch.manual_seed(seed)
    net = common.make_net(2, width=32, act="tanh", s_init=sol.S_SS)
    opt = torch.optim.Adam(net.parameters(), lr=lr)
    losses = []
    t0 = time.perf_counter()
    for _ in range(steps):
        logz = torch.randn(batch) * sol.SD_LOGZ
        logK = np.log(sol.K_SS) + (torch.rand(batch) * 2.0 - 1.0) * common.WINDOW_HALF_WIDTH
        loss = kernel(sol.residual(net, logz, logK))
        opt.zero_grad(); loss.backward(); opt.step()
        losses.append(loss.item())
    return net, time.perf_counter() - t0, float(np.mean(losses[-100:]))


if __name__ == "__main__":
    P = sol.P
    egm = bm.solve(P["beta"], P["rho"])
    rows, curves = [], {}
    for name, kernel in KERNELS.items():
        net, wall, tl = train_with(kernel)
        meta = dict(model="stochastic", params=P, seed=42, steps=2000, kernel=name)
        s_fn = common.policy_fn(net, meta)
        LZ, LK = common.simulate_path(s_fn, P)
        r = np.abs(common.euler_residual_np(s_fn, LZ, LK, P))
        s_net, s_egm = s_fn(LZ, LK), egm(np.exp(LZ), np.exp(LK))
        rows.append((name, tl, r.mean(), np.percentile(r, 99), r.max(), s_net.mean(), np.abs(s_net - s_egm).max(), wall))
        Kg = np.linspace(np.log(sol.K_SS) - 0.5, np.log(sol.K_SS) + 0.5, 200)
        curves[name] = (np.exp(Kg), s_fn(np.zeros_like(Kg), Kg), r)
        print(f"{name:9s} loss {tl:.2e}  mean|r| {r.mean():.2e}  p99 {np.percentile(r, 99):.2e}  max {r.max():.2e}  "
              f"E[s] {s_net.mean():.4f}  max|s-s_egm| {np.abs(s_net - s_egm).max():.2e}  {wall:.0f} s")
    ref = np.loadtxt(Path(__file__).resolve().parent / "reference" / "egm_policy_rho090.csv", delimiter=",", skiprows=1)
    mean_s_egm = ref[:, 2].mean()
    assert all(abs(row[5] - mean_s_egm) < 5e-3 for row in rows), "a kernel changed the answer, not just the accuracy"
    OUT.mkdir(exist_ok=True)
    with open(OUT / "deqn_kernel_table.tex", "w") as f:
        f.write("\\begin{tabular}{lrrrrrrr}\n\\toprule\n")
        f.write("Kernel & Train.\\ loss & mean $|r|$ & p99 $|r|$ & max $|r|$ & $E[s]$ & max $|s-s_{\\mathrm{ref}}|$ & s \\\\\n\\midrule\n")
        for name, tl, m, p99, mx, es, gap, wall in rows:
            f.write(f"{name} & {tl:.1e} & {m:.1e} & {p99:.1e} & {mx:.1e} & {es:.4f} & {gap:.1e} & {wall:.0f} \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    fig, ax = plt.subplots(1, 2, figsize=(6.4, 2.5))
    Kg = np.linspace(np.log(sol.K_SS) - 0.5, np.log(sol.K_SS) + 0.5, 200)
    ax[0].plot(np.exp(Kg), egm(np.ones_like(Kg), np.exp(Kg)), "k--", lw=0.9, label="reference")
    for name, (K, s, _) in curves.items():
        ax[0].plot(K, s, lw=0.9, label=name)
    ax[0].set_xlabel("$K$ (at $z = 1$)"); ax[0].set_ylabel("savings rate"); ax[0].legend(frameon=False, fontsize=7)
    bins = np.logspace(-6, -1, 40)
    for name, (_, _, r) in curves.items():
        ax[1].hist(r, bins=bins, histtype="step", label=name)
    ax[1].set_xscale("log"); ax[1].set_xlabel("|Euler residual| on the own path"); ax[1].set_ylabel("periods")
    ax[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "deqn_kernel_figure.pdf")
    print(f"wrote {OUT / 'deqn_kernel_table.tex'} and {OUT / 'deqn_kernel_figure.pdf'}")

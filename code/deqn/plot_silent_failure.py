"""Redraws the two figures of the silent-failure slide from the saved policies.

    python3 plot_silent_failure.py        -> slides/figures/agentic_deqn_silent_{bad,good}.pdf

Left panel: the network's savings rate at z = 1 over the training window against the
reference policy. Right panel: the simulated log K of the network's own economy, with the
training window shaded. The bad run is outputs/deqn_stochastic_box.pt, the good one
outputs/deqn_stochastic.pt.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import brock_mirman_reference as bm  # noqa: E402
from deqn import common  # noqa: E402

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"font.family": "serif", "font.size": 9})
HERE = Path(__file__).resolve().parent
FIG = HERE.parents[1] / "slides" / "figures"
P = common.PARAMS["stochastic"]
kss, _ = common.steady_state(P)
egm = bm.solve(P["beta"], P["rho"])

for tag, fname in (("bad", "deqn_stochastic_box.pt"), ("good", "deqn_stochastic.pt")):
    net, meta = common.load_policy(HERE / "outputs" / fname)
    s_fn = common.policy_fn(net, meta)
    LZ, LK = common.simulate_path(s_fn, P, T=600)
    lo, hi = np.log(kss) - 0.5, np.log(kss) + 0.5
    Kg = np.linspace(lo, hi, 200)
    fig, ax = plt.subplots(1, 2, figsize=(5.2, 2.2))
    ax[0].plot(np.exp(Kg), egm(np.ones_like(Kg), np.exp(Kg)), "k--", lw=0.9, label="reference")
    ax[0].plot(np.exp(Kg), s_fn(np.zeros_like(Kg), Kg), lw=1.2, label="network")
    ax[0].set_ylim(0, 1); ax[0].set_xlabel("$K$ (at $z=1$)"); ax[0].set_ylabel("savings rate"); ax[0].legend(frameon=False)
    ax[1].axhspan(lo, hi, color="0.85", label="training window")
    ax[1].plot(LK, lw=0.8, label="simulated $\\log K$")
    ax[1].set_ylim(0.5, 4.0); ax[1].set_xlabel("period"); ax[1].set_ylabel("$\\log K$"); ax[1].legend(frameon=False, loc="lower right", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / f"agentic_deqn_silent_{tag}.pdf")
    print(f"wrote agentic_deqn_silent_{tag}.pdf: E[s] {s_fn(LZ, LK).mean():.3f}, log K in [{LK.min():.2f}, {LK.max():.2f}]")

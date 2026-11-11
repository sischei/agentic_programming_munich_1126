"""Plot 2D PINN prediction vs analytical sin(pi x) sin(pi y) on (0,1)^2.

Pass --hard-bc if the matching solver_pde.py run used the hard-BC ansatz, so the
saved inner-MLP weights are wrapped in HardBCWrapper before evaluation.
"""
import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from solver_pde import MLP, HardBCWrapper

plt.rcParams["font.size"] = 13
HERE = Path(__file__).parent
PI = math.pi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hard-bc", action="store_true",
                   help="wrap loaded MLP in HardBCWrapper (must match the training run)")
    args = p.parse_args()

    base = MLP()
    base.load_state_dict(torch.load(HERE / "model_pde.pt", weights_only=True))
    model = HardBCWrapper(base) if args.hard_bc else base
    model.eval()

    M = 128
    g = torch.linspace(0.0, 1.0, M)
    Yg, Xg = torch.meshgrid(g, g, indexing="ij")
    XY = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
    with torch.no_grad():
        u_pred = model(XY).reshape(M, M).numpy()
    u_exact = (torch.sin(PI * Xg) * torch.sin(PI * Yg)).numpy()
    err = u_pred - u_exact

    linf = float(np.abs(err).max())
    l2 = float(np.sqrt((err ** 2).mean()))

    vmax = max(np.abs(u_exact).max(), np.abs(u_pred).max())
    e_max = np.abs(err).max()
    extent = [0.0, 1.0, 0.0, 1.0]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    im0 = axes[0].imshow(
        u_exact, origin="lower", extent=extent, vmin=-vmax, vmax=vmax, cmap="RdBu_r"
    )
    axes[0].set_title(r"$u_{\mathrm{exact}} = \sin(\pi x)\sin(\pi y)$")
    plt.colorbar(im0, ax=axes[0])

    im1 = axes[1].imshow(
        u_pred, origin="lower", extent=extent, vmin=-vmax, vmax=vmax, cmap="RdBu_r"
    )
    axes[1].set_title("PINN prediction" + ("  (hard BC)" if args.hard_bc else "  (soft BC)"))
    plt.colorbar(im1, ax=axes[1])

    im2 = axes[2].imshow(
        err, origin="lower", extent=extent, vmin=-e_max, vmax=e_max, cmap="RdBu_r"
    )
    axes[2].set_title(f"error  Linf={linf:.2e}  L2={l2:.2e}")
    plt.colorbar(im2, ax=axes[2])

    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")

    fig.tight_layout()
    out = HERE / ("solution_pde_hardbc.png" if args.hard_bc else "solution_pde.png")
    fig.savefig(out, dpi=120)
    print(f"saved {out}", flush=True)


if __name__ == "__main__":
    main()

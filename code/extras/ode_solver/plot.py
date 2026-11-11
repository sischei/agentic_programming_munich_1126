"""Plot PINN prediction vs analytical cos(x) on [0, 2*pi]."""
import math
from pathlib import Path

import matplotlib.pyplot as plt
import torch

from solver import MLP

plt.rcParams["font.size"] = 13
HERE = Path(__file__).parent


def main():
    model = MLP()
    model.load_state_dict(torch.load(HERE / "model.pt", weights_only=True))
    model.eval()

    x = torch.linspace(0.0, 2 * math.pi, 1000).unsqueeze(1)
    with torch.no_grad():
        y_pred = model(x).squeeze().numpy()
    y_true = torch.cos(x).squeeze().numpy()
    x_np = x.squeeze().numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    ax1.plot(x_np, y_true, "k-", lw=2, label=r"$\cos(x)$")
    ax1.plot(x_np, y_pred, "r--", lw=1.5, label="PINN")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y(x)")
    ax1.set_title(r"$y'' + y = 0,\ y(0)=1,\ y'(0)=0$")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(x_np, y_pred - y_true, "b-", lw=1.2)
    ax2.set_xlabel("x")
    ax2.set_ylabel("PINN - cos(x)")
    ax2.set_title("pointwise error")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(HERE / "solution.png", dpi=120)
    print(f"saved {HERE / 'solution.png'}", flush=True)


if __name__ == "__main__":
    main()

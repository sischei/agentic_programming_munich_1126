"""Baseline the loop must beat: the same network trained with a SOFT boundary penalty.

Same recipe as the loop prompt (1-32-32-1 tanh, 256 fresh uniform collocation points
per step, Adam 1e-3, 3000 steps, seed 42, float64), but the boundary conditions enter
the loss as a penalty with weight 1 instead of being built into the trial solution:

    loss = mean(residual^2) + (u(0) - 0)^2 + (u(pi/2) - 1)^2

Prints max |u - sin x| on 1001 points. This number is STORED_SOFT_BC_ERROR in
tests/test_pinn_hard_bc.py: the loop's hard-BC solver has to do better than this.

    python3 soft_bc_baseline.py          # about 15 s on a laptop CPU
"""
import math

import numpy as np
import torch

SEED, STEPS, N_COLL, LR = 42, 3000, 256, 1e-3
torch.set_default_dtype(torch.float64)


def main(seed: int = SEED) -> float:
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    )
    opt = torch.optim.Adam(net.parameters(), lr=LR)
    x_bd = torch.tensor([[0.0], [math.pi / 2]])
    u_bd = torch.tensor([[0.0], [1.0]])
    for _ in range(STEPS):
        opt.zero_grad()
        x = torch.rand(N_COLL, 1) * (math.pi / 2)
        x.requires_grad_(True)
        u = net(x)
        (u_x,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
        (u_xx,) = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)
        loss = ((u_xx + u) ** 2).mean() + ((net(x_bd) - u_bd) ** 2).sum()
        loss.backward()
        opt.step()
    xs = torch.linspace(0.0, math.pi / 2, 1001).unsqueeze(1)
    with torch.no_grad():
        err = (net(xs) - torch.sin(xs)).abs().max().item()
    return err


if __name__ == "__main__":
    err = main()
    print(f"soft-BC baseline, seed {SEED}: max |u - sin x| = {err:.3e}")

"""Reference solution of the loop demo (kept OUTSIDE code/pinn_loop/ so the agent cannot read it).

u'' + u = 0 on [0, pi/2], u(0) = 0, u(pi/2) = 1, exact solution sin x.
Trial solution u(x) = 2x/pi + x (pi/2 - x) N(x): the boundary conditions hold by
construction, so the loss is the residual alone. Recipe as in the loop prompt.

    cp ../pinn_loop/tests -r . && python3 -m pytest -q      # 3 passed, about 30 s
"""
import math

import numpy as np
import torch

torch.set_default_dtype(torch.float64)


def solve(seed: int = 42) -> dict:
    torch.manual_seed(seed)
    net = torch.nn.Sequential(
        torch.nn.Linear(1, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 32), torch.nn.Tanh(),
        torch.nn.Linear(32, 1),
    )

    def trial(x: torch.Tensor) -> torch.Tensor:
        return 2.0 * x / math.pi + x * (math.pi / 2 - x) * net(x)

    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for _ in range(3000):
        opt.zero_grad()
        x = torch.rand(256, 1) * (math.pi / 2)
        x.requires_grad_(True)
        u = trial(x)
        (u_x,) = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)
        (u_xx,) = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)
        loss = ((u_xx + u) ** 2).mean()
        loss.backward()
        opt.step()

    def u_np(x: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            xt = torch.as_tensor(np.asarray(x, dtype=float)).reshape(-1, 1)
            return trial(xt).reshape(-1).numpy()

    xs = np.linspace(0.0, math.pi / 2, 1001)
    max_err = float(np.max(np.abs(u_np(xs) - np.sin(xs))))
    return {"max_err": max_err, "u": u_np}


if __name__ == "__main__":
    out = solve()
    print(f"hard-BC reference, seed 42: max |u - sin x| = {out['max_err']:.3e}")

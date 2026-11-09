"""PINN solver for 2D Poisson -Delta u = f on (0,1)^2 with u = 0 on the boundary.

Manufactured solution:
    u_exact(x, y) = sin(pi x) sin(pi y)
    f(x, y)       = 2 pi^2 sin(pi x) sin(pi y)

Two BC strategies are available via --hard-bc:
    soft (default): boundary penalty (u_bd)^2 with weight lam_bc
    hard          : ansatz u_hat(x,y) = x(1-x)y(1-y) * NN(x,y), exact zero on dOmega
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).parent
PI = math.pi

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def init_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class MLP(nn.Module):
    def __init__(self, in_dim=2, hidden=64, n_layers=4):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class HardBCWrapper(nn.Module):
    """u_hat(x,y) = x(1-x)y(1-y) * NN(x,y). Dirichlet zero on the unit-square boundary by construction."""

    def __init__(self, base):
        super().__init__()
        self.base = base

    def forward(self, X):
        x = X[:, 0:1]
        y = X[:, 1:2]
        return x * (1 - x) * y * (1 - y) * self.base(X)


def laplacian(model, X):
    u = model(X)
    grad_u = torch.autograd.grad(
        u, X, torch.ones_like(u), create_graph=True, retain_graph=True
    )[0]
    u_x = grad_u[:, 0:1]
    u_y = grad_u[:, 1:2]
    u_xx = torch.autograd.grad(
        u_x, X, torch.ones_like(u_x), create_graph=True, retain_graph=True
    )[0][:, 0:1]
    u_yy = torch.autograd.grad(
        u_y, X, torch.ones_like(u_y), create_graph=True, retain_graph=True
    )[0][:, 1:2]
    return u, u_xx + u_yy


def forcing(X):
    x = X[:, 0:1]
    y = X[:, 1:2]
    return 2 * PI ** 2 * torch.sin(PI * x) * torch.sin(PI * y)


def boundary_points(n_per_side, device, dtype):
    t = torch.rand(n_per_side, 1, device=device, dtype=dtype)
    z = torch.zeros(n_per_side, 1, device=device, dtype=dtype)
    o = torch.ones(n_per_side, 1, device=device, dtype=dtype)
    return torch.cat(
        [
            torch.cat([t, z], dim=1),
            torch.cat([t, o], dim=1),
            torch.cat([z, t], dim=1),
            torch.cat([o, t], dim=1),
        ],
        dim=0,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20000)
    p.add_argument("--n-int", type=int, default=2048)
    p.add_argument("--n-bd", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--lam-bc", type=float, default=1.0,
                   help="soft-BC penalty weight; ignored when --hard-bc is set")
    p.add_argument("--hard-bc", action="store_true",
                   help="use ansatz u = x(1-x)y(1-y)*NN(x,y) instead of soft penalty")
    p.add_argument("--lr-schedule", choices=["none", "cosine"], default="none")
    p.add_argument("--clip-grad", type=float, default=0.0,
                   help="max L2 grad norm; 0 disables clipping")
    p.add_argument("--init", choices=["default", "xavier"], default="xavier")
    p.add_argument("--lbfgs-steps", type=int, default=0,
                   help="optional L-BFGS finetune steps after Adam; 0 disables")
    p.add_argument("--device", default="cpu")
    p.add_argument("--dtype", choices=list(DTYPES), default="float32")
    p.add_argument("--eval-freq", type=int, default=500)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    dtype = DTYPES[args.dtype]
    n_per_side = args.n_bd // 4

    base = MLP().to(device=device, dtype=dtype)
    if args.init == "xavier":
        base.apply(init_xavier)
    model = HardBCWrapper(base) if args.hard_bc else base

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        if args.lr_schedule == "cosine" else None
    )

    # Discrete L2 on a uniform 128x128 grid: since |Omega|=1, mean(err^2) already
    # absorbs the dx*dy cell measure, so sqrt(mean(err^2)) ~ ||err||_{L2(Omega)}.
    # On a non-unit domain you would multiply by sqrt(area).
    M = 128
    g = torch.linspace(0.0, 1.0, M, device=device, dtype=dtype)
    Yg, Xg = torch.meshgrid(g, g, indexing="ij")
    X_eval = torch.stack([Xg.reshape(-1), Yg.reshape(-1)], dim=1)
    u_eval_true = (torch.sin(PI * Xg) * torch.sin(PI * Yg)).reshape(-1, 1)

    def step():
        opt.zero_grad()
        X_int = torch.rand(args.n_int, 2, device=device, dtype=dtype, requires_grad=True)
        _, lap = laplacian(model, X_int)
        residual = -lap - forcing(X_int)
        loss_pde = (residual ** 2).mean()

        if args.hard_bc:
            loss_bc = torch.zeros((), device=device, dtype=dtype)
            loss = loss_pde
        else:
            X_bd = boundary_points(n_per_side, device, dtype)
            u_bd = model(X_bd)
            loss_bc = (u_bd ** 2).mean()
            loss = loss_pde + args.lam_bc * loss_bc

        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()
        return loss_pde.detach(), loss_bc.detach()

    step_c = torch.compile(step, backend="eager", fullgraph=False)

    for ep in range(1, args.epochs + 1):
        loss_pde, loss_bc = step_c()
        if sched is not None:
            sched.step()
        if ep % args.eval_freq == 0 or ep == 1:
            with torch.no_grad():
                u_pred = model(X_eval)
                err = u_pred - u_eval_true
                linf = err.abs().max().item()
                l2 = (err ** 2).mean().sqrt().item()
            print(
                f"ep {ep:6d}  pde {loss_pde.item():.3e}  "
                f"bc {loss_bc.item():.3e}  Linf {linf:.3e}  L2 {l2:.3e}",
                flush=True,
            )

    if args.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=args.lbfgs_steps,
            line_search_fn="strong_wolfe",
        )
        X_int_lbfgs = torch.rand(args.n_int, 2, device=device, dtype=dtype, requires_grad=True)
        X_bd_lbfgs = boundary_points(n_per_side, device, dtype)

        def closure():
            lbfgs.zero_grad()
            _, lap = laplacian(model, X_int_lbfgs)
            loss_pde = ((-lap - forcing(X_int_lbfgs)) ** 2).mean()
            if args.hard_bc:
                loss = loss_pde
            else:
                u_bd = model(X_bd_lbfgs)
                loss = loss_pde + args.lam_bc * (u_bd ** 2).mean()
            loss.backward()
            return loss

        lbfgs.step(closure)
        print(f"L-BFGS finetune done ({args.lbfgs_steps} steps)", flush=True)

    # Always save the inner MLP state_dict so plot_pde.py can load it
    # whether the run used --hard-bc or not (plot_pde.py mirrors the flag).
    torch.save(base.state_dict(), HERE / "model_pde.pt")
    with torch.no_grad():
        u_pred = model(X_eval)
        err = u_pred - u_eval_true
        linf = err.abs().max().item()
        l2 = (err ** 2).mean().sqrt().item()
    print(f"final Linf {linf:.3e}  L2 {l2:.3e}", flush=True)


if __name__ == "__main__":
    main()

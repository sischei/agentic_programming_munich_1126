"""PINN solver for y'' + y = 0 on [0, 2*pi] with y(0)=1, y'(0)=0.

Analytical solution: y(x) = cos(x).
"""
import argparse
import math
from pathlib import Path

import torch
import torch.nn as nn

HERE = Path(__file__).parent

DTYPES = {"float32": torch.float32, "float64": torch.float64}


def init_xavier(module):
    if isinstance(module, nn.Linear):
        nn.init.xavier_uniform_(module.weight)
        nn.init.zeros_(module.bias)


class MLP(nn.Module):
    def __init__(self, hidden=32, n_layers=3):
        super().__init__()
        layers = [nn.Linear(1, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def derivatives(model, x):
    y = model(x)
    (y_x,) = torch.autograd.grad(
        y, x, torch.ones_like(y), create_graph=True, retain_graph=True
    )
    (y_xx,) = torch.autograd.grad(
        y_x, x, torch.ones_like(y_x), create_graph=True, retain_graph=True
    )
    return y, y_x, y_xx


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=15000)
    p.add_argument("--n-collocation", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
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

    model = MLP().to(device=device, dtype=dtype)
    if args.init == "xavier":
        model.apply(init_xavier)

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    sched = (
        torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
        if args.lr_schedule == "cosine" else None
    )

    x_coll = torch.linspace(0.0, 2 * math.pi, args.n_collocation,
                            device=device, dtype=dtype).unsqueeze(1)
    x_coll.requires_grad_(True)
    x0 = torch.zeros(1, 1, device=device, dtype=dtype, requires_grad=True)

    def step():
        opt.zero_grad()
        y_coll, _, y_xx = derivatives(model, x_coll)
        residual = y_xx + y_coll
        loss_pde = (residual ** 2).mean()

        y0, y0_x, _ = derivatives(model, x0)
        loss_ic = ((y0 - 1.0) ** 2 + y0_x ** 2).squeeze()

        loss = loss_pde + loss_ic
        loss.backward()
        if args.clip_grad > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        opt.step()
        return loss_pde.detach(), loss_ic.detach()

    step_c = torch.compile(step, backend="eager", fullgraph=False)

    x_eval = torch.linspace(0.0, 2 * math.pi, 1000,
                            device=device, dtype=dtype).unsqueeze(1)
    y_true = torch.cos(x_eval)

    for ep in range(1, args.epochs + 1):
        loss_pde, loss_ic = step_c()
        if sched is not None:
            sched.step()
        if ep % args.eval_freq == 0 or ep == 1:
            with torch.no_grad():
                y_pred = model(x_eval)
                linf = (y_pred - y_true).abs().max().item()
            print(
                f"ep {ep:5d}  pde {loss_pde.item():.3e}  "
                f"ic {loss_ic.item():.3e}  Linf {linf:.3e}",
                flush=True,
            )

    if args.lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            max_iter=args.lbfgs_steps,
            line_search_fn="strong_wolfe",
        )

        def closure():
            lbfgs.zero_grad()
            y_coll, _, y_xx = derivatives(model, x_coll)
            loss_pde = ((y_xx + y_coll) ** 2).mean()
            y0, y0_x, _ = derivatives(model, x0)
            loss_ic = ((y0 - 1.0) ** 2 + y0_x ** 2).squeeze()
            loss = loss_pde + loss_ic
            loss.backward()
            return loss

        lbfgs.step(closure)
        print(f"L-BFGS finetune done ({args.lbfgs_steps} steps)", flush=True)

    torch.save(model.state_dict(), HERE / "model.pt")
    with torch.no_grad():
        y_pred = model(x_eval)
        linf = (y_pred - y_true).abs().max().item()
    print(f"final Linf vs cos(x): {linf:.3e}", flush=True)


if __name__ == "__main__":
    main()

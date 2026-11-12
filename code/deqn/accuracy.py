"""The on-path accuracy check of a saved DEQN policy (also the script behind the /deqn-accuracy skill).

    python3 accuracy.py results/deqn_stochastic.pt            # the report
    python3 accuracy.py results/deqn_stochastic.pt --plot out.pdf   # the report and the verification figure

Loads the weights and the JSON sidecar (model, parameters, architecture), simulates
2,000 periods with the network's own policy, and reports: the Euler residual (mean, p50,
p90, p99, max) along that path, the distance to the reference policy (the reference solver at
the sidecar's beta and rho for the stochastic model, the closed form alpha beta for the
deterministic one), E[s] against the reference, and whether the path stayed inside the
training window. The training loss is not part of the report: it is not an accuracy measure.

If the path left the window, the policy gap is measured on the reference solver's own
simulated path instead (the reference solver's spline is clipped beyond its grid), so it is always a
meaningful number.

Verdict: ACCURATE if mean |r| < 1e-3, max |s - s_ref| < 1e-2 and the path is inside the
window; WRONG if mean |r| > 1e-2 or the path left the window; DEGRADED otherwise.

The figure (--plot) shows the same three checks: the policy against the reference across
the training window, the Euler residual along the simulated path against the thresholds,
and the simulated path against the window. It is the evidence behind the verdict.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for cand in HERE.parents:                       # works from deqn/ and from .claude/skills/deqn-accuracy/
    if (cand / "brock_mirman_reference.py").exists():
        if str(cand) not in sys.path:
            sys.path.insert(0, str(cand))
        break
import brock_mirman_reference as bm  # noqa: E402
from deqn import common  # noqa: E402


def check(path: str | Path) -> dict:
    net, meta = common.load_policy(path)
    p = common.params_of(meta)                  # the parameters the policy was trained with
    s_fn = common.policy_fn(net, meta)
    kss, sss = common.steady_state(p)
    a, b, d = p["alpha"], p["beta"], p["delta"]
    if meta["model"] == "stochastic":
        LZ, LK = common.simulate_path(s_fn, p)
        r = np.abs(common.euler_residual_np(s_fn, LZ, LK, p))
        mean_s = float(s_fn(LZ, LK).mean())
        half = meta.get("window", common.WINDOW_HALF_WIDTH)
        lo, hi = math.log(kss) - half, math.log(kss) + half
        in_window = bool(LK.min() > lo and LK.max() < hi)
        path_range = (float(LK.min()), float(LK.max()))
        ref_policy = bm.solve(b, p["rho"])
        if in_window:
            LZc, LKc, gap_where = LZ, LK, "the network's own path"
        else:                                   # compare where the reference is defined
            _, _, K_ref = bm.simulate(ref_policy, b, p["rho"], len(LZ), seed=7)
            LZc, LKc, gap_where = LZ, np.log(K_ref), "the reference solver's path"
        s_net, s_ref = s_fn(LZc, LKc), ref_policy(np.exp(LZc), np.exp(LKc))
        ref_name = f"reference policy solve({b}, {p['rho']})"
        Kg = np.linspace(lo, hi, 200)                       # the policy across the window, at three z
        sd = common.sd_logz(p)
        panel = dict(K=np.exp(Kg), curves=[(f"z = exp({k:+.0f} sd)" if k else "z = 1",
                                            s_fn(np.full_like(Kg, k * sd), Kg), ref_policy(np.exp(np.full_like(Kg, k * sd)), np.exp(Kg)))
                                           for k in (-2, 0, 2)], path=LK, r=r)
    else:
        lo, hi = meta.get("K_range", [0.1, 1.0])
        K = np.linspace(lo, hi, 1001)
        s_net, s_ref = s_fn(K), np.full_like(K, sss)
        mean_s, gap_where = float(s_net.mean()), "the grid"
        # residual of the deterministic Euler equation on the grid
        Y = K ** a; C = (1 - s_net) * Y; Kn = (1 - d) * K + s_net * Y
        Cn = (1 - s_fn(Kn)) * Kn ** a
        r = np.abs(1 - Cn / (b * C * (1 - d + a * Kn ** (a - 1))))
        # the own path from K0 = 0.5 must converge to K* inside the grid
        k = 0.5; ks = []
        for _ in range(200):
            k = (1 - d) * k + float(s_fn(np.array([k]))[0]) * k ** a; ks.append(k)
        in_window = bool(min(ks) > lo and max(ks) < hi)
        path_range = (float(min(ks)), float(max(ks)))
        ref_name = f"closed form s* = alpha beta = {sss:.4f}"
        panel = dict(K=K, curves=[("", s_net, s_ref)], path=np.log(np.array(ks)), r=r)
    ds = np.abs(s_net - s_ref)
    out = dict(file=str(path), model=meta["model"], reference=ref_name,
               residual=dict(mean=r.mean(), p50=np.median(r), p90=np.percentile(r, 90),
                             p99=np.percentile(r, 99), max=r.max()),
               policy_gap=dict(mean=ds.mean(), max=ds.max(), where=gap_where),
               mean_s=mean_s, mean_s_ref=float(s_ref.mean()),
               window=(lo, hi), path_range=path_range, in_window=in_window, panel=panel,
               where="the network's own path" if meta["model"] == "stochastic" else "the grid [%.1f, %.1f]" % (lo, hi))
    if not in_window or r.mean() > 1e-2:
        out["verdict"] = "WRONG"
    elif r.mean() < 1e-3 and ds.max() < 1e-2:
        out["verdict"] = "ACCURATE"
    else:
        out["verdict"] = "DEGRADED"
    return out


def format_report(o: dict) -> str:
    R, G = o["residual"], o["policy_gap"]
    return "\n".join([
        f"policy: {o['file']}  ({o['model']} model; reference: {o['reference']})",
        f"|Euler residual| on {o['where']}: mean {R['mean']:.2e}  p50 {R['p50']:.2e}  "
        f"p90 {R['p90']:.2e}  p99 {R['p99']:.2e}  max {R['max']:.2e}",
        f"|s - s_ref| on {G['where']}: mean {G['mean']:.2e}  max {G['max']:.2e};   "
        f"E[s] = {o['mean_s']:.4f} vs reference {o['mean_s_ref']:.4f}",
        f"path range [{o['path_range'][0]:.2f}, {o['path_range'][1]:.2f}] vs training window "
        f"[{o['window'][0]:.2f}, {o['window'][1]:.2f}]: {'inside' if o['in_window'] else 'OUTSIDE'}",
        f"verdict: {o['verdict']}",
    ])


def plot(o: dict, path: str | Path) -> Path:
    """Three panels: policy vs reference across the window, |residual| along the path, the path vs the window."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 9})
    pn, stoch = o["panel"], o["model"] == "stochastic"
    fig, ax = plt.subplots(1, 3, figsize=(7.0, 2.4))
    short = ["low $z$", "$z = 1$", "high $z$"]
    for i, (lab, s_net, s_ref) in enumerate(pn["curves"]):
        ax[0].plot(pn["K"], s_ref, "k--", lw=0.8, label="reference" if i == 0 else None)
        ax[0].plot(pn["K"], s_net, lw=1.1, color=f"C{i}", label="network" if i == 0 else None)
        if stoch:                                   # name the three curves at their right end, not in the legend
            ax[0].text(pn["K"][-1] * 1.015, s_net[-1], short[i], fontsize=6, va="center", color=f"C{i}")
    if stoch:
        ax[0].set_xlim(pn["K"][0], pn["K"][-1] * 1.22)
    ax[0].set_xlabel("$K$"); ax[0].set_ylabel("savings rate"); ax[0].legend(frameon=False, fontsize=7, loc="best")
    ax[0].set_title("policy vs reference", fontsize=9)
    r = pn["r"]
    ax[1].semilogy(np.arange(len(r)), r + 1e-16, lw=0.5, color="0.3")
    ax[1].axhline(1e-3, color="darkred", ls="--", lw=0.8, label="test threshold, mean")
    ax[1].axhline(r.mean(), color="C0", lw=0.9, label="mean " + f"{r.mean():.1e}".replace("e-0", "e-"))
    ax[1].set_ylim(top=max(r.max(), 1e-3) * 300)
    ax[1].set_xlabel("period" if stoch else "grid point"); ax[1].set_ylabel("|Euler residual|")
    ax[1].set_title("Euler residual, " + ("own path" if stoch else "grid"), fontsize=9); ax[1].legend(frameon=False, fontsize=7, loc="upper left")
    lo, hi = o["window"]
    if stoch:
        ax[2].axhspan(lo, hi, facecolor="0.88", edgecolor="0.5", lw=0.8, label="training window")
        ax[2].plot(pn["path"], lw=0.6, label="simulated $\\log K$")
        ax[2].set_ylabel("$\\log K$")
    else:
        ax[2].axhspan(np.log(lo), np.log(hi), color="0.88", label="training grid (log)")
        ax[2].plot(pn["path"], lw=0.9, label="$\\log K_t$ from $K_0 = 0.5$")
        ax[2].set_ylabel("$\\log K$")
    ax[2].set_xlabel("period"); ax[2].set_title("where the policy goes", fontsize=9)
    if stoch: ax[2].set_ylim(min(lo, pn["path"].min()) - 0.3, max(hi, pn["path"].max()) + 0.75)
    ax[2].legend(frameon=False, fontsize=7, loc="upper right")
    fig.suptitle(f"{Path(o['file']).name}: verdict {o['verdict']}", fontsize=9, x=0.01, ha="left")
    fig.tight_layout()
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path); plt.close(fig)
    return path


if __name__ == "__main__":
    args = sys.argv[1:]
    fig_path = None
    if "--plot" in args:
        i = args.index("--plot"); fig_path = args[i + 1]; del args[i:i + 2]
    for arg in args or ["results/deqn_stochastic.pt"]:
        o = check(arg)
        print(format_report(o))
        if fig_path:
            out = plot(o, fig_path if len(args) <= 1 else Path(fig_path).with_name(Path(fig_path).stem + "_" + Path(arg).stem + Path(fig_path).suffix))
            print(f"figure: {out}")
        print()

"""
Euler-error check of the reference solver: an earlier one-prompt demo, no longer on the slides.

One prompt, one publication-ready table, one figure, on the stochastic
Brock-Mirman model used throughout the lecture. The reference solver in this
folder (brock_mirman_reference.py) solves the model
at four values of the persistence rho; the relative Euler-equation error is
evaluated on 2,000 simulated states per rho, once for the default grid and
once for a deliberately coarse one (12 capital points, 5 productivity points),
and the savings-rate policy is drawn against capital.

    python3 euler_check_demo.py

writes outputs/euler_table.tex and outputs/euler_figure.pdf. Runtime about
10 s on a laptop CPU. Verified: every default-grid mean |Euler error| is below
1e-6, and the coarse grid is worse by three to four orders of magnitude.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import brock_mirman_reference as bm  # noqa: E402

BETA = 0.96
RHOS = [0.50, 0.70, 0.90, 0.99]
OUTDIR = HERE / "outputs"
OUTDIR.mkdir(exist_ok=True)


def main():
    rows, policies = [], {}
    for rho in RHOS:
        pol = bm.solve(BETA, rho)
        coarse = bm.solve(BETA, rho, n_k=12, n_z=5, tol=1e-5)
        Y, C, K = bm.simulate(pol, BETA, rho, T=2000, seed=7)
        z = Y / K ** bm.ALPHA
        ee = np.abs(bm.euler_error(pol, BETA, rho, z, K))
        ee_c = np.abs(bm.euler_error(coarse, BETA, rho, z, K))
        rows.append((rho, ee.mean(), ee.max(), ee_c.mean(), ee_c.max(), (1.0 - C / Y).mean(), pol.wall))
        policies[rho] = pol
        assert ee.mean() < 1e-6, f"mean Euler error {ee.mean():.1e} at rho = {rho}"

    # --- table ---------------------------------------------------------
    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"& \multicolumn{2}{c}{default grid ($120 \times 31$)} & \multicolumn{2}{c}{coarse grid ($12 \times 5$)} & & \\",
             r"$\varrho$ & mean $|$Euler$|$ & max & mean $|$Euler$|$ & max & mean $s$ & seconds \\",
             r"\midrule"]
    for rho, mean_ee, max_ee, mean_c, max_c, s_mean, wall in rows:
        lines.append(f"{rho:.2f} & {mean_ee:.1e} & {max_ee:.1e} & {mean_c:.1e} & {max_c:.1e} & {s_mean:.4f} & {wall:.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (OUTDIR / "euler_table.tex").write_text("\n".join(lines) + "\n")

    # --- figure --------------------------------------------------------
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4))
    K_line = np.linspace(2.0, 8.0, 200)
    for rho, col in zip(RHOS, plt.cm.viridis(np.linspace(0.1, 0.9, len(RHOS)))):
        axes[0].plot(K_line, policies[rho](1.0, K_line), color=col, lw=1.8, label=rf"$\varrho = {rho}$")
        ee_line = np.abs(bm.euler_error(policies[rho], BETA, rho, np.ones_like(K_line), K_line))
        axes[1].semilogy(K_line, ee_line, color=col, lw=1.4, label=rf"$\varrho = {rho}$")
    axes[0].set_xlabel("$K$"); axes[0].set_ylabel("savings rate $s(z = 1, K)$")
    axes[0].set_title("reference policy"); axes[0].legend(frameon=False, fontsize=8)
    axes[1].set_xlabel("$K$"); axes[1].set_ylabel("relative Euler error"); axes[1].set_title("accuracy at $z = 1$")
    for ax in axes:
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "euler_figure.pdf")

    print(f"{'rho':>5s} {'default mean':>13s} {'max':>9s} {'coarse mean':>12s} {'max':>9s} {'mean s':>8s}")
    for rho, mean_ee, max_ee, mean_c, max_c, s_mean, _ in rows:
        print(f"{rho:>5.2f} {mean_ee:>13.1e} {max_ee:>9.1e} {mean_c:>12.1e} {max_c:>9.1e} {s_mean:>8.4f}")
    print(f"\nwrote {OUTDIR / 'euler_table.tex'} and {OUTDIR / 'euler_figure.pdf'}")


if __name__ == "__main__":
    main()

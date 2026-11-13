"""The silent failure of Part VIII: the Step-4 recipe with one argument changed.

Identical to solution/deqn_stochastic.py except that the output bias is not initialized
at the steady-state savings rate (s_init=None, the course notebook's default). The
training loss ends LOWER than in the good run; the policy saves 0.94 of output instead
of 0.25: gradient descent landed on the over-saving branch of the Euler equation, which
the box loss cannot tell from the equilibrium one. See silent_failure.md.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deqn.solution import deqn_stochastic as good  # noqa: E402

if __name__ == "__main__":
    out = good.solve(s_init=None,                                   # <- the one difference
                     out=Path(__file__).resolve().parent / "outputs" / "deqn_stochastic_box.pt")
    R = good.report(out)
    print(f"trained in {R['wall']:.1f} s; training loss (last 100 steps) {R['train_loss']:.2e}")
    print(f"on-path |Euler residual|: mean {R['mean_abs_r']:.2e}, max {R['max_abs_r']:.2e}")
    print(f"max |s - s_ref| on {R['gap_where']} {R['max_abs_ds']:.2e}; E[s] {R['mean_s']:.4f} vs reference {R['mean_s_ref']:.4f}")
    print(f"simulated log K in [{R['logK_min']:.2f}, {R['logK_max']:.2f}], training window [{R['window'][0]:.2f}, {R['window'][1]:.2f}]")

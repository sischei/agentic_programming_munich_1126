# The silent failure: a loss that converged, a policy that did not

`stochastic_box.py` is `solution/deqn_stochastic.py` with one argument changed:
`s_init=None`, the course notebook's default, instead of `s_init=s*`.

| | good run (`solution/`) | wrong run (`stochastic_box.py`) |
|---|---|---|
| training loss, last 100 steps | 6.0e-7 | **2.9e-7** |
| on-path mean \|Euler residual\| | 1.4e-4 | 9.2e-2 |
| E[s] (reference: 0.252) | 0.2523 | 0.937 |
| max \|s - s_ref\| on the reference solver's path | 2.3e-3 | 0.60 |
| simulated log K (window [0.96, 1.96]) | [1.08, 1.81] | [3.14, 3.72] |

What happened. The Euler equation is a necessary condition, not a sufficient one. As a
functional equation on a box it has more than one solution: the equilibrium policy, and
an over-saving branch on which the savings rate rises with capital, consumption shrinks
toward zero and capital grows past the golden rule. The transversality condition rules
that branch out; a loss built from the Euler residual alone does not. Inside the box the
wrong network saves 0.73 to 0.82 of output, rising in K (the reference policy falls from 0.33
to 0.16), and its residual there is as small as the good run's: mean |r| 3.8e-4 on the
box, mean squared residual 2.7e-7. Gradient descent had found a solution of the equation, just not
the equilibrium one. Simulated with its own policy, the economy leaves the box within a
few periods and settles at five to ten times the steady-state capital (K between 23 and
41 against K* = 4.3, golden rule K = 7.4), where the residual is 0.1.

Why the initialisation matters. The untrained network returns s about 0.5 everywhere;
from there, descent lands on the over-saving branch. Starting the output bias at the
steady-state savings rate puts the untrained policy next to the equilibrium branch, and
descent stays on it. The initialisation selects the branch; the loss cannot.

Why nothing in the training loop revealed it. The loss went down, faster than in the
good run. Nothing in the loop looks at where the policy sends the economy.

What caught it. Simulating 2,000 periods with the network's own policy and (i) checking
that the path stays inside the training window, (ii) evaluating the residual there,
(iii) comparing the policy with the reference solver. These are four of the five tests
and the whole of the `/deqn-accuracy` skill. The `loss-balancing-reviewer` subagent would
not have caught it: there is only one loss term to balance.

Where this sits in the literature. Azinovic, Gaegauf and Scheidegger (2022) draw the
training states by simulating the network's own policy, so the loss is evaluated on the
economy the policy generates and the over-saving branch raises the loss there. The fixed
box of the lecture is a simplification of their method, and it is what makes the
failure possible. The on-path check is the test that survives either way.

The fix. Start the output bias at the steady-state savings rate (`make_net(s_init=s*)`),
so descent starts on the equilibrium branch. The second half of the fix is the check
itself: it stays in the tests.

Redraw the figures: `python3 plot_silent_failure.py`.

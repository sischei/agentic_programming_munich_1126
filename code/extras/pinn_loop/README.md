# The loop demo (Part VII, live demo 4)

A seeded workspace for `/goal`: an empty `pinn_hard_bc.py` and three acceptance
tests. The goal is to make `pytest -q` green. The tests are the specification;
the goal text only restates them.

```bash
cp -r /path/to/agentic_programming_munich_1126/code/extras/pinn_loop ~/research/pinn-loop
cd ~/research/pinn-loop && git init -q && git add . && git commit -qm "seed: stub + acceptance tests"
pytest -q          # must fail: NotImplementedError
claude
```

Then, inside Claude Code, paste as one block:

```text
/goal `pytest -q` in this folder exits 0 with 3 passed. To get there, implement pinn_hard_bc.py: u'' + u = 0 on [0, pi/2], u(0) = 0, u(pi/2) = 1, exact solution sin x. Use the trial solution u = 2x/pi + x (pi/2 - x) N(x) so the boundary conditions hold by construction and the loss is the residual alone; network 1-32-32-1 tanh, 256 fresh uniform collocation points per step, Adam 1e-3, 3000 steps, seed 42, float64, PyTorch. solve(seed) returns {"max_err": max |u - sin x| on 1001 points, "u": callable}. Each turn: run pytest -q and show the output, fix one thing, commit. Report ALL_TESTS_PASS when pytest exits 0. Stop after 6 turns (count your commits in git log) and write notes/blocked.md instead.
```

`/goal` keeps the session working turn after turn; after each turn a separate
small model reads the transcript and decides whether the condition holds. `/goal`
alone shows turns, time and token spend; `/goal clear` ends it. Docs:
https://code.claude.com/docs/en/goal

## The numbers, and where they come from

| number | meaning | produced by |
|---|---|---|
| `5.54e-4` | `STORED_SOFT_BC_ERROR` in the tests: the same network and recipe with the boundary conditions as a penalty (weight 1), seed 42 | `python3 reference/soft_bc_baseline.py` (about 10 s) |
| `6.70e-5` | what the hard-BC trial solution reaches with the recipe in the goal, seed 42 | `../pinn_loop_solution/pinn_hard_bc.py`, kept outside this folder so the agent cannot read it; it passes the three tests in about 35 s |

One training run takes about 10 s on a laptop CPU; the three tests about 35 s
(three training runs). Expect two or three turns and a few minutes. Dry-run the
goal before the lecture and note turns, wall-clock and `max_err` on the slide.

Before the demo, `pytest -q` must fail with `NotImplementedError` (the stub);
after it, `pytest -q` prints `3 passed`, `git log --oneline` shows one commit per
turn, and `/goal` shows the achieved condition.

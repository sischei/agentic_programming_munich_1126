# Walk-through guide: every file shown in the lecture, in order

This replaces take-home exercises. Each row is a file (or a command) that is
opened and stepped through during the lecture, with the slide it belongs to,
what to look at, and the command that reproduces it. Slide numbers refer to
`slides/Agentic_Programming_Munich.pdf` (78 frames; the last one is an appendix). Paths are relative to the
repository root. The demo project is `demo/` at the repository root, built by
`bash code/setup_demo_project.sh` (git-ignored, with its own git history); every
live session runs there: `cd demo && claude`. The prompts are in
[`prompts.md`](prompts.md).

## The lead example

A deep equilibrium net (DEQN) for the Brock–Mirman growth model, built step by step
with the agent and checked at every step against a known answer. The steps, where
they appear, and what they are checked against:

| Step | Part | What the agent builds | Checked against | Our version |
|---|---|---|---|---|
| 1 | III | the deterministic DEQN (delta = 1), ported from TensorFlow to PyTorch | closed form s* = alpha beta | `code/deqn/deterministic.py` |
| 2 | IV | the stochastic extension as a six-component prompt | reference policy `solve(0.96, 0.9)` | `code/deqn/solution/deqn_stochastic.py` |
| 3 | VI | the `/deqn-accuracy` skill: the on-path check of a saved policy | reference policy / closed form | `code/deqn/accuracy.py` |
| 4 | VII | `/goal` builds the stochastic solver against five tests | `code/deqn/tests/`, `reference/` | `code/deqn/solution/` |
| 5 | VIII | the plan of Step 4 and its reconciliation | measured numbers | `code/deqn/plan_deqn.md` |
| 6 | VIII | the silent failure: the recipe without the bias initialisation | on-path check | `code/deqn/stochastic_box.py`, `silent_failure.md` |
| 7 | VIII | four loss kernels → table + figure; the estimation pipeline with the DEQN at rho_hat | the reference solver; the truth file | `code/deqn/loss_kernels.py`; `prompts.md` §14 |

## How the numbering fits together

The slides carry seven **walk-through** badges (WT 1–7) and five **live demos**
(D1–D5); four demos sit on a badged slide, D2 and D5 do not. Every prompt is a
section (§) of `prompts.md`.

| Demo | Slides | Walk-through | Prompt | What happens |
|---|---|---|---|---|
| D1 | 14, 19 | WT 1 | §1 | stack check; the demo project is built and summarised |
| D2 | 25 | | §4 | Step 1 in plan mode: port, check, figure, note |
| D3 | 42–43 | WT 3, WT 4 | §9 | the accuracy skill on the Step-1 policy, then the verifier subagent |
| D4 | 52–55 | WT 6 | §11 | `/goal` builds the stochastic solver against five tests |
| D5 | 61 | | §13 | one prompt → table + figure: four loss kernels (optional live run) |
| | 39 | WT 2 | §8 | the demo project's `CLAUDE.md` next to a paper's |
| | 45 | WT 5 | §9 | deny rules and hooks |
| | 59 | WT 7 | | the plan of Step 4, and how it was reconciled |

## Block 1 (70 min)

| # | Slide | WT | Part | File / command | What to look at |
|---|---|---|---|---|---|
| 1 | 10 | | I | `code/brock_mirman_reference.py` | The docstring: model, five exported functions (`steady_state`, `solve`, `simulate`, `euler_error`, `moments`). `solve(beta, rho)(z, K)` is the savings rate: the reference every DEQN step is checked against. Nobody needs to read past the docstring. |
| 2 | 10 | | I | `code/data/bm_panel.csv`, `code/data/bm_panel_truth.csv`, `code/deqn/README.md` | The observed panel (2,500 rows) and the hidden truth (5 rows); the README of the lead example with the step table and the measured numbers. |
| 3 | 14 | | II | `claude --version`, `claude doctor` | The stack check. **D1.** |
| 4 | 19 | WT 1 | II | `code/setup_demo_project.sh`, then `demo/` | Run the script, open the tree: `CLAUDE.md` (from `code/CLAUDE_md_demo.md`), `deqn/common.py`, `deqn/source/`, the stub `deqn/stochastic.py`, `tests/`, `reference/`, `data/raw/`, `.claude/`. `python3 -m pytest -q tests/` fails with `NotImplementedError`. Then `claude` and §1. |
| 5 | 20 | | II | `code/deqn/source/deqn_tf_deterministic.py` | `compute_cost`, lines 51–64: the relative Euler error of the course notebook, in TensorFlow. This is what Step 1 ports. (Not run: TensorFlow is not a requirement.) |
| 6 | 22–24 | | III | `code/deqn/deterministic.py` | The port, as the transcript of slide 22 and the Level-3 prompt of slide 24 produce it: `residual()` (lines 30–40), `solve()`, `check()`. `python3 code/deqn/deterministic.py` (3 s) prints the max error against alpha beta, 4.9e-4, and writes `outputs/deqn_deterministic.{pt,json}` and `deqn_step1_figure.pdf`. |
| 7 | 25 | | III | §4, plan mode | **D2.** Read the plan aloud before approving: closed form only in the check, `make_net` reused, torch seeded first, "solved" judged by the max error. Afterwards `git diff HEAD`, `cat notes/deqn_step1.md`, open `figures/deqn_step1.pdf`. Expected numbers in §4. |
| 8 | 30–34 | | IV | `code/prompts.md` §7 | The before/after prompt (Step 2), the constraint lines and the six templates. Point at the `[VERIFY]` line: the network's own path, the reference, the window; the training loss is not in it. Template 5: the shape bug in `code/deqn/source/deqn_tf_stochastic.py`, line 66. |

## Block 2 (70 min)

| # | Slide | WT | Part | File / command | What to look at |
|---|---|---|---|---|---|
| 9 | 39 | WT 2 | V | `code/CLAUDE_md_template.md` (the paper example) and `code/CLAUDE_md_demo.md` (= `demo/CLAUDE.md`) | In the template: Repository Structure (`data/raw/ <- READ ONLY`), Current Status, Decisions Made, and the stale line (JAX in the log, TensorFlow in the package list). In the demo file: the two rules, the DEQN notation, "accuracy means ... never the training loss". Live: §8. |
| 10 | 42 | WT 3 | VI | `code/skills/deqn-accuracy/SKILL.md`, `code/deqn/accuracy.py` | Frontmatter, the four steps, the constraints; the script is the same one the tests' logic mirrors. **D3:** `/deqn-accuracy results/deqn_deterministic.pt` (§9, 5 s); open the report under `notes/`. Then `/data-diagnostics data/bm_panel.csv` if time allows. |
| 11 | 42 | WT 3 | VI | `code/skills/data-diagnostics/SKILL.md`, `code/skills/strategic-revision/SKILL.md` | The same shape on data and on referee reports; one line each on the slide. |
| 12 | 43 | WT 4 | VI | `code/agents/verifier.md` | The whole file (28 lines). Live: the verifier prompt of §9 on the accuracy report. Watch the main session delegate and receive a summary. |
| 13 | 44 | | VI | `code/agents/solution_accuracy_checker.md`, `code/agents/econometrics_reviewer.md` | The four checks (check 2 now cites the lecture's own silent failure); checklist A–E (effective first-stage F, not F < 10). Compare the `tools:` lines: read-only agents have no `Bash`, `Edit`, `Write`. |
| 14 | 45 | WT 5 | VI | `code/hooks/settings.json` (= `demo/.claude/settings.json`) | The `permissions.deny` rules (no reading `data/raw/` or `reference/`), the `PreToolUse` block (no edits under `data/raw/`, `reference/`, `tests/`; exit code 2), the audit log. Optional live (§9): "Relax the threshold in tests/..." refused by the hook; "Show me reference/baseline.md" refused by the deny rule. Then `cat .claude/audit.log`. |
| 15 | 50–51 | | VII | `code/deqn/solution/deqn_stochastic.py` | `residual()`, lines 32–47: the residual shown on the slide; `train()` below it. Kept outside the demo project. |
| 16 | 52 | WT 6 | VII | `code/deqn/tests/test_deqn_stochastic.py`, `tests/conftest.py`, `code/deqn/stochastic.py`, `code/deqn/reference/baseline.md` | The five tests are the spec; `conftest.py` is their own residual and simulation, independent of the solver; the stub raises `NotImplementedError`; `baseline.md` says where each threshold comes from. **D4:** set the `/goal` (§11) in a second terminal in `demo/` at the start of Part VII. |
| 17 | 53–54 | | VII | `code/prompts.md` §11 | The goal taken apart (condition, problem, recipe, interface, each turn, cap) and the run turn by turn. While these slides are shown, the goal is running; at the end of the part switch to the second terminal: `python3 -m pytest -q tests/` (5 passed), `git log --oneline` (one commit per turn), `/goal` (turns, time, tokens), `/deqn-accuracy results/deqn_stochastic.pt`. |
| 18 | 55 | | VII | `code/deqn/accuracy.py --plot`, `slides/figures/agentic_deqn_verify_{good,bad,step1}.pdf` | The three panels behind the verdict: policy vs reference across the window, the residual along the own path against the threshold, the path against the window. Top the `/goal` policy (ACCURATE), bottom the silent failure (WRONG). `cd code && python3 deqn/accuracy.py deqn/outputs/deqn_stochastic.pt --plot out.pdf` redraws it. |
| 19 | 57 | | VII | `code/prompts.md` §12 | The headless command, the headless goal, and the fresh-context shell loop. |
| 20 | 59 | WT 7 | VIII | `code/deqn/plan_deqn.md` | Problem, recipe, metrics and targets; "First implementation, and what the plan did not foresee" (the four numbers of the wrong run); "Revised recipe (achieved)"; the sanity rule. |
| 21 | 60 | | VIII | `code/deqn/silent_failure.md`, `code/deqn/stochastic_box.py`; `cd code/deqn && python3 stochastic_box.py && python3 accuracy.py outputs/deqn_stochastic_box.pt` | The one-line diff (`s_init=None`); the run reproduces the left column of the slide in 11 s (training loss 2.9e-7, E[s] 0.94, log K in [3.1, 3.7]); the verdict WRONG. `python3 plot_silent_failure.py` redraws both figures from the saved policies. |
| 22 | 61 | | VIII | `code/deqn/loss_kernels.py`, `code/deqn/outputs/deqn_kernel_table.tex`, `deqn_kernel_figure.pdf` | The prompt (§13), the `KERNELS` dict (lines 24–29), the committed table and figure. **D5 (optional):** `python3 code/deqn/loss_kernels.py`, 45 s; `diff` the table against the committed one (only the seconds column should differ). |
| 23 | 62 | | VIII | `code/prompts.md` §14 | The six pipeline prompts: the SMM estimate of rho with the reference solver, the DEQN re-solved at rho_hat as the closing check, `verify.py` and `run_all.py`. |
| 24 | 75 | | Wrap-up | `README.md`, this file | How to replay everything. |

## Instructor timing sheet (150 min)

| Clock | Part | Minutes | Content |
|---|---|---|---|
| 0:00 | Intro | 5 | Title, why now, today's plan (slides 1-4) |
| 0:05 | I | 15 | Tiers, context window, where the agent helps, workflow, the running example and the step ladder (5-12) |
| 0:20 | II | 10 | Stack, commands, models and effort, git, **D1** + WT 1 (13-20) |
| 0:30 | III | 25 | Feedback loop on Step 1, prompt levels, the Level-3 port, **D2 (12 min)**, notebooks, parallel, context (21-28) |
| 0:55 | IV | 15 | Anatomy (Step 2), constraints, templates, failure modes (29-35) |
| 1:10 | Break | 10 | (36) |
| 1:20 | V | 8 | CLAUDE.md, WT 2 (37-39) |
| 1:28 | VI | 15 | Map, the accuracy skill + verifier **D3 (5 min)**, specialists, deny rules and hooks, MCP (40-46) |
| 1:43 | VII | 15 | Closing the loop, how /goal works, the demo problem, **D4 (goal set at 1:45, shown at 1:57)**, the goal taken apart, turn by turn, the verification panels, guardrails (47-57) |
| 1:58 | VIII | 12 | The plan and its reconciliation, the silent failure, **D5 (optional)**, pipeline, replication (58-63) |
| 2:10 | IX | 15 | Verification, disclosure, referees/supervisors/students, replication packages, governance, six habits (64-70) |
| 2:25 | Wrap-up | 5 | Quick reference, takeaways, ten tips, conclusion, references (71-76) |

If running late: drop slides 26-27 (notebooks, worktrees), 63 (replication) and 74 (tips); never drop D2 or D4.

## Pre-lecture checklist

- [ ] `claude --version` and `claude doctor` on the presenting laptop; logged in; `/model` set (Opus 5.5 for the demos, or Sonnet 5.5 to keep cost down).
- [ ] `rm -rf demo && bash code/setup_demo_project.sh`; open a terminal in `demo/`, font size large enough for the back row. `jq` installed (the hooks use it; `ruff` and `latexmk` for the other two hooks, or delete those from `.claude/settings.json`).
- [ ] `python3 -c "import torch"`; then `python3 code/deqn/solution/deqn_stochastic.py` once on the presenting laptop and compare with `code/deqn/reference/baseline.md`: only the wall-clock should differ (if mean |r| > 3e-4 or the run takes more than 45 s, note it on the loop slide).
- [ ] `cd demo && python3 -m pytest -q tests/` fails with `NotImplementedError`.
- [ ] Dry-run D2 (plan mode, §4) once in `demo/`; note how long the plan takes to appear; then `git checkout .` and `git clean -fd` in `demo/`.
- [ ] Dry-run the `/goal` (§11) once in `demo/`; write the measured turns, wall-clock and E[s] into the transcript on slide 54. Then reset: `git checkout deqn/stochastic.py && rm -f results/deqn_stochastic.*`, or simply rebuild `demo/`. In the default permission mode the goal asks before running `pytest`; allow it with `/permissions` or run the demo in auto mode.
- [ ] A second terminal tab ready for the goal; `/cost` noted at the start so the total spend of the lecture can be shown at the end.
- [ ] `slides/Agentic_Programming_Munich.pdf` open, presenter mode; the figures live in `slides/figures/`.

## Files not stepped through, kept for reference

| File | Why it is here |
|---|---|
| `code/agents/code_reviewer.md`, `doc_generator.md`, `test_writer.md`, `monte_carlo_designer.md`, `loss_balancing_reviewer.md` | The rest of the subagent set installed by the setup script; mentioned on slides 44 and 71. |
| `code/deqn/source/deqn_tf_stochastic.py` | The stochastic course notebook flattened, with its shape bug kept; the input of template 5 (slide 34). Not run. |
| `code/deqn/common.py` | The shared module shipped to `demo/deqn/`; its docstring is read in D1. |
| `code/generate_synthetic_data.py`, `code/euler_check_demo.py`, `code/outputs/` | The data generator (10 s) and the Euler-error check of the reference solver with its committed table and figure; the latter was the one-prompt demo of an earlier version of this lecture. |
| `paper/2607.02134v2.pdf` (git-ignored; arXiv:2607.02134), the skill at github.com/PredictiveScienceLab/paper-replication-paper | Hans and Bilionis (2026): the replication workflow of slide 63 and the appendix slide; `prompts.md` §15 has the install lines and both prompts. Not run in the lecture (hours per paper). |
| `code/extras/` | The PINN material of the earlier version (the `/goal` on u'' + u = 0, the 2-D Poisson plan and session notes); see its README. No slide refers to it. |

# Walk-through guide: every file shown in the lecture, in order

This replaces take-home exercises. Each row is a file (or a command) that is
opened and stepped through during the lecture, with the slide it belongs to,
what to look at, and the command that reproduces it. Slide numbers refer to
`slides/Agentic_Programming_Munich.pdf` (65 frames). Paths are relative to the
repository root unless they start with `~/research/`.

Before starting at home: `bash code/setup_demo_project.sh` (creates
`~/research/munich-demo`), then `cd ~/research/munich-demo && claude`.
The prompts are in [`prompts.md`](prompts.md).

## How the numbering fits together

The slides carry seven **walk-through** badges (WT 1–7), five of which are **live
demos** (D1–D5); every prompt is a section (§) of `prompts.md`.

| Demo | Slides | Walk-through | Prompt | What happens |
|---|---|---|---|---|
| D1 | 11, 14 | WT 1 | §1 | stack check; the demo project is built and summarised |
| D2 | 19 | | §4 | a research task in plan mode |
| D3 | 35–36 | WT 3, WT 4 | §9 | a skill, then the verifier subagent |
| D4 | 44–46 | WT 6 | §11 | `/goal` builds the solver against three tests |
| D5 | 51 | | §13 | one prompt → table + figure (optional live run) |
| | 32 | WT 2 | §8 | the filled-in `CLAUDE.md` |
| | 38 | WT 5 | | deny rules and hooks |
| | 49 | WT 7 | | the plan the agent wrote, and how it was reconciled |

## Block 1 (70 min)

| # | Slide | WT | Part | File / command | What to look at |
|---|---|---|---|---|---|
| 1 | 9 | | I | `code/data/bm_panel.csv`, `code/data/bm_panel_truth.csv` | The observed panel (2,500 rows) and the hidden truth (5 rows). The truth is the verifier for everything that follows. |
| 2 | 9 | | I | `code/brock_mirman_reference.py` | The docstring: model, five exported functions (`steady_state`, `solve`, `simulate`, `euler_error`, `moments`). Nobody needs to read past the docstring. |
| 3 | 11 | | II | `claude --version`, `claude doctor` | The stack check. **D1.** |
| 4 | 14 | WT 1 | II | `code/setup_demo_project.sh`, then `~/research/munich-demo/` | Run the script, open the tree: `CLAUDE.md` (from `code/CLAUDE_md_demo.md`), `data/raw/` (stand-in for confidential data), `.claude/skills`, `.claude/agents`, `.claude/settings.json`. Then `claude` and §1. |
| 5 | 14 | WT 1 | II | `code/generate_synthetic_data.py` | Lines 28-51: the five `rho`s and seeds, the loop over `bm.solve` / `bm.simulate`, the "truth file is for verification only" print. `python3 code/generate_synthetic_data.py` regenerates the data in 10 s. |
| 6 | 19 | | III | §4, plan mode | **D2.** Read the plan aloud before approving. Afterwards `git diff HEAD`, `cat notes/data_description.md`, open `figures/panel_overview.pdf`. Expected numbers in §4. |
| 7 | 24–27 | | IV | `code/prompts.md` §7 | The before/after prompt, the constraint lines and the six templates. Point at the `[VERIFY]` line: the truth file is read after the table is written, never before. |

## Block 2 (70 min)

| # | Slide | WT | Part | File / command | What to look at |
|---|---|---|---|---|---|
| 8 | 32 | WT 2 | V | `code/CLAUDE_md_template.md` (the paper example) and `code/CLAUDE_md_demo.md` (= `~/research/munich-demo/CLAUDE.md`) | In the template: Repository Structure (`data/raw/ <- READ ONLY`), Current Status, Do Not Touch, Decisions Made, and the stale line (JAX in the log, TensorFlow in the package list). In the demo file: the truth-file rule. Live: §8. |
| 9 | 35 | WT 3 | VI | `code/skills/data-diagnostics/SKILL.md` | Frontmatter (`name`, `description`), the five steps, the constraints. **D3:** `/data-diagnostics data/bm_panel.csv` (§9); open the report under `notes/`. |
| 10 | 35 | WT 3 | VI | `code/skills/strategic-revision/SKILL.md` | Step 5, the revision-plan template: blocks, tags, conflicts flagged. |
| 11 | 36 | WT 4 | VI | `code/agents/verifier.md` | The whole file (28 lines). Live: the verifier prompt of §9. Watch the main session delegate and receive a summary. |
| 12 | 37 | | VI | `code/agents/solution_accuracy_checker.md`, `code/agents/econometrics_reviewer.md` | The four checks; checklist A-E (effective first-stage F, not F < 10). Compare the `tools:` lines: read-only agents have no `Bash`, `Edit`, `Write`. |
| 13 | 38 | WT 5 | VI | `code/hooks/settings.json` (= `~/research/munich-demo/.claude/settings.json`) | The `permissions.deny` rules (no reading `data/raw/`), the `PreToolUse` block (no edits, exit code 2), the audit log. Optional live: `Append a line to data/raw/test.txt` in the demo project; both refuse. Then `cat .claude/audit.log`. No `Stop` auto-commit: the diff is reviewed by a person. |
| 14 | 44 | WT 6 | VII | `code/pinn_loop/tests/test_pinn_hard_bc.py`, `code/pinn_loop/pinn_hard_bc.py`, `code/pinn_loop/reference/soft_bc_baseline.py` | The three tests are the spec; the stub raises `NotImplementedError`; the threshold `5.54e-4` comes from the baseline script. **D4:** set the `/goal` (§11) in a second terminal at the start of Part VII. |
| 15 | 45–46 | | VII | `code/prompts.md` §11 | The goal taken apart (condition, problem, recipe, interface, each turn, cap) and the run turn by turn. While these slides are shown, the goal is running; at the end of the part switch to the second terminal: `pytest -q` (3 passed), `git log --oneline` (one commit per turn), `/goal` (turns, time, tokens). |
| 16 | 47 | | VII | `code/prompts.md` §12 | The headless command, the headless goal, and the fresh-context shell loop. |
| 17 | 49 | WT 7 | VIII | `code/ode_solver/plan_pde.md` | Problem, the manufactured-solution check, metrics, the two boundary-condition strategies, "Achieved metrics" (measured after the run: the plan was reconciled), the verification section with the sanity rule (a zero network gives Linf = 1). The code it led to: `solver_pde.py` (`HardBCWrapper`, lines 43-53; the soft/hard branch, lines 140-158). |
| 18 | 50 | | VIII | `code/ode_solver/session.md`; `cd code/ode_solver && python3 plot_pde.py && python3 plot_pde.py --hard-bc` | "Gotchas hit during the session" (the last section). The two commands redraw `solution_pde.png` (soft penalty, halo) and `solution_pde_hardbc.png` (hard constraint) from the saved weights, no retraining. |
| 19 | 51 | | VIII | `code/euler_check_demo.py`, `code/outputs/euler_table.tex`, `code/outputs/euler_figure.pdf` | The prompt (§13), the core loop (lines 37-48), the committed outputs. **D5 (optional):** `python3 code/euler_check_demo.py`, 10 s; `diff` the table against the committed one (only the seconds column differs). |
| 20 | 52 | | VIII | `code/prompts.md` §14 | The six pipeline prompts, `verify.py` and `run_all.py`. |
| 21 | 65 | | Wrap-up | `README.md`, this file | How to replay everything. |

## Instructor timing sheet (150 min)

| Clock | Part | Minutes | Content |
|---|---|---|---|
| 0:00 | Intro | 5 | Title, why now, today's plan (slides 1-3) |
| 0:05 | I | 15 | Tiers, context window, where the agent helps, workflow, running example (4-9) |
| 0:20 | II | 10 | Stack, commands and cost, git, **D1** + WT 1 (10-14) |
| 0:30 | III | 25 | Feedback loop, prompt levels, **D2 (12 min)**, notebooks, parallel, context (15-22) |
| 0:55 | IV | 15 | Anatomy, constraints, templates, failure modes (23-28) |
| 1:10 | Break | 10 | (29) |
| 1:20 | V | 8 | CLAUDE.md, WT 2 (30-32) |
| 1:28 | VI | 15 | Map, skills + subagents **D3 (5 min)**, specialists, deny rules and hooks, MCP (33-39) |
| 1:43 | VII | 15 | Closing the loop, how /goal works, demo problem, **D4 (goal set at 1:45, shown at 1:57)**, the goal taken apart, turn by turn, guardrails (40-47) |
| 1:58 | VIII | 12 | The plan and its reconciliation, the silent failure, **D5 (optional)**, pipeline, replication (48-53) |
| 2:10 | IX | 15 | Verification, disclosure, referees/supervisors/students, replication packages, governance, six habits (54-60) |
| 2:25 | Wrap-up | 5 | Quick reference, takeaways, ten tips, taking it home (61-65) |

If running late: drop slides 20-21 (notebooks, worktrees), 53 (replication) and 64 (tips); never drop D2 or D4.

## Pre-lecture checklist

- [ ] `claude --version` and `claude doctor` on the presenting laptop; logged in; `/model` set (Opus 5.5 for the demos, or Sonnet 5.5 to keep cost down).
- [ ] `rm -rf ~/research/munich-demo && bash code/setup_demo_project.sh`; open a terminal in it, font size large enough for the back row. `jq`, `ruff` and `latexmk` installed (the hooks use them), or the corresponding hooks deleted from `.claude/settings.json`.
- [ ] Dry-run D2 (plan mode) once; note how long the plan takes to appear.
- [ ] `cp -r code/pinn_loop ~/research/pinn-loop`, `git init`, commit, `pytest -q` fails with `NotImplementedError`. Dry-run the `/goal` once in a scratch copy; write the measured turns, wall-clock and `max_err` into the note on slide 44 (and adjust the transcript on slide 46 if it differs materially). Then reset the demo copy to the stub. In the default permission mode the goal will ask before running `pytest`; allow it with `/permissions` or run the demo in auto mode.
- [ ] `python3 -c "import torch"` (the loop needs it).
- [ ] A second terminal tab ready for the goal; `/cost` noted at the start so the total spend of the lecture can be shown at the end.
- [ ] `slides/Agentic_Programming_Munich.pdf` open, presenter mode; the three figures live in `slides/figures/`.

## Files not stepped through, kept for reference

| File | Why it is here |
|---|---|
| `code/agents/code_reviewer.md`, `doc_generator.md`, `test_writer.md`, `monte_carlo_designer.md`, `loss_balancing_reviewer.md` | The rest of the subagent set installed by the setup script; mentioned on slides 36 and 50. |
| `code/ode_solver/solver.py`, `plot.py`, `model.pt`, `solution.png` | The agent's 1-D warm-up (`y'' + y = 0`, `cos x`, soft initial-condition penalty), a different problem from the loop's; its figure is no longer on a slide. |
| `code/pinn_loop_solution/pinn_hard_bc.py` | The reference solution of the loop demo, kept outside `code/pinn_loop/` so the agent cannot read it; `max_err = 6.70e-5`, passes the three tests. |
| `slides/figures/agentic_pinn_ode.png` | The 1-D warm-up figure, kept with the other figures. |

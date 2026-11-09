# Every prompt in the lecture, ready to paste

One section per prompt, in lecture order, with the slide it appears on. The
code blocks are **plain text**: no `>` marker, no indentation, so you can
select a block and paste it straight into Claude Code. Slash commands
(`/data-diagnostics ...`, `/goal ...`) are typed as shown.

All prompts assume the demo project built by `setup_demo_project.sh`
(`~/research/munich-demo`: the data in `data/`, `brock_mirman_reference.py` in
the project root, `CLAUDE.md` from `CLAUDE_md_demo.md`, `.claude/`). Slide numbers refer to
`slides/Agentic_Programming_Munich.pdf`.

| § | Slide | Demo / walk-through | Prompt |
|---|---|---|---|
| 1 | 14 | D1, WT 1 | Setup check: summarise the project without writing |
| 2 | 16 | | The feedback loop: two turns and a steer |
| 3 | 18 | | A Level-3 prompt: moments with bootstrap uncertainty |
| 4 | 19 | D2 | A research task in plan mode |
| 5 | 20 | | Run a notebook headless and compare with its stored outputs |
| 6 | 22 | | Context management: the progress file |
| 7 | 24–27 | | Prompt anatomy: before and after; the constraint lines; six templates |
| 8 | 32 | WT 2 | Orientation from `CLAUDE.md` alone |
| 9 | 35–37 | D3, WT 3–4 | A skill, the verifier subagent, a specialist |
| 10 | 39 | | MCP: Zotero and GitHub |
| 11 | 44–46 | D4, WT 6 | The `/goal` that builds the solver, and its template |
| 12 | 47 | | Headless: `claude -p`, a headless goal, the fresh-context shell loop |
| 13 | 51 | D5 | One prompt → table + figure (Euler errors) |
| 14 | 52 | | The full pipeline: six prompts, `verify.py` and `run_all.py` |
| 15 | 53 | | Replication |

D = live demo (five), WT = walk-through badge on the slides (seven); the same
crosswalk is at the top of `WALKTHROUGH.md`.

---

## 1. Setup check (slide 14, Walk-through 1)

```text
What files are in this directory? Summarise the structure and the docstring of the solver. Do not write anything yet.
```

A healthy answer lists the files, reads `CLAUDE.md` and the docstring of
`brock_mirman_reference.py`, notes that `data/bm_panel_truth.csv` is off limits,
and writes nothing.

## 2. The feedback loop (slide 16)

Turn 1:

```text
data/bm_panel.csv has Y, C, K per economy and period. Print Autocorr(log Y) per economy. Use pandas. No plot yet.
```

Turn 3, the steer, after reading the output:

```text
Did you demean within each economy before the autocorrelation, or pool the panel? Print the per-economy means of log Y first.
```

## 3. A Level-3 prompt (slide 18)

```text
Load data/bm_panel.csv (economy, t, Y, C, K, s; 5 economies x 500 periods).
- For each economy compute Std(dlog C), Autocorr(dlog C), Autocorr(log Y), mean investment share
- Use brock_mirman_reference.moments(Y, C, K); do NOT re-implement it
- Print a table, economies as rows, with the sd of each moment across 200 block-bootstrap resamples; write it to results/moments.json
- Do NOT read data/bm_panel_truth.csv
Verify: assert 500 rows per economy and 0 < s < 1 everywhere.
Expected output format: Markdown table printed to stdout.
```

## 4. Live demo 2: a research task in plan mode (slide 19)

Press Shift+Tab until the prompt shows `[plan]`, then paste:

```text
I have a simulated panel at data/bm_panel.csv: five economies, 500 periods, columns economy, t, Y, C, K, s. Each is a stochastic Brock-Mirman growth model with its own persistence of productivity, which I do not tell you. brock_mirman_reference.py in this folder solves and simulates it. Without loading anything yet, first tell me how you would inspect the data. Then:
1. Load it and print diagnostics (columns, shape, periods per economy, missing values)
2. Compute Std(dlog C), Autocorr(dlog C), Autocorr(log Y), mean s per economy
3. Generate a 2x2 panel of plots (log Y paths; dlog C histograms; Autocorr(log Y) vs Std(dlog C); s vs K for economy 4; matplotlib, no gridlines, serif font, 6x5 inches)
4. Save the figure to figures/panel_overview.pdf
5. Write 3 sentences to notes/data_description.md: which economy has the most persistent productivity, and how you can tell
Do NOT read data/bm_panel_truth.csv.
```

Read the plan before approving it: every file it will create, no mention of the
truth file, the autocorrelation computed within economy, a verification step
(500 rows per economy, 0 < s < 1) before the note is written.

Expected numbers (the agent must not know them; you do):

| economy | Std(dlog C) | Autocorr(dlog C) | Autocorr(log Y) | mean s | hidden rho |
|---|---|---|---|---|---|
| 1 | 0.0119 | 0.159 | 0.566 | 0.254 | 0.50 |
| 2 | 0.0146 | 0.300 | 0.806 | 0.253 | 0.70 |
| 3 | 0.0167 | 0.284 | 0.867 | 0.254 | 0.80 |
| 4 | 0.0208 | 0.313 | 0.963 | 0.254 | 0.90 |
| 5 | 0.0226 | 0.302 | 0.973 | 0.255 | 0.95 |

The note should name economy 5 as the most persistent one.

## 5. Notebooks (slide 20)

```text
Run estimation.ipynb headless with N_SIM = 200. Compare the point estimates and standard errors with the stored outputs in the notebook. Report differences larger than one standard error. Do not edit the notebook.
```

The headless command the agent should use:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output out.ipynb estimation.ipynb
```

## 6. Context management (slide 22)

At the end of a long session:

```text
Write notes/session_state.md: 1. What we have done so far. 2. What is working. 3. What we need to do next. 4. Design decisions and why.
```

At the start of the next one:

```text
Read notes/session_state.md and continue from where we left off.
```

## 7. Prompt anatomy and the templates (slides 24–27)

### Before and after (slide 24)

Level 0:

```text
Analyze the data and estimate the model and make a table.
```

Level 3, the same task:

```text
[ROLE] You are a computational macroeconomist.
[CONTEXT] data/bm_panel.csv holds five economies, 500 periods each, columns economy, t, Y, C, K, s. Each is the stochastic Brock-Mirman model (alpha 0.36, delta 0.10, sigma 0.04, beta 0.96) with its own persistence rho of log productivity. brock_mirman_reference.py in this folder solves and simulates it.
[TASK] Estimate rho for each economy by the simulated method of moments: match Std(dlog C), Autocorr(dlog C), Autocorr(log Y), weighted by the inverse of their bootstrap variance from results/moments.json; simulate 10 x 500 periods per candidate rho with common random numbers; a grid of 15 points on [0.3, 0.99], then minimize_scalar on the best bracket.
[CONSTRAINTS] numpy and scipy only. Do not modify brock_mirman_reference.py. Do not read data/bm_panel_truth.csv. Fix every seed.
[OUTPUT] results/rho_hat.json and paper/tables/tab_rho.tex (booktabs, the LaTeX table style journals expect): rho_hat, data moments, fitted moments, criterion value, one row per economy.
[VERIFY] After the table is written, and only then, read data/bm_panel_truth.csv and print |rho_hat - rho| per economy; below 0.05 (about two Monte Carlo standard deviations at T = 500) is a pass.
```

### The constraint lines (slide 25), to append to any prompt

```text
Do NOT modify brock_mirman_reference.py. Do NOT change the sample or the presets. Do NOT read the truth file before verifying. Do NOT hardcode file paths or seeds. Do NOT invent citations.
Always use type hints. Reproducible from one script: python run_all.py. Use the notation of the paper (K, z, s(z, K), rho). Cluster standard errors at the state level. Seeds set in config.py.
Only modify files in the code/ directory. Do not touch the notebooks or data/. Make the minimum change needed to fix the bug. Stop and report before the next step.
```

The "think before you code" sentence:

```text
BEFORE writing any code: (1) describe your plan, (2) list every file you will modify, (3) list the risks. Do not proceed until I say "go ahead".
```

### Template 1: replication check, code vs paper (slide 26)

```text
Read code/estimate.py and Section 3 of paper/main.tex. Check: does the estimator in the code match the one in the paper? Discrepancy report, one line per item: OK [item]: matches; X [item]: paper says X, code does Y; ? [item]: unclear from the paper.
```

### Template 2: robustness table (slide 26)

```text
Add 3 robustness checks to the baseline in code/estimate.py: (1) drop the post-2008 sample, (2) cluster at the region level, (3) add region-specific linear trends. For each: a separate script, results to results/robustness_{name}.json. Then one LaTeX table, all 4 specifications side by side (booktabs), coefficient and standard error.
```

### Template 3: autonomous session (slide 26)

```text
I am stepping away for 30 minutes. Task: implement the pipeline per plan.md. Rules while I am away: commit after each major step; if stuck after 3 attempts, write notes/blocked.md and STOP; do NOT modify the raw data or the reference solver; write progress every 10 steps to notes/progress.md. When done: summary to notes/session_end.md.
```

### Template 4: literature writing (slide 26)

```text
Do NOT invent any citations. Only cite papers already in paper/references.bib. If a claim needs a source that is not there, write [CITE: author, year -- verify] and list the gaps at the end.
```

### Template 5: translate legacy code, and prove it (slide 27)

```text
Port model/solve_rbc.m (MATLAB, 2014) to python/solve_rbc.py. NumPy/SciPy only. Do NOT change the algorithm, only the language. Then write tests/test_port.py: steady state agrees to within 1e-10; policy on the MATLAB grid within 1e-8 (exported to data/matlab_policy.csv); 500-period simulation with the shocks in data/shocks.csv, paths within 1e-6 (max abs). Run pytest. Report every deviation with the line in each file that causes it.
```

### Template 6: check a derivation with computer algebra (slide 27)

```text
Model: log utility, Y = z K^alpha, K' = (1-delta) K + s Y, log z AR(1). Derive the Euler equation and the deterministic steady state (K*, s*). Verify every step with sympy in notes/check_euler.py: substitute the steady state into the Euler equation and print the residual (must be 0). Cross-check K* against brock_mirman_reference.steady_state(0.96). If a residual is not zero, say so; do not adjust the derivation to fit.
```

Reference answer: K* = 4.294, s* = 0.254 at beta = 0.96.

## 8. Orientation from `CLAUDE.md` (slide 32)

In a fresh session (`/clear`) in the demo project:

```text
Summarise this project and what I should work on next.
```

The agent answers from `CLAUDE.md` alone (the HANK fiscal-multiplier project, the
referee comment, the blocked item). Then the correction loop:

```text
Update CLAUDE.md so you don't make that mistake again.
```

## 9. Live demo 3: skill, verifier, specialist (slides 35–37); the deny-rule check (slide 38)

```text
/data-diagnostics data/bm_panel.csv
```

```text
Use the verifier subagent to check the claims in the diagnostics report it just wrote under notes/.
```

A specialist, after the agent has saved a policy (for example from Template 6 or the pipeline):

```text
Use the solution-accuracy-checker subagent on results/policy.csv. Reference: brock_mirman_reference.solve(0.96, 0.9).
```

```text
Use the econometrics-reviewer subagent on the last commit.
```

The deny rule and the hook of slide 38, in the demo project (both refuse):

```text
Append a line to data/raw/test.txt.
```

## 10. MCP (slide 39)

After `claude mcp add ...` for a Zotero and a GitHub server:

```text
Search Zotero for papers on fiscal multipliers in HANK models; list the ones not yet in paper/references.bib.
```

```text
Open a GitHub issue for the pre-trend test with the plan from notes/plan.md.
```

## 11. Live demo 4: the `/goal` that builds the solver (slides 44–46)

In a *second* terminal, in a copy of `code/pinn_loop/` with its own git history:

```bash
cp -r code/pinn_loop ~/research/pinn-loop
cd ~/research/pinn-loop && git init -q && git add . && git commit -qm "seed: stub + acceptance tests"
pytest -q          # must fail: NotImplementedError
claude
```

Then paste as one block (including the `/goal`):

```text
/goal `pytest -q` in this folder exits 0 with 3 passed. To get there, implement pinn_hard_bc.py: u'' + u = 0 on [0, pi/2], u(0) = 0, u(pi/2) = 1, exact solution sin x. Use the trial solution u = 2x/pi + x (pi/2 - x) N(x) so the boundary conditions hold by construction and the loss is the residual alone; network 1-32-32-1 tanh, 256 fresh uniform collocation points per step, Adam 1e-3, 3000 steps, seed 42, float64, PyTorch. solve(seed) returns {"max_err": max |u - sin x| on 1001 points, "u": callable}. Each turn: run pytest -q and show the output, fix one thing, commit. Report ALL_TESTS_PASS when pytest exits 0. Stop after 6 turns (count your commits in git log) and write notes/blocked.md instead.
```

`/goal` keeps the session working turn after turn; after every turn a separate
small model reads the transcript and judges whether the condition holds, so the
condition must be something the agent's own output shows (hence "run pytest -q
and show the output"). In the default permission mode the agent asks before
running `pytest`; allow it with `/permissions` or use auto mode for an unattended run.

Afterwards: `pytest -q` prints `3 passed`; `git log --oneline` shows one commit
per turn; `/goal` shows the achieved condition with turns, time and token spend.
The threshold `5.54e-4` in the tests is produced by
`code/pinn_loop/reference/soft_bc_baseline.py`; the reference solution in
`code/pinn_loop_solution/` reaches `6.70e-5`. Dry-run the goal before the lecture
and note turns, wall-clock and `max_err`.

### The six parts (slide 45), as a template for your own goal

```text
/goal [CONDITION: `<verifier command>` exits 0.] [PROBLEM: To get there, implement <file>: <the mathematics or the estimator, the data, the domain>.] [RECIPE: <method, sample, seed, precision, libraries>.] [INTERFACE: <the function your tests import and what it returns>.] Each turn: run <verifier command> and show the output, fix one thing, commit. Report ALL_TESTS_PASS when <verifier command> exits 0. Stop after 6 turns (count your commits in git log) and write notes/blocked.md instead.
```

Keep the last three sentences word for word. The verifier must be a command that
exits 0 when, and only when, the task is done. An empirical example: a simulated
staggered-adoption panel with a known treatment effect of 0.10, and tests that
the estimator recovers it within 0.02 with standard errors clustered at the unit
level and that the same seed gives the same table twice. Run goals on a
throw-away branch, with the deny rules of §9's settings file installed, and
check `/goal` for the token spend.

To end a goal yourself:

```text
/goal clear
```

## 12. Headless, and the fresh-context shell loop (slide 47)

One prompt from a script, cron or a Slurm job:

```bash
claude -p "Re-run estimate.py on data/2026m11.csv, refresh tables/, summarise the changes vs last month in notes/changelog.md" --output-format json --allowedTools "Read,Edit,Bash(python *)" > logs/run_$(date +%F).json
```

A goal, headless, the whole loop in one invocation:

```bash
claude -p "/goal pytest -q exits 0 with 3 passed. To get there, ..." --output-format stream-json --verbose
```

The fresh-context version (the "Ralph" pattern): a new process per run, so each
run starts with an empty context window and reads everything from disk; your test
decides when to stop:

```bash
while ! pytest -q; do
  claude -p "$(cat PROMPT.md)" --allowedTools "Read,Edit,Bash"
done
```

## 13. Live demo 5: one prompt → table + figure (slide 51)

```text
Role: computational macroeconomist. Code: brock_mirman_reference.py (read the docstring). Task: for rho in {0.5, 0.7, 0.9, 0.99}, beta = 0.96, solve on the default grid and on a coarse one (n_k=12, n_z=5, tol=1e-5); simulate 2000 periods (seed 7) from the default solution; evaluate the relative Euler error of BOTH on those states. Outputs: 1) outputs/euler_table.tex -- booktabs, both grids, mean and max, mean s, solve time. 2) outputs/euler_figure.pdf -- s(z=1,K), K in [2,8]; |Euler| at z=1, log axis; serif, 8x3.4 inches. Constraints: do NOT modify the module. Verify: assert default mean |Euler| < 1e-6; print the ratio coarse/default.
```

`euler_check_demo.py` is a reference answer; `diff` its `outputs/euler_table.tex`
against what the agent produces (only the seconds column should differ).

## 14. The full pipeline, six prompts (slide 52)

One prompt per step; each step writes a file the next step reads. The truth file
is read once, at the end.

```text
Step 1. Read data/bm_panel.csv (do NOT read data/bm_panel_truth.csv). Write notes/report.md: columns, shape, periods per economy, missing values, summary statistics per economy.
```
(or simply `/data-diagnostics data/bm_panel.csv`)

```text
Step 2. Write code/moments.py: for each economy compute Std(dlog C), Autocorr(dlog C), Autocorr(log Y) with brock_mirman_reference.moments (do not re-implement it); save results/moments.json. Run it.
```

```text
Step 3. Write code/estimate_rho.py: for each economy estimate rho by the simulated method of moments. Simulate 10 x 500 periods per candidate rho with brock_mirman_reference.solve and .simulate, common random numbers across candidates, a grid of 15 points on [0.3, 0.99] then minimize_scalar on the best bracket. Weight the three moments by the inverse of their bootstrap variance from results/moments.json. Save results/rho_hat.json with rho_hat, the criterion on the grid, and the fitted moments. Fix every seed. Run it; it should take under 3 minutes.
```

```text
Step 4. Write code/plot_criterion.py: one panel per economy, the SMM criterion against rho on the grid, log axis, the minimum marked; serif font, 12 x 2.6 inches, 300 DPI; save figures/criterion.pdf.
```

```text
Step 5. Write code/make_table.py: paper/tables/tab_rho.tex (booktabs), one row per economy: rho_hat, the three data moments, the three fitted moments, the criterion value. Raw LaTeX, no document preamble.
```

```text
Step 6. The verification step has been reached: write verify.py in the project root (not under code/). It reads data/bm_panel_truth.csv, prints |rho_hat - rho| per economy, and passes if every value is below 0.05 (about two Monte Carlo standard deviations at T = 500). Run it. Then write notes/results.md: 300 words describing the estimates, every number traced to a file in results/.
```

```text
Write run_all.py that chains steps 1-5 and then verify.py, with a --dry-run flag that prints what each step would do. grep truth code/ must return nothing: only verify.py reads the truth file.
```

Expected: every |rho_hat - rho| below 0.05; economy 4 lands near 0.92 at T = 500.

## 15. Replication (slide 53)

```text
Role: replication engineer. Paper: [title, authors, DOI]. Mission: replicate ALL figures and tables. Hard rules: 1. Do NOT use the authors' code. 2. Track progress in notes/todo.md. 3. Commit after each step. 4. If stuck x3, write notes/blocked.md, STOP. Before implementing: extract figure axes, ranges, CIs; write the plan to notes/plan.md and wait for my approval. Implementation: Python + NumPy/SciPy. Figures to figures/replication/ (PDF). Final: a table "paper value vs our value"; < 5 % = match, 5-20 % = close, > 20 % = fail.
```

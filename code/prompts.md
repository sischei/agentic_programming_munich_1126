# Every prompt in the lecture, ready to paste

One section per prompt, in lecture order. The
code blocks are **plain text**: no `>` marker, no indentation, so you can
select a block and paste it straight into Claude Code. Slash commands
(`/deqn-accuracy ...`, `/goal ...`) are typed as shown.

All prompts assume the demo project built by `setup_demo_project.sh`: `demo/` at the
root of this repository (git-ignored, with its own git history), holding
`brock_mirman_reference.py`, the DEQN scaffold (`deqn/`, `tests/`,
`reference/`), `CLAUDE.md` from `CLAUDE_md_demo.md` and `.claude/`. Every session of
the lecture runs in `demo/`.

| § | Demo / walk-through | Prompt |
|---|---|---|
| 1 | D1, WT 1 | Setup check: summarise the project without writing |
| 2 | | The feedback loop: Step 1 without a plan, and the steer |
| 3 | | A Level-3 prompt: the port of Step 1 with its own check |
| 4 | D2 | Step 1 in plan mode: port, check, figure, note |
| 5 | | Run a notebook headless and compare with its stored outputs |
| 6 | | Context management: the state file |
| 7 | | Prompt anatomy: Step 2 as the six-component prompt; the constraint lines; six templates |
| 8 | WT 2 | Orientation from `CLAUDE.md` alone |
| 9 | D3, WT 3–5 | The accuracy skill, the verifier subagent, a specialist; the deny rule and the hook |
| 10 | | MCP: Zotero and GitHub |
| 11 | D4, WT 6 | The `/goal` that builds the stochastic solver, and its template |
| 12 | | Headless: `claude -p`, a headless goal, the fresh-context shell loop |
| 13 | D5 | One prompt → table + figure (four loss kernels) |
| 14 | | Replication |

D = live demo (five), WT = walk-through badge on the slides (seven); the same
crosswalk is at the top of `WALKTHROUGH.md`.

---

## 1. Setup check (Walk-through 1)

```text
What is in this directory? Summarise the structure, the two rules of CLAUDE.md and the docstring of deqn/common.py. Do not write anything yet.
```

A healthy answer lists the files, reads `CLAUDE.md`, names the two rules (`tests/`
and `reference/` are the exam; `data/raw/` is off limits), notes that `deqn/stochastic.py` is a
stub whose tests fail with `NotImplementedError`, and writes nothing.

## 2. The feedback loop

Step 1 done without a plan. Turn 1:

```text
Port deqn/source/deqn_tf_deterministic.py to PyTorch as deqn/deterministic.py. Same recipe. Print the training loss at the end.
```

Turn 3, the steer, after the agent reports "the loss has converged":

```text
A small loss is not a small error. The answer is known: s* = alpha*beta. Print max |s(K) - alpha*beta| on 1,001 points of [0.1, 1] and plot s(K) against the constant. Then tell me whether it is solved.
```

Our version of the result: `code/deqn/deterministic.py` (training loss 3.1e-7,
max |s - alpha beta| = 4.9e-4, 3 s).

## 3. A Level-3 prompt: the port of Step 1

```text
Role: computational macroeconomist. Port deqn/source/deqn_tf_deterministic.py (TensorFlow) to deqn/deterministic.py in PyTorch, same recipe: delta = 1, beta = 0.99, K ~ U[0.1, 1], 64 states per step, Adam 1e-3, 2000 steps; a 1-32-32-1 tanh network for the savings rate, built with deqn.common.make_net(1).
- float64; torch.manual_seed(42) before the network
- the loss is the mean squared relative Euler error, as in the original; do NOT put the closed form in the loss
- save results/deqn_deterministic.pt via common.save_policy
Verify: max |s(K) - alpha*beta| on 1,001 points of [0.1, 1] must be < 1e-2; print it, and the Euler residual there.
```

Expected: about 3 s of training; max |s(K) - alpha beta| around 5e-4 (the threshold
is 1e-2); `results/deqn_deterministic.pt` and its `.json` sidecar written.

## 4. Live demo 2: Step 1 in plan mode

Press Shift+Tab until the status line shows plan mode (sessions start in auto mode, so it takes three presses), then paste:

```text
Step 1 of the DEQN build. deqn/source/deqn_tf_deterministic.py is a TensorFlow notebook flattened to a script: a neural network for the savings rate of the deterministic growth model with delta = 1, trained on the Euler residual. The answer is known: s* = alpha*beta. Without writing anything yet, tell me how you would port it. Then:
1. Port it to deqn/deterministic.py: PyTorch, float64, seed 42, same recipe (K ~ U[0.1, 1], 64 states per step, Adam 1e-3, 2000 steps); deqn.common.make_net(1) for the network and common.save_policy for results/deqn_deterministic.pt
2. Write scripts/check_step1.py: max |s(K) - alpha*beta| and the Euler residual on 1,001 points of [0.1, 1]; print both
3. Save figures/deqn_step1.pdf: s(K) against the constant alpha*beta, and |residual| on a log axis (matplotlib, serif font, no gridlines, 6 x 2.4 inches)
4. Write 3 sentences to notes/deqn_step1.md: is it solved, and how you can tell
Do NOT use the closed form anywhere except in the check. Do NOT modify deqn/common.py.
```

Read the plan before approving it: every file it will create; the closed form only in
the check, not in the loss; `make_net` and `save_policy` reused, not re-implemented;
torch seeded before the network is built; "solved" judged by the max error, not by
the loss.

Expected (our version is `code/deqn/deterministic.py`, figure
`code/deqn/outputs/deqn_step1_figure.pdf`):

| quantity | value |
|---|---|
| training time | about 3 s |
| training loss, last 100 steps | 3.1e-7 |
| max \|s(K) - alpha beta\| on [0.1, 1] | 4.9e-4 (threshold 1e-2) |
| Euler residual on the grid, mean / max | 4.7e-4 / 1.9e-3 |

Afterwards: `/deqn-accuracy results/deqn_deterministic.pt` (Part VI) grades the same
policy in five lines.

## 5. Notebooks

```text
Run estimation.ipynb headless with N_SIM = 200. Compare the point estimates and standard errors with the stored outputs in the notebook. Report differences larger than one standard error. Do not edit the notebook.
```

The headless command the agent should use:

```bash
jupyter nbconvert --to notebook --execute --ExecutePreprocessor.timeout=-1 --output out.ipynb estimation.ipynb
```

## 6. Context management

At the end of a long session:

```text
Write notes/session_state.md: 1. What we have done so far. 2. What is working. 3. What we need to do next. 4. Design decisions and why.
```

At the start of the next one:

```text
Read notes/session_state.md and continue from where we left off.
```

## 7. Prompt anatomy and the templates

### Before and after

Level 0:

```text
Add productivity shocks to the solver.
```

Level 3, the same task (Step 2: the stochastic solver; the same text is the problem
and recipe of the goal in §11):

```text
[ROLE] You are a computational macroeconomist.
[CONTEXT] deqn/deterministic.py (Step 1) is verified against the closed form. Now the stochastic model of CLAUDE.md: Y = z K^alpha, log z AR(1); alpha 0.36, delta 0.10, sigma 0.04, beta 0.96, rho 0.9. No closed form; the reference is brock_mirman_reference.solve(0.96, 0.9). deqn/common.py has the parameters, gh_nodes, make_net, encode_state, save_policy and simulate_path; read its docstring first.
[TASK] Write deqn/stochastic.py with solve(seed=42, steps=2000): state (log z / sd_z, log K - log K*); network 2-32-32-1 tanh, sigmoid output, s_init = s*; loss = mean squared Euler residual beta C E[(1 - delta + R') / C'] - 1 with the 5 Gauss-Hermite nodes; 128 fresh states per step, log z ~ N(0, sd_z), log K ~ U(log K* +- 0.5); Adam 1e-3.
[CONSTRAINTS] torch, float64, torch.manual_seed(seed) first. Never use the reference policy in a loss or an initialisation. Do not modify deqn/common.py, tests/ or reference/.
[OUTPUT] results/deqn_stochastic.pt with its JSON sidecar; solve returns {"s": callable (log_z, log_K) -> s, numpy in and out, "net", "wall"}.
[VERIFY] Simulate 2,000 periods with the network's own policy (common.simulate_path). Report mean |residual| there, max |s - s_ref| at those states, E[s] against the reference, and whether log K stayed inside [log K* - 0.5, log K* + 0.5]. Then run python3 -m pytest -q tests/ and show the output.
```

Expected: 5 passed in about 14 s; on-path mean |residual| about 1.4e-4; max |s - s_ref|
about 2.3e-3; E[s] 0.252; log K in [1.08, 1.81]. Our version: `code/deqn/solution/deqn_stochastic.py`.

### The constraint lines, to append to any prompt

```text
Do NOT use reference/ in a loss, an initialisation or a fit. Do NOT edit tests/ or reference/. Do NOT change the seed or the step budget. Do NOT read the truth file before verifying. Do NOT report the training loss as accuracy.
Use deqn/common.py: make_net, encode_state, save_policy, simulate_path. float64; torch.manual_seed before the network. Save every policy with its JSON sidecar. Use the notation of CLAUDE.md (K, z, s, rho, K*, s*).
Only write deqn/stochastic.py and results/. Do not touch deqn/common.py, deqn/source/ or data/. Make the minimum change that fixes the failing test. Stop and report after each pytest run.
```

For an empirical project the same three columns read, for example: "Do NOT change the
sample or the presets. Do NOT invent citations. / Cluster standard errors at the state
level. Seeds set in config.py. / Only modify files in code/; do not touch the
notebooks or data/."


The "think before you code" sentence:

```text
BEFORE writing any code: (1) describe your plan, (2) list every file you will modify, (3) list the risks. Do not proceed until I say "go ahead".
```

### Template 1: replication check, code vs paper

```text
Read code/estimate.py and Section 3 of paper/main.tex. Check: does the estimator in the code match the one in the paper? Discrepancy report, one line per item: OK [item]: matches; X [item]: paper says X, code does Y; ? [item]: unclear from the paper.
```

### Template 2: robustness table

```text
Add 3 robustness checks to the baseline in code/estimate.py: (1) drop the post-2008 sample, (2) cluster at the region level, (3) add region-specific linear trends. For each: a separate script, results to results/robustness_{name}.json. Then one LaTeX table, all 4 specifications side by side (booktabs), coefficient and standard error.
```

### Template 3: autonomous session

```text
I am stepping away for 30 minutes. Task: implement the pipeline per plan.md. Rules while I am away: commit after each major step; if stuck after 3 attempts, write notes/blocked.md and STOP; do NOT modify the raw data or the reference solver; write progress every 10 steps to notes/progress.md. When done: summary to notes/session_end.md.
```

### Template 4: literature writing

```text
Do NOT invent any citations. Only cite papers already in paper/references.bib. If a claim needs a source that is not there, write [CITE: author, year -- verify] and list the gaps at the end.
```

### Template 5: translate legacy code, and prove it

```text
Port deqn/source/deqn_tf_stochastic.py (TensorFlow) to deqn/stochastic_port.py. Do NOT change the algorithm, only the language. Then write deqn/test_port.py: the residual has shape (n, 1) for n states; the Gauss-Hermite weights sum to 1; at sigma = 0, delta = 1 and beta = 0.99 the residual equals the one of deqn/deterministic.py at the same states (within 1e-12). Run pytest on it. Report every deviation from the original with the line that causes it.
```

The original has `expectation = tf.zeros((n_data, dim_state))`, shape (n, 2) instead
of (n, 1): the residual silently becomes two identical columns and the mean squared
error is unchanged. No training run ever revealed it; the first test does. The same
template ports MATLAB, Stata, Dynare, Julia or Fortran code: export the old numbers,
then demand equality.

### Template 6: check a derivation with computer algebra

```text
Model: log utility, Y = z K^alpha, K' = (1-delta) K + s Y, log z AR(1). Derive the Euler equation and the deterministic steady state (K*, s*). Verify every step with sympy in notes/check_euler.py: substitute the steady state into the Euler equation and print the residual (must be 0). Cross-check K* against brock_mirman_reference.steady_state(0.96). If a residual is not zero, say so; do not adjust the derivation to fit.
```

Reference answer: K* = 4.294, s* = 0.254 at beta = 0.96. That Euler equation is the DEQN's residual.

## 8. Orientation from `CLAUDE.md`

In a fresh session (`/clear`) in the demo project:

```text
Summarise this project and what I should work on next.
```

The agent answers from `CLAUDE.md` alone: the growth-model sandbox, the two rules
(`tests/` and `reference/` are the exam; `data/raw/` is off limits), the DEQN notation, the stub
in `deqn/stochastic.py` whose tests fail, and the status line. Then the correction loop:

```text
Update CLAUDE.md so you don't make that mistake again.
```

## 9. Live demo 3: the accuracy skill, the verifier, a specialist; the deny rule and the hook

The skill of Step 3, on the policy that Step 1 saved (5 s):

```text
/deqn-accuracy results/deqn_deterministic.pt
```

```text
Use the verifier subagent to check the claims in the accuracy report it just wrote under notes/.
```

A specialist, once the stochastic policy exists (after §11, or after copying
`code/deqn/solution/deqn_stochastic.py` over `deqn/stochastic.py` and running
`python3 -c "from deqn.stochastic import solve; solve()"`):

```text
Use the solution-accuracy-checker subagent on the accuracy report under notes/ and on results/deqn_stochastic.json (the sidecar of the policy). Reference: brock_mirman_reference.solve(0.96, 0.9).
```

```text
Use the econometrics-reviewer subagent on the last commit.
```

The hook and the deny rule of Walk-through 5, in the demo project (both refuse):

```text
Relax the threshold in tests/test_deqn_stochastic.py to 0.1.
```

```text
Show me reference/baseline.md.
```

## 10. MCP

After `claude mcp add ...` for a Zotero and a GitHub server:

```text
Search Zotero for papers on fiscal multipliers in HANK models; list the ones not yet in paper/references.bib.
```

```text
Open a GitHub issue for the pre-trend test with the plan from notes/plan.md.
```

## 11. Live demo 4: the `/goal` that builds the solver

In a *second* terminal, in the demo project (it has its own git history from the
setup script):

```bash
cd demo
python3 -m pytest -q tests/     # must fail: NotImplementedError
claude
```

Then paste as one block (including the `/goal`):

```text
/goal `python3 -m pytest -q tests/` in this project exits 0 with 5 passed. To get there, implement solve(seed, steps) in deqn/stochastic.py: a DEQN for the stochastic growth model of CLAUDE.md. State (log z / sd_z, log K - log K*) via common.encode_state; network common.make_net(2, s_init=s*); loss = mean squared Euler residual beta C E[(1 - delta + R') / C'] - 1 with common.gh_nodes(5); 128 fresh states per step, log z ~ N(0, sd_z), log K ~ U(log K* +- 0.5); Adam 1e-3, 2000 steps, torch.manual_seed(seed) first, float64. solve returns {"s": callable (log_z, log_K) -> savings rate, numpy in and out, "net", "wall"} and saves results/deqn_stochastic.pt via common.save_policy. Each turn: run the tests and show the output, fix one thing, commit. Never edit tests/ or reference/. Report ALL_TESTS_PASS when pytest exits 0. Stop after 6 turns (count your commits in git log) and write notes/blocked.md instead.
```

`/goal` keeps the session working turn after turn; after every turn a separate
small model reads the transcript and judges whether the condition holds, so the
condition must be something the agent's own output shows (hence "run the tests
and show the output"). In the default permission mode the agent asks before
running `pytest`; allow it with `/permissions` or use auto mode for an unattended run.

Afterwards: `python3 -m pytest -q tests/` prints `5 passed`; `git log --oneline`
shows one commit per turn; `/goal` shows the achieved condition with turns, time
and token spend; `/deqn-accuracy results/deqn_stochastic.pt` grades the result.
The thresholds in the tests are 4–25x what the reference solution reaches
(`reference/baseline.md`); the reference solution in `code/deqn/solution/` passes
all five in 14 s. Dry-run the goal before the lecture and note turns, wall-clock
and E[s]. To reset the demo project to the stub afterwards:
`git checkout deqn/stochastic.py && rm -f results/deqn_stochastic.*`.

### The six parts, as a template for your own goal

```text
/goal [CONDITION: `<verifier command>` exits 0.] [PROBLEM: To get there, implement <file>: <the mathematics or the estimator, the data, the domain>.] [RECIPE: <method, sample, seed, precision, libraries>.] [INTERFACE: <the function your tests import and what it returns>.] Each turn: run <verifier command> and show the output, fix one thing, commit. Report ALL_TESTS_PASS when <verifier command> exits 0. Stop after 6 turns (count your commits in git log) and write notes/blocked.md instead.
```

Keep the last three sentences word for word; add "Never edit <the test folder>" when
the tests live in the project. The verifier must be a command that exits 0 when, and
only when, the task is done. An empirical example: a simulated
staggered-adoption panel with a known treatment effect of 0.10, and tests that
the estimator recovers it within 0.02 with standard errors clustered at the unit
level and that the same seed gives the same table twice. Run goals on a
throw-away branch, with the deny rules of §9's settings file installed, and
check `/goal` for the token spend.

To end a goal yourself:

```text
/goal clear
```

## 12. Headless, and the fresh-context shell loop

One prompt from a script, cron or a Slurm job:

```bash
claude -p "Re-run estimate.py on data/2026m11.csv, refresh tables/, summarise the changes vs last month in notes/changelog.md" --output-format json --allowedTools "Read,Edit,Bash(python *)" > logs/run_$(date +%F).json
```

A goal, headless, the whole loop in one invocation:

```bash
claude -p "/goal python3 -m pytest -q tests/ exits 0 with 5 passed. To get there, ..." --output-format stream-json --verbose
```

The fresh-context version (the "Ralph" pattern): a new process per run, so each
run starts with an empty context window and reads everything from disk; your test
decides when to stop:

```bash
while ! python3 -m pytest -q tests/; do
  claude -p "$(cat PROMPT.md)" --allowedTools "Read,Edit,Bash"
done
```

## 13. Live demo 5: one prompt → table + figure

Step 7, the loss kernel. Needs the verified solver of §11 (or the copied solution).

```text
Role: computational macroeconomist. Code: deqn/stochastic.py (the verified solver), deqn/common.py. Task: train the solver four times with the loss kernel changed and nothing else: MSE, MAE, Huber (delta 1e-2), log-cosh on the Euler residual; seed 42, 2000 steps each. Evaluate each on the path its own policy generates (common.simulate_path): |residual| mean, p99, max; E[s]; max |s - s_ref| from brock_mirman_reference.solve(0.96, 0.9). Outputs: 1) outputs/deqn_kernel_table.tex -- booktabs: kernel, training loss, the three residual columns, E[s], max |s - s_ref|, seconds. 2) outputs/deqn_kernel_figure.pdf -- s(z = 1, K) per kernel vs the reference; histogram of on-path |residual|, log axis; serif. Constraints: do NOT modify deqn/common.py or tests/. Verify: assert every E[s] is within 5e-3 of the reference value; print which kernel has the smallest mean |residual|.
```

`code/deqn/loss_kernels.py` is a reference answer (45 s); its committed table is
`code/deqn/outputs/deqn_kernel_table.tex`. Expected: all four E[s] within 3e-4 of
each other, on-path mean |residual| between 1.0e-4 and 1.7e-4; the training-loss
column (3e-4 for MAE, 2e-7 for Huber) is in different units per row and says nothing
about which is more accurate.

## 14. Replication

The paper-replication slide follows Hans and Bilionis (2026, arXiv:2607.02134), who replicate papers from
the LaTeX source alone with a coding-agent skill (Apache 2.0):
https://github.com/PredictiveScienceLab/paper-replication-paper. Two routes.

**With the skill** (hours per paper; the paper's LaTeX source in `paper_src/`):

```bash
git clone https://github.com/PredictiveScienceLab/paper-replication-paper
cp -r paper-replication-paper/skill/claude-code/paper-replication .claude/skills/
python3 .claude/skills/paper-replication/scripts/paper_replication.py bootstrap --project-dir rep/ --paper-title "<title>" --paper-source paper_src/
```

```text
Use the paper-replication skill on rep/ and keep going until validate-completion passes. No author code. Keep exactly one target active at a time; record every run and its provenance; mark a target MATCHED only when its artifact, provenance, comparison and report embedding exist. If a target cannot be reproduced from the paper materials, leave it unmatched and record why.
```

Read `rep/spec/reproduction_matrix.csv` (one row per target: claim, acceptance rule,
status, report slot) and run `python3 .claude/skills/paper-replication/scripts/paper_replication.py validate-completion --project-dir rep/`
yourself: completion is a workspace state, not the agent's last message.

**Without the skill**, for a small paper:

```text
Role: replication engineer. Paper: [title, authors, DOI; LaTeX source in paper_src/]. Mission: replicate every figure and table. Hard rules: 1. Do NOT use the authors' code. 2. Before implementing, list every target (figure, table, reported number) with the acceptance rule you will apply, in notes/targets.md; wait for my approval. 3. One target per commit; record the command, seed and output file of every run in notes/runs.md. 4. If stuck x3 on a target, mark it unmatched in notes/targets.md with the reason and move on. Final: a table "paper value vs our value" with the rule applied per target; unmatched targets listed, never dropped.
```

Expected: a target table before any code; per-target verdicts; some targets unmatched
with a recorded reason. The paper's twelve runs took one to thirteen hours each.

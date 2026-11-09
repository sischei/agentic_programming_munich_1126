# Agentic Programming for Economic Research

A 150-minute lecture at the **Center for Economic Studies (CES), LMU Munich**,
**November 2026**, for research economists who want to know what AI coding agents
such as Claude Code can and cannot be trusted with in a research workflow.

Slides: [`slides/Agentic_Programming_Munich.pdf`](slides/Agentic_Programming_Munich.pdf)
(65 frames, source in the same folder). Everything shown in the lecture is in
[`code/`](code), and [`code/WALKTHROUGH.md`](code/WALKTHROUGH.md) lists every
file in lecture order with its slide, what to look at and the command that
reproduces it.

---

## What the lecture is about

* An **AI coding agent** reads your project files, proposes and edits code or
  text, runs commands, inspects the output, and iterates towards a goal instead
  of answering once. Treated like a fast junior research assistant who never
  tires and never asks whether a result makes economic sense, it changes what
  the bottleneck in computational research is: no longer *can you write the
  code*, but *can you specify what you want and verify what you get*.
* The lecture builds the workflow around that fact. Block 1 is about working
  with the agent: the context window as the resource everything is managed
  around, the explore → plan → implement → commit discipline, and a prompt
  anatomy (task, context, constraints, output, verification) that turns "is this
  right?" into a number that is zero when it is. Block 2 is about extending and
  trusting it: project memory in `CLAUDE.md`, skills, subagents, permission
  rules and hooks as plain files in your repository, a `/goal` that keeps the
  agent working until a machine-checkable condition holds, and the verification,
  reproducibility, disclosure and data-governance rules under which the agent
  writes the code and you remain the author, whether you submit, referee or
  supervise.
* One running example for every demo on data: a stochastic growth model with a
  reference solver, five simulated economies with hidden persistence, and a truth
  file the agent is told not to read until verification. Every number the agent
  produces can be checked against a known answer, which is the pattern the
  lecture recommends for your own data. The autonomous-loop demo uses a
  ten-second differential-equation toy for the same reason: a known answer. No
  neural-network or dynamic-programming background is assumed.
* Python is the demo language because the sandbox is in Python; the prompt
  patterns are language-agnostic, and the agent writes and runs Stata, R and
  Julia files the same way.
* The tool is Claude Code (Anthropic's terminal agent, version 2.x as of
  November 2026). The ideas transfer to Codex CLI, Gemini CLI or Cursor; the
  file formats differ.

## Format

* **2 × 70 minutes with a 10-minute break.** Five live demos run on the
  lecturer's laptop; participants watch. There are no in-class exercises and no
  homework: instead, the relevant files are opened and stepped through on the
  slides, each with its location in this repository, so that everything can be
  replayed afterwards.
* **Nothing to install to attend.** To replay the demos at home you need Python,
  a Claude account (a paid plan or an API key) and Claude Code; see below.

## Schedule

| Clock | Part | Topic | Slides | Files stepped through |
|---|---|---|---|---|
| 0:00 | Intro | Why now; what "agentic" means; today's plan | 1–3 | |
| 0:05 | I | Orientation: chatbox vs agent, the context window, where the agent excels and struggles, the workflow, the running example | 4–9 | `code/data/`, `code/brock_mirman_reference.py` |
| 0:20 | II | Setting up: Claude Code, commands and models, git, the demo project (**demo 1**) | 10–14 | `code/setup_demo_project.sh`, `code/generate_synthetic_data.py` |
| 0:30 | III | The core loop: feedback and interruption, prompt quality levels, a research task in plan mode (**demo 2**), notebooks, parallel sessions, context management | 15–22 | `code/prompts.md` §2–6 |
| 0:55 | IV | Prompt engineering for research: six components, the constraint vocabulary, six templates, failure modes | 23–28 | `code/prompts.md` §7 |
| 1:10 | | *Break* | 29 | |
| 1:20 | V | Project memory: `CLAUDE.md` | 30–32 | `code/CLAUDE_md_template.md`, `code/prompts.md` §8 |
| 1:28 | VI | Skills, subagents, deny rules and hooks, MCP (**demo 3**: a skill and a verifier subagent) | 33–39 | `code/skills/`, `code/agents/`, `code/hooks/settings.json`, `code/prompts.md` §9–10 |
| 1:43 | VII | Autonomous loops: the pattern and three ways to run it, how `/goal` works, `/goal` builds a solver against three tests (**demo 4**), the goal taken apart, the run turn by turn, guardrails, headless use | 40–47 | `code/pinn_loop/`, `code/prompts.md` §11–12 |
| 1:58 | VIII | From plan to table and figure: the plan the agent wrote and how it was reconciled, the silent failure; one prompt → table + figure (**demo 5**); the full pipeline; replication | 48–53 | `code/ode_solver/`, `code/euler_check_demo.py`, `code/outputs/`, `code/prompts.md` §13–15 |
| 2:10 | IX | Verification, disclosure norms, what they mean for referees, supervisors and students, replication packages, GDPR / EU AI Act / data agreements, six habits | 54–60 | |
| 2:25 | Wrap-up | Quick reference, takeaways, ten tips, taking it home | 61–65 | `code/WALKTHROUGH.md` |

## The five live demos

| # | Slide | What happens | Prompt / command |
|---|---|---|---|
| 1 | 11, 14 | The stack check, then one script builds the demo project and the agent summarises it without writing anything | `claude doctor`; `bash code/setup_demo_project.sh`; `prompts.md` §1 |
| 2 | 19 | A Level-3 research task on the simulated panel in **plan mode**: the plan is read and approved before anything is written; diagnostics, moments, a figure and a note come out in one pass | `prompts.md` §4 |
| 3 | 35–36 | The `/data-diagnostics` skill on the panel, then the read-only `verifier` subagent checks the report's claims | `prompts.md` §9 |
| 4 | 44–46 | `/goal` keeps the agent working until three `pytest` tests pass on a small differential-equation solver; set in a second terminal while the goal is taken apart and the run is followed turn by turn on the slides; result shown at the end of Part VII | `prompts.md` §11 |
| 5 | 51 | One prompt → a `booktabs` table and a PDF figure of Euler-equation errors, with a verification assertion; the committed reference run is in `code/outputs/` | `prompts.md` §13; `python3 code/euler_check_demo.py` |

## Replaying the lecture at home

Python 3.10–3.12; everything runs on a laptop CPU. The example settings file
needs the command-line tools `jq`, `ruff` and `latexmk` for its hooks (or delete
the hooks you do not want). On Windows run the `bash` lines in Git Bash or WSL.

```bash
git clone https://github.com/sischei/agentic_programming_munich_1126.git
cd agentic_programming_munich_1126

python3 -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt ruff # numpy, scipy, pandas, matplotlib, torch, pytest

# Claude Code (needs a Claude account with a paid plan, or an API key)
curl -fsSL https://claude.ai/install.sh | bash     # macOS: brew install --cask claude-code
claude --version && claude doctor                  # /login opens the browser

# The demo project of Walk-through 1 (data, solver, CLAUDE.md, skills, agents, hooks)
bash code/setup_demo_project.sh ~/research/munich-demo
cd ~/research/munich-demo && claude
```

Then follow [`code/WALKTHROUGH.md`](code/WALKTHROUGH.md) in order and paste the
prompts from [`code/prompts.md`](code/prompts.md): plain-text blocks, one per
prompt, with the slide each belongs to. Two checks that need no agent:

```bash
python3 code/generate_synthetic_data.py   # regenerates code/data/ (10 s)
python3 code/euler_check_demo.py          # regenerates code/outputs/ (10 s)
```

Cost: a Pro subscription is enough for every demo one at a time. On the API the
demos together spend well under a dollar on Sonnet 5.5 and a few dollars on
Opus 5.5; the `/goal` demo is the most expensive single item. Retention and
training-use defaults differ by plan (check the privacy settings on Pro and Max;
API, Team and Enterprise data are not used for training; verify the current
terms before using any non-public data).

## Repository layout

```
README.md                      this page
requirements.txt
LICENCE                        CC0
slides/
  Agentic_Programming_Munich.tex / .pdf   the deck (65 frames); make / latexmk -pdf
  figures/                     the figures the deck includes
code/
  WALKTHROUGH.md               every file shown, in lecture order, with slide numbers and commands
  prompts.md                   every prompt of the lecture, plain text, ready to paste (15 sections, with slide numbers)
  setup_demo_project.sh        builds ~/research/munich-demo in one command
  brock_mirman_reference.py    the growth-model reference solver (the known answer)
  generate_synthetic_data.py   writes data/: five economies with hidden persistence
  euler_check_demo.py          one prompt -> table + figure, reference answer; writes outputs/
  data/                        bm_panel.csv (observed), bm_panel_truth.csv (verification only)
  outputs/                     euler_table.tex, euler_figure.pdf (committed reference run)
  CLAUDE_md_template.md        a filled-in CLAUDE.md for a (fictional) paper under revision (Part V)
  CLAUDE_md_demo.md            the demo project's own CLAUDE.md: structure, notation, the truth-file rule
  skills/                      data-diagnostics/SKILL.md, strategic-revision/SKILL.md (Part VI)
  agents/                      eight subagents: verifier, code reviewer, econometrics reviewer,
                               solution-accuracy checker, loss-balancing reviewer, test writer,
                               doc generator, Monte Carlo designer (Part VI)
  hooks/settings.json          permission deny rules (no reading data/raw/), PreToolUse edit guard,
                               audit log, LaTeX/ruff on edit (Part VI)
  ode_solver/                  the agent-built PINN: plan_pde.md, session.md, solver.py,
                               solver_pde.py, plot*.py, saved weights and figures (Part VIII)
  pinn_loop/                   the /goal demo: stub, three acceptance tests, the soft-penalty
                               baseline that sets the threshold, README (Part VII)
  pinn_loop_solution/          the reference solution, kept outside the demo folder
```

## Readings

* Korinek, A. (2023). Generative AI for economic research: use cases and implications for economists. *Journal of Economic Literature* 61(4), 1281–1317.
* Schluntz, E. and B. Zhang (2024). Building effective agents. Anthropic engineering blog.
* Goldsmith-Pinkham, P. (2026). Getting started with Claude Code. *paulgp.substack.com*.
* Bilionis, I. et al. (2026). We asked an AI agent to replicate three papers. *ebilionis.substack.com*.
* Cherny, B. (2025). 10 Claude Code tips from the creator of Claude Code.
* Brynjolfsson, E., D. Li and L. Raymond (2025). Generative AI at work. *Quarterly Journal of Economics*.
* METR (2025). Measuring the impact of early-2025 AI on experienced open-source developer productivity.
* Liu, N. F. et al. (2024). Lost in the middle: how language models use long contexts. *TACL* 12, 157–173.
* Walters, W. H. and E. I. Wilder (2023). Fabrication and errors in the bibliographic citations generated by ChatGPT. *Scientific Reports* 13, 14045.
* Anthropic (2026). Claude Code documentation: best practices, context window, hooks, subagents, scheduled tasks. https://code.claude.com/docs

## Lineage

The material grew out of a workshop day in Geneva (May 2026) and Session 9 of the
course *Deep Learning for Solving and Estimating Dynamic Economic Models* at
RWTH Aachen (October 2026). This version is standalone, written for a general
research-economics audience, twice the length, and was reviewed from the
perspectives of a computational economist, a PhD student and an applied
economist before release.

## Lecturer

**[Simon Scheidegger](https://sischei.github.io/)**
[HEC, University of Lausanne](https://www.unil.ch/hec/en/home.html) · Grantham Research Institute,
London School of Economics

## License

See [LICENCE](LICENCE) (CC0 1.0).

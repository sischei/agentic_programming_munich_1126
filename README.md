# Agentic Programming for Economic Research

![Agentic Programming for Economic Research, CES Munich, 18 November 2026](fig/theme.png)

> **AI coding agents can now take on much of the computational work of economic research. This lecture shows how to put them to work, and how to keep every result verifiable.**

**A 150-minute lecture at the [Center for Economic Studies (CES)](https://www.en.ces.econ.uni-muenchen.de/), LMU Munich, 18 November 2026.**
One example runs through the whole lecture: a neural solver for a growth model, built step by step *with* the agent
and graded at every step against a known answer. You leave with a working solver, a prompt discipline that transfers
to your own projects, and a repository that replays everything at home.

## Contents

1. [Install](#1-install)
2. [Replay the lecture](#2-replay-the-lecture)
3. [What is in this repository](#3-what-is-in-this-repository)
4. [The lecture at a glance](#4-the-lecture-at-a-glance)
5. [The lead example](#5-the-lead-example)
6. [Readings](#6-readings)
7. [Author](#7-author)

## 1. Install

Needs Python 3.10–3.12 and a laptop CPU; to run the agent, a Claude account (paid plan or API key).

```bash
git clone https://github.com/sischei/agentic_programming_munich_1126.git
cd agentic_programming_munich_1126
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt        # numpy, scipy, matplotlib, torch, pytest
```

Claude Code itself: `curl -fsSL https://claude.ai/install.sh | bash` (macOS: `brew install --cask claude-code`),
then `claude --version` and `claude doctor`. The optional hooks in `code/hooks/settings.json` also use
`jq`, `ruff` and `latexmk`.

## 2. Replay the lecture

```bash
bash code/setup_demo_project.sh        # builds demo/: the project every live session runs in
cd demo && claude                      # then paste the prompts from code/prompts.md
```

[`code/WALKTHROUGH.md`](code/WALKTHROUGH.md) lists every file in lecture order with the command that reproduces it;
[`code/prompts.md`](code/prompts.md) has every prompt, ready to paste. Without an agent, from the repository root:

```bash
python3 code/deqn/deterministic.py                    # Step 1: the port, 4.9e-4 from the closed form (3 s)
python3 code/deqn/solution/deqn_stochastic.py         # Step 4: our solver and its on-path numbers (11 s)
python3 code/deqn/stochastic_box.py                   # the silent failure: lower loss, wrong policy (11 s)
python3 code/deqn/accuracy.py code/deqn/outputs/deqn_stochastic.pt --plot check.pdf   # verdict and figure
```

Cost: a Pro subscription covers every demo; on the API, a few dollars on Opus, about half that on Sonnet.

## 3. What is in this repository

| Path | What |
|---|---|
| [`slides/Agentic_Programming_Munich.pdf`](slides/Agentic_Programming_Munich.pdf) | the deck, 77 pages, the last one an appendix (LaTeX source and figures next to it) |
| [`code/WALKTHROUGH.md`](code/WALKTHROUGH.md), [`code/prompts.md`](code/prompts.md) | every file and every prompt, in lecture order |
| [`code/deqn/`](code/deqn/README.md) | the lead example: scripts, the five tests, the reference, the committed numbers and figures |
| `code/brock_mirman_reference.py` | the reference solver: the known answer |
| `code/skills/`, `code/agents/`, `code/hooks/`, `code/CLAUDE_md_*.md` | the skills, eight subagents, the settings file with deny rules and hooks, two `CLAUDE.md` files |
| `code/setup_demo_project.sh` | builds `demo/` (git-ignored) from the pieces above |
| `code/extras/`, `code/euler_check_demo.py` | material from earlier versions, not on the slides |

## 4. The lecture at a glance

| Part | Topic |
|---|---|
| Intro | Why now; what "agentic" means; today's plan |
| I | Mental models: tiers, the context window, the workflow; the running example and its seven steps |
| II | Setting up: Claude Code, models and effort, open-weight models, git, the demo project (demo 1) |
| III | The core loop on Step 1; the same task in plan mode (demo 2); notebooks, parallel sessions, context |
| IV | Prompt engineering: the six-component prompt, constraints, templates, failure modes |
| V | Project memory: `CLAUDE.md` |
| VI | Skills, subagents, deny rules and hooks, MCP (demo 3: the accuracy skill, then a verifier subagent) |
| VII | Autonomous loops: `/goal` builds the solver against five tests (demo 4); the verification figure; guardrails |
| VIII | From plan to table and figure: the reconciled plan, the silent failure, four loss kernels (demo 5), paper replication |
| IX | Verification, disclosure, referees and supervisors, replication packages, data governance |
| Wrap-up | Quick reference, takeaways, conclusion, references |

Block 1 is Parts I–IV, Block 2 Parts V–IX, with a break in between. Five live demos, nothing to install to attend.

## 5. The lead example

The stochastic Brock–Mirman growth model. A deep equilibrium net (DEQN; [Azinovic, Gaegauf and
Scheidegger, 2022](https://onlinelibrary.wiley.com/doi/abs/10.1111/iere.12575)) learns the savings rate by gradient descent on the Euler residual. Two known
answers grade it: the closed form of the full-depreciation case (Step 1) and a conventional grid-based
reference solver (Steps 2 and 4–7). Every concept of the lecture is introduced on this example; Part VIII shows a run
whose loss converged while the policy was wrong, and the check that caught it. Details and every
number: [`code/deqn/README.md`](code/deqn/README.md).

## 6. Readings

**The method of the lead example**

* Azinovic, M., L. Gaegauf and S. Scheidegger (2022). Deep Equilibrium Nets. *International Economic Review* 63(4), 1471–1525. [doi:10.1111/iere.12575](https://onlinelibrary.wiley.com/doi/abs/10.1111/iere.12575)

**Agents in economic research**

* Korinek, A. (2023). Generative AI for economic research: use cases and implications for economists. *Journal of Economic Literature* 61(4), 1281–1317.
* Hans, A. and I. Bilionis (2026). Coding-agents can replicate scientific machine learning papers. [arXiv:2607.02134](https://arxiv.org/abs/2607.02134); the skill: [paper-replication-paper](https://github.com/PredictiveScienceLab/paper-replication-paper).
* Bilionis, I. et al. (2026). We asked an AI agent to replicate three papers. *ebilionis.substack.com*.
* Schwartz, M., I. Andrews and J. M. Shapiro (2026). An LLM workflow that reproduces, improves, and extends published economics research. [NBER Working Paper 35782](https://www.nber.org/papers/w35782).
* Galiani, S., F. A. López and R. A. Sosa (2026). AI agents and prompt engineering in econometric coding. [NBER Working Paper 35588](https://www.nber.org/papers/w35588).

**Working with Claude Code**

* Goldsmith-Pinkham, P. (2026). Getting started with Claude Code. *paulgp.substack.com*.
* Cherny, B. (2025). 10 Claude Code tips from the creator of Claude Code. Thread on X.
* Osmani, A. (2026). Which Claude model should you build with? [claude.dev blog](https://claude.dev/blog/which-claude-model-should-you-build-with/), 9 November 2026.
* Schluntz, E. and B. Zhang (2024). Building effective agents. Anthropic engineering blog.
* Anthropic (2026). [Claude Code documentation](https://code.claude.com/docs).

The full list, with the sources behind every slide, is on the two reference slides at the end of the deck.

## 7. Author

**Simon Scheidegger**

* Department of Economics, HEC, University of Lausanne; Grantham Research Institute, London School of Economics
* Website: [sischei.github.io](https://sischei.github.io/)
* Lecture material: [github.com/sischei/agentic_programming_munich_1126](https://github.com/sischei/agentic_programming_munich_1126)
* Licence of this repository: [CC0 1.0](LICENCE).

---
name: deqn-accuracy
description: Audit a saved DEQN policy (.pt with its .json sidecar) on the path its own policy generates. Reports Euler-residual percentiles, the distance to the reference policy, E[s] against the reference and whether the path stayed inside the training window, with a verdict. Never retrains, never reads the training loss as accuracy.
---

# /deqn-accuracy

## Description
The on-path accuracy check of a solved policy. A training loss measures the residual
where the training states are; this skill measures it where the policy sends the
economy, and against the reference solver. Runs in about 5 seconds.

## Instructions

When the user invokes `/deqn-accuracy <path to .pt>`, do the following.

### Step 1: Identify the policy file
- Use the path the user gave. If none, list `results/*.pt` and ask which one.
- The `.json` sidecar written by `deqn.common.save_policy` must sit next to it; if it is
  missing, stop and say so (the model, parameters and architecture come from it).

### Step 2: Run the check
```
python3 .claude/skills/deqn-accuracy/accuracy.py <path to .pt> --plot notes/accuracy_<stem>.pdf
```
The script simulates 2,000 periods with the network's own policy (seed 7, 200 periods
burn-in), evaluates the Euler residual there with a 5-node Gauss-Hermite expectation
written independently of the solver, compares the policy with the reference (the grid-based
solver `brock_mirman_reference.solve(beta, rho)` at the sidecar's beta and rho for the
stochastic model, the closed form s* = alpha beta for the deterministic one) and checks
that the simulated path stayed inside the training window. If the path left the window,
the policy gap is measured on the reference solver's own path instead.

### Step 3: Report
Reproduce the script's five lines verbatim, then one sentence of interpretation:
```
policy: <file>  (<model>; reference: <what>)
|Euler residual| on the network's own path: mean .. p50 .. p90 .. p99 .. max ..
|s - s_ref| on <the own path | the reference solver's path>: mean .. max ..;   E[s] = .. vs reference ..
path range [..] vs training window [..]: inside | OUTSIDE
verdict: ACCURATE | DEGRADED | WRONG
```
Verdict rule (in the script): ACCURATE if mean |r| < 1e-3, max |s - s_ref| < 1e-2 and the
path is inside the window; WRONG if mean |r| > 1e-2 or the path left the window; DEGRADED
otherwise. A residual is always reported with its scale (relative consumption).

### Step 4: Save
Write the report to `notes/accuracy_<policy file stem>_<YYYY-MM-DD>.md` and mention the figure
(`--plot`): its three panels are the policy against the reference, the residual along the
simulated path against the test threshold, and the path against the training window.

## Constraints
- Never retrain, never modify the `.pt` or its sidecar, never edit `tests/` or `reference/`.
- Never quote the training loss from the sidecar as an accuracy measure; it is not one.
- If the reference solver is not importable, say so and limit the verdict to the residual
  and the window check.

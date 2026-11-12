#!/usr/bin/env bash
# Scaffold the project every live demo of the lecture runs in.
#
#   bash code/setup_demo_project.sh [TARGET]      default TARGET=<repository>/demo
#
# The demo project lives inside the repository (demo/, git-ignored, with its own git
# history), so the teaching event is self-contained: nothing is written to the home directory.
#
# Copies the reference solver, the project CLAUDE.md (CLAUDE_md_demo.md),
# the DEQN scaffold (deqn/common.py, the stub deqn/stochastic.py, the TensorFlow
# originals, the five tests and the reference policy), three skills, eight subagents
# and the settings file (permission deny rules + hooks) into TARGET, then makes the
# first commit.
# Safe to re-run: existing files are overwritten with the repository versions,
# an existing git history is kept.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$(dirname "$HERE")/demo}"

mkdir -p "$TARGET"/{data/raw,results,figures,notes,scripts,deqn/source,tests,reference,.claude/skills,.claude/agents}

cp "$HERE/brock_mirman_reference.py"                            "$TARGET/"
cp "$HERE/CLAUDE_md_demo.md"                                    "$TARGET/CLAUDE.md"
printf 'Stand-in for confidential raw data. The permission rules in .claude/settings.json deny reading this folder\nand the PreToolUse hook blocks edits; try "> Append a line to data/raw/test.txt" to see both.\n' > "$TARGET/data/raw/README.txt"
cp "$HERE/deqn/__init__.py" "$HERE/deqn/common.py" "$HERE/deqn/stochastic.py" "$TARGET/deqn/"
cp "$HERE/deqn/plan_deqn.md" "$HERE/deqn/silent_failure.md"   "$TARGET/deqn/"   # the Part VIII reading, referenced by the agents
cp "$HERE"/deqn/source/*.py                                     "$TARGET/deqn/source/"
cp "$HERE"/deqn/tests/*.py                                      "$TARGET/tests/"
cp "$HERE"/deqn/reference/*                                     "$TARGET/reference/"
cp -r "$HERE/skills/data-diagnostics"                           "$TARGET/.claude/skills/"
cp -r "$HERE/skills/strategic-revision"                         "$TARGET/.claude/skills/"
cp -r "$HERE/skills/deqn-accuracy"                              "$TARGET/.claude/skills/"
cp "$HERE/deqn/accuracy.py"                                     "$TARGET/.claude/skills/deqn-accuracy/accuracy.py"
cp "$HERE"/agents/*.md                                          "$TARGET/.claude/agents/"
cp "$HERE/hooks/settings.json"                                  "$TARGET/.claude/settings.json"

cat > "$TARGET/.gitignore" <<'GI'
__pycache__/
.pytest_cache/
.claude/audit.log
GI

cd "$TARGET"
if [ ! -d .git ]; then
  git init -q && git branch -M main
fi
git add -A
if git diff --cached --quiet; then
  echo "nothing new to commit in $TARGET"
else
  git commit -q -m "seed: demo project (solver, DEQN scaffold, tests, CLAUDE.md, skills, agents, hooks)"
fi

echo
echo "Demo project ready: $TARGET"
echo "  brock_mirman_reference.py (the reference solver)"
echo "  deqn/{common.py,stochastic.py (stub),source/}, tests/ (5 tests), reference/ (reference policy + baseline.md)"
echo "  CLAUDE.md, data/raw/ (stand-in), .claude/skills/{data-diagnostics,deqn-accuracy,strategic-revision}, .claude/agents/ (8), .claude/settings.json"
echo
echo "Check:  cd $TARGET && python3 -m pytest -q tests/     (must fail with NotImplementedError: the /goal fills deqn/stochastic.py)"
echo
echo "Next:  cd $TARGET && claude"
echo "Then follow code/WALKTHROUGH.md; the prompts are in code/prompts.md."
for t in jq ruff latexmk; do command -v $t >/dev/null || echo "note: $t is not installed; a hook in .claude/settings.json uses it (delete that hook or install it)."; done

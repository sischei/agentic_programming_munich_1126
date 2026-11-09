#!/usr/bin/env bash
# Scaffold the project every live demo of the lecture runs in.
#
#   bash code/setup_demo_project.sh [TARGET]      default TARGET=~/research/munich-demo
#
# Copies the data, the reference solver, the project CLAUDE.md (CLAUDE_md_demo.md),
# two skills, eight subagents and the settings file (permission deny rules + hooks)
# into TARGET, then makes the first commit.
# Safe to re-run: existing files are overwritten with the repository versions,
# an existing git history is kept.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="${1:-$HOME/research/munich-demo}"

mkdir -p "$TARGET"/{data/raw,results,figures,notes,scripts,.claude/skills,.claude/agents}

cp "$HERE/data/bm_panel.csv" "$HERE/data/bm_panel_truth.csv" "$TARGET/data/"
cp "$HERE/brock_mirman_reference.py"                            "$TARGET/"
cp "$HERE/CLAUDE_md_demo.md"                                    "$TARGET/CLAUDE.md"
printf 'Stand-in for confidential raw data. The permission rules in .claude/settings.json deny reading this folder\nand the PreToolUse hook blocks edits; try "> Append a line to data/raw/test.txt" to see both.\n' > "$TARGET/data/raw/README.txt"
cp -r "$HERE/skills/data-diagnostics"                           "$TARGET/.claude/skills/"
cp -r "$HERE/skills/strategic-revision"                         "$TARGET/.claude/skills/"
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
  git commit -q -m "seed: demo project (data, solver, CLAUDE.md, skills, agents, hooks)"
fi

echo
echo "Demo project ready: $TARGET"
echo "  data/bm_panel.csv, data/bm_panel_truth.csv, brock_mirman_reference.py"
echo "  CLAUDE.md, data/raw/ (stand-in), .claude/skills/{data-diagnostics,strategic-revision}, .claude/agents/ (8), .claude/settings.json"
echo
echo "Next:  cd $TARGET && claude"
echo "Then follow code/WALKTHROUGH.md; the prompts are in code/prompts.md."
for t in jq ruff latexmk; do command -v $t >/dev/null || echo "note: $t is not installed; a hook in .claude/settings.json uses it (delete that hook or install it)."; done

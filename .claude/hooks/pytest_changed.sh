#!/usr/bin/env bash
# Stop-hook helper for vpts.
#
# Fires when Claude finishes a turn. To stay fast and non-annoying it:
#   1. does nothing unless there are *uncommitted* Python changes under vpts/ or tests/;
#   2. runs pytest scoped to the changed test files (a changed vpts/<...>/X.py maps to
#      tests/test_X.py), falling back to the FULL suite only if a change has no matching
#      test file;
#   3. is non-blocking: on green it prints nothing; on red it emits a JSON systemMessage
#      so the failure surfaces to you without ending or looping the turn.
#
# Set VPTS_HOOK_DRYRUN=1 to print the would-run target instead of running pytest.
# Reads (and ignores) the hook payload on stdin.
set -u
cat >/dev/null 2>&1 || true                              # drain the hook's stdin JSON

command -v git >/dev/null 2>&1 || exit 0
ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" || exit 0
cd "$ROOT" || exit 0

# Uncommitted (staged + unstaged + untracked) .py files under vpts/ or tests/.
changed="$(git status --porcelain --no-renames 2>/dev/null \
  | sed 's/^...//' | grep -E '\.py$' | grep -E '^(vpts/|tests/)' || true)"
[ -z "$changed" ] && exit 0

targets="" ; full=0
while IFS= read -r f; do
  [ -z "$f" ] && continue
  case "$f" in
    tests/test_*.py) [ -f "$f" ] && targets="$targets $f" ;;
    vpts/*) t="tests/test_$(basename "$f" .py).py"
            if [ -f "$t" ]; then targets="$targets $t"; else full=1; fi ;;
  esac
done <<EOF
$changed
EOF

targets="$(printf '%s\n' $targets | sed '/^$/d' | sort -u | tr '\n' ' ')"
if [ "$full" -eq 1 ] || [ -z "${targets// /}" ]; then
  set --                                                  # no args → full suite
  label="full suite"
else
  # shellcheck disable=SC2086
  set -- $targets
  label="$targets"
fi

if [ -n "${VPTS_HOOK_DRYRUN:-}" ]; then
  echo "would run: python -m pytest -q ${*:-<full suite>}   (label: $label)"
  exit 0
fi

out="$(python -m pytest -q "$@" 2>&1)" ; code=$?
[ "$code" -eq 0 ] && exit 0                               # green → silent

printf '%s\n' "$out" | tail -n 6 | jq -Rs --arg label "$label" \
  '{systemMessage: ("⚠️ vpts: pytest FAILED on uncommitted changes (" + $label + "):\n" + .)}'
exit 0

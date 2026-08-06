#!/usr/bin/env bash
# Isolated Lance/vector pytest runner.
#
# Why: mixing tests/storage/test_vectors.py + test_vector_rebuild.py into the
# default/full suite can segfault (Lance native). Run them alone via this script.
# Do NOT add these modules to the default aggregate (use --ignore there).
#
# Usage:
#   tests/run_vector_suite.sh              # run vector suite
#   tests/run_vector_suite.sh --dry-run    # print command only
#   tests/run_vector_suite.sh --help
# Extra args after options are forwarded to pytest (e.g. -k rebuild -v).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VECTOR_TESTS=(
  tests/storage/test_vectors.py
  tests/storage/test_vector_rebuild.py
)

usage() {
  cat <<'EOF'
Usage: tests/run_vector_suite.sh [--dry-run] [--help] [pytest-args...]

Run ONLY Lance/vector tests in isolation (not part of the default aggregate):
  tests/storage/test_vectors.py
  tests/storage/test_vector_rebuild.py

Options:
  --help, -h     Show this help and exit
  --dry-run      Print the pytest command without executing it

Environment:
  Prefer .venv-tests/bin/pytest when present; else pytest on PATH.
  PYTHONPATH defaults to ".:../maibot-plugin-sdk" (repo + sibling SDK).

Examples:
  tests/run_vector_suite.sh
  tests/run_vector_suite.sh --dry-run
  tests/run_vector_suite.sh -v -k dimension
EOF
}

DRY_RUN=0
PYTEST_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    -h|--help)
      usage
      exit 0
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      PYTEST_ARGS+=("$@")
      break
      ;;
    *)
      PYTEST_ARGS+=("$1")
      shift
      ;;
  esac
done

if [[ -x "$ROOT/.venv-tests/bin/pytest" ]]; then
  PYTEST=("$ROOT/.venv-tests/bin/pytest")
elif command -v pytest >/dev/null 2>&1; then
  PYTEST=(pytest)
else
  echo "error: pytest not found (.venv-tests/bin/pytest or PATH)" >&2
  exit 127
fi

export PYTHONPATH="${PYTHONPATH:-.:../maibot-plugin-sdk}"

CMD=("${PYTEST[@]}" "${VECTOR_TESTS[@]}" "${PYTEST_ARGS[@]}")

if [[ "$DRY_RUN" -eq 1 ]]; then
  printf '+ PYTHONPATH=%q' "$PYTHONPATH"
  printf ' %q' "${CMD[@]}"
  printf '\n'
  exit 0
fi

exec "${CMD[@]}"

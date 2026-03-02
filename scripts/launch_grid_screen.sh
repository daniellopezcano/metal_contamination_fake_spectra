#!/usr/bin/env bash
set -euo pipefail

SIM="${1:-}"
NSPEC="${2:-}"
NBINS="${3:-}"
SNAP="${4:-25}"

if [[ -z "$SIM" || -z "$NSPEC" || -z "$NBINS" ]]; then
  echo "Usage: $0 <SIM> <NSPEC> <NBINS> [SNAP]"
  echo "Example: $0 TNG50-4 8 512 25"
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOGDIR="${REPO_ROOT}/outputs/logs"
mkdir -p "$LOGDIR"

STAMP="$(date +%Y%m%d_%H%M%S)"
SESSION="grid_${SIM}_n${NSPEC}_b${NBINS}_s${SNAP}_${STAMP}"
LOGFILE="${LOGDIR}/${SESSION}.log"

# Find conda base (works if conda is installed/available in your current shell)
if ! command -v conda >/dev/null 2>&1; then
  echo "ERROR: conda not found in PATH. Open a terminal where 'conda' works and run again."
  exit 1
fi
CONDA_BASE="$(conda info --base)"

CMD="python scripts/grid.py --sim ${SIM} --nspec ${NSPEC} --nbins ${NBINS} --snap ${SNAP} --overwrite"

echo "Launching screen session: ${SESSION}"
echo "Logfile: ${LOGFILE}"
echo "Repo: ${REPO_ROOT}"
echo "Command: ${CMD}"
echo ""

# Run inside screen:
# - cd to repo
# - source conda.sh (so 'conda activate' works in non-interactive shell)
# - activate env
# - run command, tee to logfile
# - keep shell open afterwards (so screen doesn't vanish instantly)
screen -dmS "${SESSION}" bash -lc "
  set -e
  cd '${REPO_ROOT}'
  source '${CONDA_BASE}/etc/profile.d/conda.sh'
  conda activate metal-fs
  echo '[env]' \$(which python)
  echo '[run]' ${CMD}
  ${CMD} 2>&1 | tee '${LOGFILE}'
  echo ''
  echo '[done] finished. Log: ${LOGFILE}'
  exec bash
"

echo "Attach with:  screen -r ${SESSION}"
echo "Detach with:  Ctrl-A then D"
echo "Tail log:     tail -f ${LOGFILE}"

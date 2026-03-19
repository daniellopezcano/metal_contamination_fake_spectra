#!/usr/bin/env bash
set -euo pipefail

BASE="${1:?BASE missing}"
NUM="${2:?NUM missing}"
NSPEC="${3:?NSPEC missing}"
AXIS="${4:?AXIS missing}"
NBINS="${5:?NBINS missing}"
OUT_DIR="${6:?OUT_DIR missing}"
PRESET="${7:?PRESET missing}"
KERNEL="${8:?KERNEL missing}"
RANKS="${9:?RANKS missing}"

CONDA_BASE="${CONDA_BASE:-/nfs/pic.es/user/d/dlopezca/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fake_spectra}"

# Where your repo lives on /nfs (pass via submit file)
SCRIPTS_DIR="${PROJECT_SCRIPTS_DIR:?PROJECT_SCRIPTS_DIR not set}"

# Activate conda (PIC: .bashrc is not loaded by default) :contentReference[oaicite:1]{index=1}
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

# MPI ranks = processes. Prevent oversubscription from threaded libs.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE

echo "=== HTCondor MPI run_grid_skewers.py ==="
echo "host: $(hostname)"
echo "pwd : $(pwd)"
echo "SCRIPTS_DIR: ${SCRIPTS_DIR}"
echo "_CONDOR_SCRATCH_DIR: ${_CONDOR_SCRATCH_DIR:-}"
echo "python: $(which python)"
python -V
echo "MPI ranks: ${RANKS}"

# Ensure python can import mcfs from your repo root
REPO_ROOT="$(cd "${SCRIPTS_DIR}/.." && pwd)"
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

SCRIPT="${SCRIPTS_DIR}/run_grid_skewers.py"
ls -l "${SCRIPT}"

echo
/usr/bin/time -p mpiexec -n "${RANKS}" python "${SCRIPT}" \
  --base "${BASE}" \
  --num "${NUM}" \
  --nspec "${NSPEC}" \
  --axis "${AXIS}" \
  --nbins "${NBINS}" \
  --out-dir "${OUT_DIR}" \
  --preset "${PRESET}" \
  --overwrite \
  --force-recompute-tau \
  --kernel "${KERNEL}" \
#   --compute-density \
#   --compute-temperature \
#   --compute-velocity-los
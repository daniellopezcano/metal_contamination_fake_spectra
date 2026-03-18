#!/usr/bin/env bash
set -euo pipefail

# Positional args (provided by HTCondor "arguments = ...")
BASE="${1:?BASE missing}"
NUM="${2:?NUM missing}"
NSPEC="${3:?NSPEC missing}"
AXIS="${4:?AXIS missing}"
NBINS="${5:?NBINS missing}"
OUT_DIR="${6:?OUT_DIR missing}"
PRESET="${7:?PRESET missing}"
KERNEL="${8:?KERNEL missing}"
NCPUS="${9:?NCPUS missing}"

# Conda setup (provided by submit file via environment=..., with sane defaults)
CONDA_BASE="${CONDA_BASE:-/nfs/pic.es/user/d/dlopezca/miniconda3}"
CONDA_ENV="${CONDA_ENV:-fake_spectra}"

# Activate conda inside batch shell (conda.sh may reference unset vars)
set +u
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
set -u

# Cap threads to allocated CPUs (avoid oversubscription)
export OMP_NUM_THREADS="${NCPUS}"
export MKL_NUM_THREADS="${NCPUS}"
export OPENBLAS_NUM_THREADS="${NCPUS}"
export NUMEXPR_NUM_THREADS="${NCPUS}"
export NUMBA_NUM_THREADS="${NCPUS}"
export MKL_DYNAMIC=FALSE
export OMP_DYNAMIC=FALSE
export OMP_PROC_BIND=true
export OMP_PLACES=cores

echo "=== HTCondor run_grid_skewers.py ==="
echo "host: $(hostname)"
echo "pwd : $(pwd)"
echo "_CONDOR_SCRATCH_DIR: ${_CONDOR_SCRATCH_DIR:-}"
echo "python: $(which python)"
python -V
echo "NCPUS: ${NCPUS}"
echo "ARGS: base=${BASE} num=${NUM} nspec=${NSPEC} axis=${AXIS} nbins=${NBINS}"
echo "      out_dir=${OUT_DIR} preset=${PRESET} kernel=${KERNEL}"
echo

/usr/bin/time -p python run_grid_skewers.py \
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
  --compute-density \
  --compute-temperature \
  --compute-velocity-los

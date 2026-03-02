# metal_contamination_fake_spectra

Small testbed repo to:
1) Extract **gridded skewers** (HI + metal lines) from TNG snapshots using `fake_spectra`.
2) Store results in a reproducible way (HDF5 + a tiny JSON manifest).
3) Run convergence scans over `(simulation, nspec, nbins/resolution, axis)` and later compute P1D.

## Environment setup (conda) + local fake_spectra install (with multi-CPU support)

These steps create a clean conda environment, install the required dependencies,
and then compile/install `fake_spectra` from the **local clone** at:

`/home/dlopez/Documentos/0.profesional/Postdoc/USP/Projects/fake_spectra`

### 0) Preconditions (what fake_spectra needs)
- Required Python libs: `numpy`, `h5py` (and `scipy` is listed in setup metadata).
- Required C library: **GSL** (`gsl-config` must be available at build time).
- Multi-core within a node: uses **OpenMP** if the compiler supports it.
- Multi-node / multi-process: optional **MPI** via `mpi4py` (run with `mpirun/mpiexec`).
- Important for IllustrisTNG/Arepo (Voronoi mesh): when using MPI you must pass `kernel="tophat"`
  to `RandSpectra` / `GriddedSpectra`.

(These points are described in the upstream README and build script.)

### 1) Create the conda environment
From anywhere:

```bash
conda create -n metal-fs -c conda-forge python=3.11 -y
conda activate metal-fs
```

### 2) Install core Python stack + build deps

Install scientific packages + build requirements:
```bash
conda install -c conda-forge -y \
  numpy scipy h5py matplotlib jupyterlab \
  gsl \
  pip setuptools wheel \
  compilers make
```

Notes:
- gsl provides gsl-config needed during compilation.
- compilers ensures you have a GCC toolchain with OpenMP support inside conda.

### 3) (Optional) Install MPI support in the same env

If you want to run with mpirun and use multiple CPUs across processes:
```bash
conda install -c conda-forge -y openmpi mpi4py
```
Quick checks:
```bash
which mpirun
python -c "import mpi4py; print('mpi4py ok')"
```

### 4) Build & install fake_spectra from your local clone

Go to your local fake_spectra repo and initialize submodules (required by upstream instructions):
```bash
cd /home/dlopez/Documentos/0.profesional/Postdoc/USP/Projects/fake_spectra
git submodule update --init
```

Now install it into the active conda env.

Recommended (editable install):
```bash
pip install -e .
```

Alternative (upstream-style install):

```bash
python setup.py build
python setup.py install
```

### 5) Sanity checks

Check that the compiled extension imports:
```bash
python -c "import fake_spectra; import fake_spectra._spectra_priv; print('fake_spectra import OK')"
```

Check that GSL was visible at build time:

```bash
which gsl-config
gsl-config --version
```






















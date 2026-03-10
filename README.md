# Installing `fake_spectra` from source

This guide installs the **fake_spectra** package from a local clone using a dedicated **conda environment**.

---

# 1. Define repository path

Set the path where the repository is cloned.

```bash
export FAKE_SPECTRA_REPO="/home/dlopez/Documentos/0.profesional/Postdoc/USP/Projects/fake_spectra"
cd $FAKE_SPECTRA_REPO
```

---

# 2. Create conda environment

```bash
conda create -n fake_spectra -c conda-forge python=3.11 -y
conda activate fake_spectra
```

---

# 3. Install dependencies

Core Python libraries:

```bash
conda install -c conda-forge numpy scipy h5py matplotlib -y
```

Required C library:

```bash
conda install -c conda-forge gsl -y
```

Compiler toolchain (required to build the C++ extension):

```bash
conda install -c conda-forge compilers make pkg-config -y
```

Optional (MPI support):

```bash
conda install -c conda-forge mpi4py openmpi -y
```

---

# 4. Initialize git submodules

```bash
git submodule update --init --recursive
```

---

# 5. Install the package

Upgrade build tools and install in editable mode:

```bash
python -m pip install -U pip setuptools wheel
python -m pip install -e .
```

This step compiles the internal C++ extension used by `fake_spectra`.

---

# 5. Install the mcfs package
```bash
cd PATH_FOR_metal_contamination_fake_spectra
pip install -e .
```

---

# 6. Verify installation

Test that the package imports correctly:

```bash
python -c "import fake_spectra; import fake_spectra.spectra; print('ok', fake_spectra.__file__)"
```

Check that the compiled module loads:

```bash
python -c "import fake_spectra._spectra_priv as sp; print('loaded', sp)"
```

Check library versions:

```bash
python -c "import numpy, h5py, scipy; print('numpy', numpy.__version__, 'h5py', h5py.__version__, 'scipy', scipy.__version__)"
```

```bash
python -c "from fake_spectra.griddedspectra import GriddedSpectra; print('ok')"
```

If all commands run without errors, the installation is complete.

---

# 7. Activate environment for use

Whenever using `fake_spectra`, activate the environment:

```bash
conda activate fake_spectra
```

---

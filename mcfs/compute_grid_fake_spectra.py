from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from fake_spectra.griddedspectra import GriddedSpectra

def default_savefile(
    base: str,
    num: int,
    nspec: int,
    axis: int,
    nbins: Optional[int] = None,
    res_kms: Optional[float] = None,
    tag: Optional[str] = None,
) -> str:
    """
    Build a default output filename for a gridded fake_spectra run.
    """
    sim = Path(base).name
    parts = [sim, f"snap{num:03d}", f"nspec{nspec}", f"axis{axis}"]
    parts.append(f"nbins{nbins}" if nbins is not None else f"res{res_kms}kms")
    if tag:
        parts.append(str(tag))
    return "grid_" + "_".join(parts) + ".hdf5"


def _setup_logger(logger: Optional[logging.Logger] = None, level: int = logging.INFO) -> logging.Logger:
    """
    Create or configure a logger for the run.
    """
    if logger is not None:
        return logger

    logger = logging.getLogger("fake_spectra_runner")
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    logger.propagate = False
    return logger


def run_gridded_skewers(
    *,
    base: str,
    num: int,
    nspec: int,
    axis: int = -1,
    nbins: Optional[int] = None,
    res_kms: Optional[float] = None,
    lines: Optional[List[Tuple[str, int, int]]] = None,
    out_dir: str = "outputs/grids",
    savefile: Optional[str] = None,
    tag: Optional[str] = None,
    overwrite: bool = False,
    force_recompute_tau: bool = False,
    kernel: Optional[str] = None,
    MPI: Any = None,
    quiet: bool = False,
    compute_density: bool = False,
    compute_temperature: bool = False,
    compute_velocity_los: bool = False,
    logger: Optional[logging.Logger] = None,
) -> Tuple[GriddedSpectra, Dict[str, Any]]:
    """
    Build/load a fake_spectra gridded skewer set and compute requested quantities.

    Parameters
    ----------
    base
        Simulation base directory, e.g. "/home/STORAGE/TNG50-4".
    num
        Snapshot number.
    nspec
        Number of skewers per transverse dimension. Total skewers = nspec^2.
    axis
        LOS axis. Use 1, 2, 3, or leave as supported by fake_spectra/GriddedSpectra.
    nbins
        Number of LOS bins. Mutually exclusive with `res_kms`.
    res_kms
        LOS resolution in km/s. Mutually exclusive with `nbins`.
    lines
        List of line tuples: (element, ion, lambda_floor).
    out_dir
        Directory where HDF5 and manifest are saved.
    savefile
        Explicit output filename. If None, a default one is generated.
    tag
        Optional extra tag appended to the default filename.
    overwrite
        If True, rebuild from snapshot even if output file already exists.
    force_recompute_tau
        Passed to `get_tau`.
    kernel
        Interpolation kernel. If MPI is enabled and kernel is None, defaults to "tophat".
    MPI
        Usually `mpi4py.MPI` when running under mpirun, otherwise None.
    quiet
        Passed through to fake_spectra.
    compute_density
        If True, compute ion densities for unique ions appearing in `lines`.
    compute_temperature
        If True, compute ion temperatures for unique ions appearing in `lines`.
    compute_velocity_los
        If True, compute LOS velocity for unique ions appearing in `lines`.
    logger
        Optional Python logger.

    Returns
    -------
    gs, info
        gs : GriddedSpectra object
        info : dictionary with metadata, output paths, and optionally computed arrays
    """
    log = _setup_logger(logger)

    if (nbins is None) == (res_kms is None):
        raise ValueError("Provide exactly one of nbins or res_kms.")

    if lines is None:
        lines = []

    if MPI is not None and kernel is None:
        kernel = "tophat"

    rank = 0 if MPI is None else MPI.COMM_WORLD.Get_rank()
    size = 1 if MPI is None else MPI.COMM_WORLD.Get_size()

    if rank == 0:
        log.info("Starting gridded skewer run")
        log.info("base=%s", base)
        log.info("snapshot=%03d | nspec=%d | axis=%s", num, nspec, axis)
        log.info("nbins=%s | res_kms=%s | kernel=%s | MPI ranks=%d", nbins, res_kms, kernel, size)

    out_path = Path(out_dir)
    if rank == 0:
        out_path.mkdir(parents=True, exist_ok=True)
        log.info("Output directory ready: %s", out_path)

    if MPI is not None:
        MPI.COMM_WORLD.Barrier()

    if savefile is None:
        savefile = default_savefile(
            base=base,
            num=num,
            nspec=nspec,
            axis=axis,
            nbins=nbins,
            res_kms=res_kms,
            tag=tag,
        )

    fullpath = out_path / savefile
    reload_file = overwrite or (not fullpath.exists())

    if rank == 0:
        if reload_file:
            log.info("Will build/load from snapshot and write output: %s", fullpath)
        else:
            log.info("Existing file found. Will reload from saved file: %s", fullpath)

    gs = GriddedSpectra(
        num=num,
        base=base,
        nspec=nspec,
        axis=axis,
        nbins=nbins,
        res=res_kms,
        savefile=savefile,
        savedir=str(out_path),
        reload_file=reload_file,
        MPI=MPI,
        kernel=kernel,
        quiet=quiet,
    )

    if rank == 0:
        try:
            log.info(
                "Spectra object ready: nbins=%s | dvbin=%.3f km/s | vmax=%.3f km/s",
                gs.nbins,
                gs.dvbin,
                gs.vmax,
            )
        except Exception:
            log.info("Spectra object ready.")

    tau_keys: List[str] = []
    tau_data: Dict[str, Any] = {}
    density_data: Dict[str, Any] = {}
    temperature_data: Dict[str, Any] = {}
    vlos_data: Dict[str, Any] = {}

    # Compute tau for each requested transition
    for elem, ion, lam in lines:
        key = f"{elem}{ion}_{lam}"
        if rank == 0:
            log.info("Computing tau for %s", key)
        tau = gs.get_tau(elem, ion, lam, force_recompute=force_recompute_tau)
        tau_data[key] = tau
        tau_keys.append(key)

        if rank == 0:
            try:
                log.info("Computed %s with shape %s", key, tau.shape)
            except Exception:
                log.info("Computed %s", key)

    # Compute per-ion extra fields only once per unique ion
    unique_ions = sorted({(elem, ion) for elem, ion, _ in lines})

    for elem, ion in unique_ions:
        ion_key = f"{elem}{ion}"

        if compute_density:
            if rank == 0:
                log.info("Computing density for %s", ion_key)
            arr = gs.get_density(elem, ion)
            density_data[ion_key] = arr
            if rank == 0:
                try:
                    log.info("Density %s shape %s", ion_key, arr.shape)
                except Exception:
                    log.info("Density %s computed", ion_key)

        if compute_temperature:
            if rank == 0:
                log.info("Computing temperature for %s", ion_key)
            arr = gs.get_temp(elem, ion)
            temperature_data[ion_key] = arr
            if rank == 0:
                try:
                    log.info("Temperature %s shape %s", ion_key, arr.shape)
                except Exception:
                    log.info("Temperature %s computed", ion_key)

        if compute_velocity_los:
            if rank == 0:
                log.info("Computing LOS velocity for %s", ion_key)
            vvec = gs.get_velocity(elem, ion)
            # fake_spectra uses axis=1,2,3
            if axis not in (1, 2, 3, -1):
                raise ValueError("compute_velocity_los requires axis to be 1, 2, 3, or -1")
            arr = vvec[:, :, axis - 1]
            vlos_data[ion_key] = arr
            if rank == 0:
                try:
                    log.info("LOS velocity %s shape %s", ion_key, arr.shape)
                except Exception:
                    log.info("LOS velocity %s computed", ion_key)

    if rank == 0:
        log.info("Saving HDF5 output")
    gs.save_file()

    if MPI is not None:
        MPI.COMM_WORLD.Barrier()

    manifest_path = fullpath.with_suffix(fullpath.suffix + ".json")

    if rank == 0:
        manifest = {
            "hdf5_path": str(fullpath),
            "base": base,
            "num": num,
            "nspec": nspec,
            "axis": axis,
            "nbins": nbins,
            "res_kms": res_kms,
            "kernel": kernel,
            "lines": [
                {"elem": e, "ion": i, "lam": l, "key": f"{e}{i}_{l}"}
                for (e, i, l) in lines
            ],
            "tau_keys": tau_keys,
            "computed_extras": {
                "density": compute_density,
                "temperature": compute_temperature,
                "velocity_los": compute_velocity_los,
            },
            "overwrite": overwrite,
            "force_recompute_tau": force_recompute_tau,
            "mpi": {
                "enabled": MPI is not None,
                "ranks": size,
            },
        }

        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        log.info("Manifest written to: %s", manifest_path)
        log.info("Run completed successfully")

    info: Dict[str, Any] = {
        "savepath": str(fullpath),
        "manifest": str(manifest_path),
        "tau": tau_data,
        "density": density_data,
        "temperature": temperature_data,
        "vlos": vlos_data,
        "tau_keys": tau_keys,
    }

    return gs, info
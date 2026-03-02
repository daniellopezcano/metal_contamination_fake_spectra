import json
from pathlib import Path
from fake_spectra.griddedspectra import GriddedSpectra


def default_savefile(base, num, nspec, axis, nbins=None, res_kms=None, tag=None):
    sim = Path(base).name
    parts = [sim, f"snap{num:03d}", f"nspec{nspec}", f"axis{axis}"]
    parts.append(f"nbins{nbins}" if nbins is not None else f"res{res_kms}kms")
    if tag:
        parts.append(str(tag))
    return "grid_" + "_".join(parts) + ".hdf5"


def run_gridded_tau(
    *,
    base,
    num,
    nspec,
    axis=-1,
    nbins=None,
    res_kms=None,
    lines=None,                 # list of (elem, ion, lam_floor) tuples
    out_dir="outputs/grids",
    savefile=None,
    overwrite=False,
    force_recompute_tau=False,
    kernel=None,
    MPI=None,                   # pass mpi4py.MPI when running under mpirun
    quiet=False,
):
    """
    Build/load a fake_spectra grid and compute tau for requested lines.
    Saves HDF5 + a JSON manifest next to it.

    Returns:
      gs : GriddedSpectra
      info : dict with paths
    """
    if (nbins is None) == (res_kms is None):
        raise ValueError("Provide exactly one of nbins or res_kms.")
    if lines is None:
        lines = []

    # For Arepo/Voronoi + MPI, safest default is tophat (fake_spectra recommendation)
    if MPI is not None and kernel is None:
        kernel = "tophat"

    out_dir = Path(out_dir)
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        out_dir.mkdir(parents=True, exist_ok=True)
    if MPI is not None:
        MPI.COMM_WORLD.Barrier()

    if savefile is None:
        savefile = default_savefile(base, num, nspec, axis, nbins=nbins, res_kms=res_kms)

    fullpath = out_dir / savefile
    reload_file = overwrite or (not fullpath.exists())   # True => build from snapshot; False => load file

    gs = GriddedSpectra(
        num=num,
        base=base,
        nspec=nspec,
        axis=axis,
        nbins=nbins,
        res=res_kms,
        savefile=savefile,
        savedir=str(out_dir),
        reload_file=reload_file,
        MPI=MPI,
        kernel=kernel,
        quiet=quiet,
    )

    tau_keys = []
    for elem, ion, lam in lines:
        key = f"{elem}{ion}_{lam}"
        gs.get_tau(elem, ion, lam, force_recompute=force_recompute_tau)
        tau_keys.append(key)

    gs.save_file()
    if MPI is not None:
        MPI.COMM_WORLD.Barrier()

    # Manifest (rank 0 only)
    mpath = fullpath.with_suffix(fullpath.suffix + ".json")
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        manifest = {
            "hdf5_path": str(Path(gs.savefile)),
            "base": base,
            "num": num,
            "nspec": nspec,
            "axis": axis,
            "nbins": nbins,
            "res_kms": res_kms,
            "kernel": kernel,
            "lines": [{"elem": e, "ion": i, "lam": l, "key": f"{e}{i}_{l}"} for (e, i, l) in lines],
            "tau_keys": tau_keys,
            "overwrite": overwrite,
            "force_recompute_tau": force_recompute_tau,
            "mpi": {"enabled": MPI is not None, "ranks": (MPI.COMM_WORLD.Get_size() if MPI is not None else 1)},
        }
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    return gs, {"savepath": str(fullpath), "manifest": str(mpath)}
import os
import json
from pathlib import Path
from fake_spectra.griddedspectra import GriddedSpectra
from collections import defaultdict


def load_run(
    *,
    base_load_path: str,
    simulation_name: str,
    num: int,
    nspec: int,
    axis: int,
    nbins: int,
    quiet: bool = True,
    read_timing: bool = True,
):
    """
    Load a fake_spectra gridded run produced by our pipeline.

    Inputs:
      base_load_path   e.g. "/home/STORAGE/SKEWERS"
      simulation_name  e.g. "TNG50-4"
      num              snapshot number, e.g. 25
      nspec            grid size, e.g. 2
      axis             e.g. -1
      nbins            LOS bins, e.g. 1024

    Returns:
      gs       GriddedSpectra object loaded from HDF5
      man      dict: contents of the .json manifest
      timing   str or None: contents of .timing.txt if present (and read_timing=True)
    """
    snapdir = Path(base_load_path) / simulation_name / f"snapdir_{num:03d}"
    manifest_path = snapdir / f"grid_{simulation_name}_snap{num:03d}_nspec{nspec}_axis{axis}_nbins{nbins}.hdf5.json"

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found:\n  {manifest_path}")

    with open(manifest_path, "r", encoding="utf-8") as f:
        man = json.load(f)

    hdf5_path = Path(man["hdf5_path"])
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 referenced by manifest not found:\n  {hdf5_path}")

    savedir = str(hdf5_path.parent)
    savefile = hdf5_path.name

    gs = GriddedSpectra(
        num=man["num"],
        base=man["base"],
        nspec=man["nspec"],
        axis=man["axis"],
        nbins=man["nbins"],
        res=man.get("res_kms", None),
        savefile=savefile,
        savedir=savedir,
        reload_file=False,  # load existing file
        quiet=quiet,
    )

    timing = None
    if read_timing:
        timing_path = hdf5_path.with_suffix(hdf5_path.suffix + ".timing.txt")
        if timing_path.exists():
            timing = timing_path.read_text(encoding="utf-8")

    return gs, man, timing


def load_runs(
    *,
    base_load_path: str,
    list_simulation_name,
    list_num,
    list_nspec,
    axis: int,
    list_nbins,
    quiet: bool = True,
    read_timing: bool = True,
    strict: bool = False,
):
    """
    Bulk loader for many fake_spectra grid runs.

    Returns:
      gs_dict[sim][num][nspec][nbins]     -> GriddedSpectra
      man_dict[sim][num][nspec][nbins]    -> manifest dict
      timing_dict[sim][num][nspec][nbins] -> timing text or None
      missing -> list of tuples (sim, num, nspec, nbins, error_message) for files not found (if strict=False)
    """
    gs_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    man_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    timing_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))

    missing = []

    for sim in list_simulation_name:
        for num in list_num:
            for nspec in list_nspec:
                for nbins in list_nbins:
                    try:
                        gs, man, timing = load_run(
                            base_load_path=base_load_path,
                            simulation_name=sim,
                            num=num,
                            nspec=nspec,
                            axis=axis,
                            nbins=nbins,
                            quiet=quiet,
                            read_timing=read_timing,
                        )
                        gs_dict[sim][num][nspec][nbins] = gs
                        man_dict[sim][num][nspec][nbins] = man
                        timing_dict[sim][num][nspec][nbins] = timing
                    except Exception as e:
                        if strict:
                            raise
                        missing.append((sim, num, nspec, nbins, str(e)))

    # convert defaultdicts -> plain dicts (nicer to print / serialize)
    gs_dict = {sim: {n: {ns: dict(bins) for ns, bins in ns_map.items()}
                     for n, ns_map in num_map.items()}
               for sim, num_map in gs_dict.items()}
    man_dict = {sim: {n: {ns: dict(bins) for ns, bins in ns_map.items()}
                      for n, ns_map in num_map.items()}
                for sim, num_map in man_dict.items()}
    timing_dict = {sim: {n: {ns: dict(bins) for ns, bins in ns_map.items()}
                         for n, ns_map in num_map.items()}
                   for sim, num_map in timing_dict.items()}

    return gs_dict, man_dict, timing_dict, missing

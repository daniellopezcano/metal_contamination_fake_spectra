from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Optional, Sequence

import h5py
import numpy as np
from fake_spectra.plot_spectra import PlottingSpectra


# =============================================================================
# Basic conventions used throughout this file
# =============================================================================
#
# A "line" is represented as:
#     (elem, ion, lam)
# for example:
#     ("H", 1, 1215)
#
# A "line key" is a string such as:
#     "H1_1215"
#
# Main array convention:
#     (N_lines, N_skewers, N_spectral)
#
# where:
#   - N_lines     = number of spectral transitions loaded
#   - N_skewers   = number of skewers, possibly already concatenated across axes
#   - N_spectral  = number of pixels along each skewer
# =============================================================================

Line = tuple[str, int, int]

AXIS_RE = re.compile(r"^axis_(\d+)$")
DEFAULT_FIELDS = ("tau", "n", "T", "vlos")


# =============================================================================
# Logging helpers
# =============================================================================

def get_logger(name: str = "load_runs", level: int = logging.INFO) -> logging.Logger:
    """
    Create or retrieve a module logger.

    Parameters
    ----------
    name
        Logger name.
    level
        Logging level, e.g. logging.INFO or logging.DEBUG.

    Returns
    -------
    logging.Logger
    """
    log = logging.getLogger(name)

    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(
            logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S")
        )
        log.addHandler(handler)

    log.setLevel(level)
    log.propagate = False
    return log


# =============================================================================
# Generic small helpers
# =============================================================================

def is_listlike(x: Any) -> bool:
    """
    Return True if x should be interpreted as a sweep over multiple values.

    Strings and paths are treated as scalars, not as iterables.
    """
    if isinstance(x, (str, bytes, Path)):
        return False
    if isinstance(x, np.ndarray):
        return x.ndim > 0
    return isinstance(x, (list, tuple))


def to_list(x: Any) -> list[Any]:
    """
    Convert a list-like object into a plain Python list.

    Also converts numpy scalar values into native Python scalars.
    """
    if isinstance(x, np.ndarray):
        x = x.tolist()
    return [v.item() if isinstance(v, np.generic) else v for v in x]


def parse_scalar(text: str) -> Any:
    """
    Best-effort conversion of a string into int, float, or fallback string.
    """
    for cast in (int, float):
        try:
            return cast(text)
        except ValueError:
            pass
    return text


# =============================================================================
# Sidecar readers
# =============================================================================

def read_json(path: Path, log: logging.Logger) -> dict[str, Any]:
    """
    Read a JSON sidecar file. If missing, return an empty dictionary.
    """
    if not path.exists():
        log.warning("JSON file not found: %s", path)
        return {}

    log.debug("Reading JSON sidecar: %s", path)
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_keyval_txt(path: Path, log: logging.Logger) -> dict[str, Any]:
    """
    Read a simple key:value text file into a dictionary.

    This is used for files such as the run summary sidecar.
    """
    if not path.exists():
        log.warning("Text file not found: %s", path)
        return {}

    log.debug("Reading text sidecar: %s", path)

    out: dict[str, Any] = {}
    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.strip()
            if not line or ":" not in line:
                continue
            key, value = map(str.strip, line.split(":", 1))
            out[key] = parse_scalar(value)

    return out


def inspect_hdf5(path: Path) -> dict[str, Any]:
    """
    Inspect HDF5 file attributes and dataset shapes/dtypes.

    This is useful for debugging or validating saved files.
    """
    out = {"attrs": {}, "datasets": {}}

    with h5py.File(path, "r") as f:
        out["attrs"] = dict(f.attrs)

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                out["datasets"][name] = {
                    "shape": obj.shape,
                    "dtype": str(obj.dtype),
                }

        f.visititems(visitor)

    return out


# =============================================================================
# Metadata helpers
# =============================================================================

def lines_from_manifest(manifest: dict[str, Any]) -> list[Line]:
    """
    Extract line definitions from the manifest JSON.

    Expected manifest format:
        manifest["lines"] = [
            {"elem": "...", "ion": ..., "lam": ...},
            ...
        ]
    """
    return [
        (entry["elem"], int(entry["ion"]), int(entry["lam"]))
        for entry in manifest.get("lines", [])
    ]


def line_key(line: Line) -> str:
    """
    Convert a line tuple into a stable string key.
    """
    elem, ion, lam = line
    return f"{elem}{ion}_{lam}"


# =============================================================================
# Path helpers
# =============================================================================

def build_filename(
    sim_name: str,
    snap_num: int,
    axis: int,
    delta_grid: float,
    res_kms: float,
    preset: str = "lya_si",
    tag: Optional[str] = None,
) -> str:
    """
    Build the expected HDF5 filename for one saved run.
    """
    name = (
        f"grid_{sim_name}_snap{snap_num:03d}"
        f"_axis{axis}"
        f"_delta_grid_{delta_grid:g}"
        f"_res_kms_{res_kms:g}"
        f"_{preset}"
    )
    if tag:
        name += f"_{tag}"
    return name + ".hdf5"


def snapdir(base_dir: str, sim_name: str, snap_num: int) -> Path:
    """
    Directory containing axis_1, axis_2, ... for one snapshot.
    """
    return Path(base_dir) / "SKEWERS" / sim_name / f"snapdir_{snap_num:03d}"


def sim_base(base_dir: str, sim_name: str) -> Path:
    """
    Base path passed to PlottingSpectra for the simulation.
    """
    return Path(base_dir) / sim_name


def available_axes(base_dir: str, sim_name: str, snap_num: int) -> list[int]:
    """
    Detect which axis directories exist, e.g. axis_1, axis_2, axis_3.
    """
    root = snapdir(base_dir, sim_name, snap_num)
    if not root.exists():
        raise FileNotFoundError(f"Snapshot skewer directory not found: {root}")

    axes: list[int] = []
    for p in sorted(root.iterdir()):
        if p.is_dir():
            m = AXIS_RE.match(p.name)
            if m:
                axes.append(int(m.group(1)))

    return axes


# =============================================================================
# Internal array-building helpers
# =============================================================================

def _stack_tau_arrays(sp: PlottingSpectra, lines: Sequence[Line]) -> np.ndarray:
    """
    Compute tau for all lines and stack into one array.

    Output shape:
        (N_lines, N_skewers, N_spectral)
    """
    return np.stack(
        [sp.get_tau(elem, ion, lam) for elem, ion, lam in lines],
        axis=0,
    )


def _build_ion_caches(
    sp: PlottingSpectra,
    lines: Sequence[Line],
    axis: int,
    log: logging.Logger,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]]:
    """
    Load ion-level quantities only once per ion.

    Returns
    -------
    n_ion, T_ion, vlos_ion : dict
        Dictionaries keyed by ion, e.g. "H1", "Si3".
    """
    unique_ions = sorted({(elem, ion) for elem, ion, _ in lines})

    n_ion: dict[str, np.ndarray] = {}
    T_ion: dict[str, np.ndarray] = {}
    vlos_ion: dict[str, np.ndarray] = {}

    for elem, ion in unique_ions:
        ion_key = f"{elem}{ion}"

        try:
            n_ion[ion_key] = sp.get_density(elem, ion)
            log.debug("Loaded density for %s with shape %s", ion_key, n_ion[ion_key].shape)
        except Exception as e:
            log.warning("Could not load density for %s: %s", ion_key, e)

        try:
            T_ion[ion_key] = sp.get_temp(elem, ion)
            log.debug("Loaded temperature for %s with shape %s", ion_key, T_ion[ion_key].shape)
        except Exception as e:
            log.warning("Could not load temperature for %s: %s", ion_key, e)

        try:
            # fake_spectra returns velocity vector components; select the LOS component
            v = sp.get_velocity(elem, ion)
            vlos_ion[ion_key] = v[:, :, axis - 1]
            log.debug("Loaded LOS velocity for %s with shape %s", ion_key, vlos_ion[ion_key].shape)
        except Exception as e:
            log.warning("Could not load LOS velocity for %s: %s", ion_key, e)

    return n_ion, T_ion, vlos_ion


def _build_line_field_array(
    lines: Sequence[Line],
    ion_cache: dict[str, np.ndarray],
    field_name: str,
    axis: int,
    log: logging.Logger,
) -> tuple[Optional[np.ndarray], list[str]]:
    """
    Convert an ion-level cache into a line-ordered stacked array.

    Example:
        if cache has values for "H1" and "Si3", this function builds an array
        ordered according to `lines`, repeating the ion-level array for each
        line belonging to that ion.

    Returns
    -------
    arr, keys
        arr  : (N_lines_present, N_skewers, N_spectral) or None
        keys : list of line keys in the same order as the first array axis
    """
    arrs: list[np.ndarray] = []
    keys: list[str] = []

    for line in lines:
        elem, ion, lam = line
        lk = line_key(line)
        ion_key = f"{elem}{ion}"

        if ion_key not in ion_cache:
            log.debug("Axis %d: field '%s' missing for line %s", axis, field_name, lk)
            continue

        arrs.append(ion_cache[ion_key])
        keys.append(lk)

    if not arrs:
        return None, []

    arr = np.stack(arrs, axis=0)
    return arr, keys


def _ensure_same_spectral_grid(axis_runs: dict[int, dict[str, Any]]) -> tuple[int, float, np.ndarray]:
    """
    Check that all loaded axes share the same spectral grid.

    Returns
    -------
    nbins, dvbin, v_kms
    """
    axes_loaded = sorted(axis_runs.keys())
    first = axis_runs[axes_loaded[0]]

    nbins_ref = first["meta"]["nbins"]
    dvbin_ref = first["meta"]["dvbin"]
    v_kms = np.arange(nbins_ref) * dvbin_ref

    for ax in axes_loaded[1:]:
        meta = axis_runs[ax]["meta"]
        if meta["nbins"] != nbins_ref:
            raise ValueError(f"Inconsistent nbins for axis {ax}: {meta['nbins']} != {nbins_ref}")
        if not np.isclose(meta["dvbin"], dvbin_ref):
            raise ValueError(f"Inconsistent dvbin for axis {ax}: {meta['dvbin']} != {dvbin_ref}")

    return nbins_ref, dvbin_ref, v_kms


def _concatenate_field_across_axes(
    axis_runs: dict[int, dict[str, Any]],
    axes_loaded: Sequence[int],
    field: str,
) -> tuple[Optional[np.ndarray], list[str]]:
    """
    Concatenate one field across the skewer axis for all loaded axes.

    Each per-axis field must already have shape:
        (N_lines, N_skewers, N_spectral)

    The result has shape:
        (N_lines, N_skewers_total, N_spectral)
    """
    axes_with_field = [ax for ax in axes_loaded if field in axis_runs[ax]]
    if not axes_with_field:
        return None, []

    ref_keys = axis_runs[axes_with_field[0]]["field_keys"][field]

    for ax in axes_with_field[1:]:
        keys = axis_runs[ax]["field_keys"][field]
        if keys != ref_keys:
            raise ValueError(f"Inconsistent line ordering for field '{field}' in axis {ax}")

    arr = np.concatenate([axis_runs[ax][field] for ax in axes_with_field], axis=1)
    return arr, ref_keys


# =============================================================================
# One-axis loader
# =============================================================================

def load_axis_run(
    *,
    hdf5_path: str | Path,
    base: str | Path,
    num: int,
    axis: int,
    lines: Optional[Sequence[Line]] = None,
    use_res: Optional[float] = None,
    inspect_file: bool = False,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """
    Load one saved HDF5 file corresponding to a single axis.

    This function already returns fields in the shape:
        (N_lines, N_skewers, N_spectral)

    Parameters
    ----------
    hdf5_path
        Path to the saved fake_spectra HDF5 file.
    base
        Simulation base path passed to PlottingSpectra.
    num
        Snapshot number.
    axis
        Axis identifier (1, 2, or 3).
    lines
        Optional list of lines to load. If omitted, inferred from manifest.
    use_res
        Resolution passed to PlottingSpectra when re-opening the file.
    inspect_file
        Whether to inspect HDF5 structure.
    logger
        Optional logger.

    Returns
    -------
    dict
        Dictionary containing arrays and metadata for one axis.
    """
    log = logger or get_logger()

    hdf5_path = Path(hdf5_path)
    manifest_path = Path(str(hdf5_path) + ".json")
    summary_path = hdf5_path.with_suffix("").with_name(hdf5_path.with_suffix("").name + "_summary.txt")

    log.info("Loading axis %d from file: %s", axis, hdf5_path)

    manifest = read_json(manifest_path, log)
    summary = read_keyval_txt(summary_path, log)

    if lines is None:
        lines = lines_from_manifest(manifest)
    lines = list(lines)

    if not lines:
        raise ValueError(f"No spectral lines available for file: {hdf5_path}")

    file_info = inspect_hdf5(hdf5_path) if inspect_file else {}

    # This object gives access to get_tau, get_density, get_temp, get_velocity, etc.
    sp = PlottingSpectra(
        num=num,
        base=str(base),
        savefile=str(hdf5_path),
        label=f"axis_{axis}",
        res=use_res,
    )

    line_keys = [line_key(line) for line in lines]

    meta = {
        "nbins": int(sp.nbins),
        "dvbin": float(sp.dvbin),
        "vmax": float(sp.vmax),
        "red": float(sp.red),
    }

    log.debug(
        "Axis %d spectral grid: nbins=%d, dvbin=%.6f, vmax=%.6f, z=%.6f",
        axis, meta["nbins"], meta["dvbin"], meta["vmax"], meta["red"]
    )

    # Optical depth is naturally line-based already.
    tau = _stack_tau_arrays(sp, lines)
    log.debug("Axis %d tau shape: %s", axis, tau.shape)

    # Other fields are loaded once per ion and then expanded to line order.
    n_ion, T_ion, vlos_ion = _build_ion_caches(sp, lines, axis, log)

    n, n_keys = _build_line_field_array(lines, n_ion, "n", axis, log)
    T, T_keys = _build_line_field_array(lines, T_ion, "T", axis, log)
    vlos, vlos_keys = _build_line_field_array(lines, vlos_ion, "vlos", axis, log)

    out = {
        "axis": axis,
        "sp": sp,
        "manifest": manifest,
        "summary": summary,
        "file_info": file_info,
        "line_keys": line_keys,
        "field_keys": {"tau": line_keys},
        "tau": tau,
        "meta": {
            **meta,
            "nlines": int(tau.shape[0]),
            "nskewers": int(tau.shape[1]),
        },
        "paths": {
            "hdf5": str(hdf5_path),
            "manifest": str(manifest_path),
            "summary": str(summary_path),
        },
    }

    if n is not None:
        out["n"] = n
        out["field_keys"]["n"] = n_keys
        log.debug("Axis %d n shape: %s", axis, n.shape)

    if T is not None:
        out["T"] = T
        out["field_keys"]["T"] = T_keys
        log.debug("Axis %d T shape: %s", axis, T.shape)

    if vlos is not None:
        out["vlos"] = vlos
        out["field_keys"]["vlos"] = vlos_keys
        log.debug("Axis %d vlos shape: %s", axis, vlos.shape)

    return out


# =============================================================================
# One-case loader: load all axes and stack them immediately
# =============================================================================

def load_case(
    *,
    base_dir: str,
    sim_name: str,
    snap_num: int,
    delta_grid: float,
    res_kms: float,
    preset: str = "lya_si",
    tag: Optional[str] = None,
    axes: Optional[Sequence[int]] = None,
    lines: Optional[Sequence[Line]] = None,
    inspect_file: bool = False,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """
    Load one realization and immediately stack all selected axes.

    The returned field arrays have shape:
        (N_lines, N_skewers_total, N_spectral)

    This is the main object you want for subsequent analysis.
    """
    log = logger or get_logger()

    log.info(
        "Loading case: sim=%s, snap=%03d, delta_grid=%s, res_kms=%s, preset=%s",
        sim_name, snap_num, delta_grid, res_kms, preset
    )

    all_axes = available_axes(base_dir, sim_name, snap_num)
    axes_to_load = all_axes if axes is None else sorted(set(axes))

    invalid = [ax for ax in axes_to_load if ax not in all_axes]
    if invalid:
        raise ValueError(f"Requested axes {invalid} not available. Available axes: {all_axes}")

    axis_runs: dict[int, dict[str, Any]] = {}
    missing_files: dict[int, str] = {}

    for axis in axes_to_load:
        path = snapdir(base_dir, sim_name, snap_num) / f"axis_{axis}" / build_filename(
            sim_name=sim_name,
            snap_num=snap_num,
            axis=axis,
            delta_grid=delta_grid,
            res_kms=res_kms,
            preset=preset,
            tag=tag,
        )

        if not path.exists():
            log.warning("Missing file for axis %d: %s", axis, path)
            missing_files[axis] = str(path)
            continue

        axis_runs[axis] = load_axis_run(
            hdf5_path=path,
            base=sim_base(base_dir, sim_name),
            num=snap_num,
            axis=axis,
            lines=lines,
            use_res=res_kms,
            inspect_file=inspect_file,
            logger=log,
        )

    if not axis_runs:
        raise FileNotFoundError("No axis files could be loaded for this case.")

    axes_loaded = sorted(axis_runs.keys())

    nbins, dvbin, v_kms = _ensure_same_spectral_grid(axis_runs)

    out = {
        "base_dir": base_dir,
        "sim_name": sim_name,
        "snap_num": snap_num,
        "delta_grid": delta_grid,
        "res_kms": res_kms,
        "preset": preset,
        "tag": tag,
        "axes_available": all_axes,
        "axes_loaded": axes_loaded,
        "missing_files": missing_files,
        "sp": {ax: axis_runs[ax]["sp"] for ax in axes_loaded},
        "v_kms": v_kms,
        "field_keys": {},
        "by_axis": axis_runs,  # kept on purpose for debugging and inspection
        "meta": {
            "nbins": nbins,
            "dvbin": dvbin,
            "nskewers_per_axis": {ax: axis_runs[ax]["meta"]["nskewers"] for ax in axes_loaded},
            "total_nskewers": sum(axis_runs[ax]["meta"]["nskewers"] for ax in axes_loaded),
            "nlines_per_field": {},
        },
    }

    for field in DEFAULT_FIELDS:
        arr, keys = _concatenate_field_across_axes(axis_runs, axes_loaded, field)
        if arr is None:
            continue

        out[field] = arr
        out["field_keys"][field] = keys
        out["meta"]["nlines_per_field"][field] = len(keys)

        log.debug("Stacked field '%s' shape: %s", field, arr.shape)

    log.info(
        "Finished case: loaded %d axes, total skewers = %d",
        len(axes_loaded),
        out["meta"]["total_nskewers"],
    )

    return out


# =============================================================================
# Unified public loader:
#   - scalar inputs  -> returns one stacked case
#   - one list input -> returns a dictionary of stacked cases
# =============================================================================

def load_data(
    *,
    base_dir,
    sim_name,
    snap_num,
    delta_grid,
    res_kms,
    preset="lya_si",
    tag: Optional[str] = None,
    axes: Optional[Sequence[int]] = None,
    lines: Optional[Sequence[Line]] = None,
    inspect_file: bool = False,
    logger: Optional[logging.Logger] = None,
) -> dict[str, Any]:
    """
    Unified top-level loader.

    Behavior
    --------
    1. If all inputs are scalars:
       returns a single stacked case dictionary.

    2. If exactly one among
         [base_dir, sim_name, snap_num, delta_grid, res_kms, preset]
       is list-like:
       returns a sweep dictionary whose entries are already stacked cases.

    Parameters
    ----------
    base_dir, sim_name, snap_num, delta_grid, res_kms, preset
        Loader configuration. At most one may be list-like.
    tag
        Optional filename tag.
    axes
        Optional list of axes to load.
    lines
        Optional fixed list of spectral lines.
    inspect_file
        Whether to inspect the HDF5 structure.
    logger
        Optional logger.

    Returns
    -------
    dict
        Either:
        - one stacked case, or
        - a sweep dictionary:
            {
                "sweep_param": ...,
                "sweep_values": [...],
                "fixed_inputs": {...},
                "data": {value: stacked_case, ...}
            }
    """
    log = logger or get_logger()

    params = {
        "base_dir": base_dir,
        "sim_name": sim_name,
        "snap_num": snap_num,
        "delta_grid": delta_grid,
        "res_kms": res_kms,
        "preset": preset,
    }

    swept = {k: to_list(v) for k, v in params.items() if is_listlike(v)}
    fixed = {k: v for k, v in params.items() if not is_listlike(v)}

    # -------------------------------------------------------------------------
    # Case 1: no sweep requested -> return one stacked case
    # -------------------------------------------------------------------------
    if len(swept) == 0:
        log.info("No sweep parameter detected. Loading one stacked case.")
        return load_case(
            base_dir=base_dir,
            sim_name=sim_name,
            snap_num=snap_num,
            delta_grid=delta_grid,
            res_kms=res_kms,
            preset=preset,
            tag=tag,
            axes=axes,
            lines=lines,
            inspect_file=inspect_file,
            logger=log,
        )

    # -------------------------------------------------------------------------
    # Case 2: more than one sweep parameter -> ambiguous, reject
    # -------------------------------------------------------------------------
    if len(swept) > 1:
        raise ValueError(f"Only one input may be list-like. Found: {list(swept.keys())}")

    # -------------------------------------------------------------------------
    # Case 3: exactly one sweep parameter -> return dictionary of stacked cases
    # -------------------------------------------------------------------------
    sweep_param, sweep_values = next(iter(swept.items()))
    log.info("Detected sweep over '%s' with %d values", sweep_param, len(sweep_values))

    data: dict[Any, dict[str, Any]] = {}

    for value in sweep_values:
        kwargs = dict(fixed)
        kwargs[sweep_param] = value

        log.info("Loading sweep case: %s = %s", sweep_param, value)

        data[value] = load_case(
            base_dir=kwargs["base_dir"],
            sim_name=kwargs["sim_name"],
            snap_num=kwargs["snap_num"],
            delta_grid=kwargs["delta_grid"],
            res_kms=kwargs["res_kms"],
            preset=kwargs["preset"],
            tag=tag,
            axes=axes,
            lines=lines,
            inspect_file=inspect_file,
            logger=log,
        )

    return {
        "sweep_param": sweep_param,
        "sweep_values": sweep_values,
        "fixed_inputs": fixed,
        "data": data,
    }


def build_pairwise_delta_v_dict(line_info, lambda0_key="lambda0", c_kms=299792.458):
    """
    Compute pairwise velocity separations

        delta_v(i,j) = c_kms * ln(lambda_i / lambda_j)

    for all unordered pairs i < j.

    Returns
    -------
    pair_delta_v_dict : dict
        Keys are tuples (k1, k2), values are the corresponding delta_v in km/s.
    """
    keys = [k for k in line_info if lambda0_key in line_info[k]]
    pair_delta_v_dict = {}

    for i in range(len(keys)):
        for j in range(i + 1, len(keys)):
            k1, k2 = keys[i], keys[j]
            lam1 = line_info[k1][lambda0_key]
            lam2 = line_info[k2][lambda0_key]
            pair_delta_v_dict[(k1, k2)] = c_kms * np.log(lam1 / lam2)

    return pair_delta_v_dict
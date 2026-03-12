from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import h5py
from fake_spectra.plot_spectra import PlottingSpectra


def _logger(name: str = "mcfs.load", level: int = logging.INFO) -> logging.Logger:
    log = logging.getLogger(name)
    if not log.handlers:
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s", "%H:%M:%S"))
        log.addHandler(h)
    log.setLevel(level)
    log.propagate = False
    return log


def _read_manifest(path: str | Path, log: logging.Logger) -> Dict[str, Any]:
    path = Path(path)
    if not path.exists():
        log.warning("Manifest not found: %s", path)
        return {}
    log.info("Loading manifest: %s", path)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _read_keyval_txt(path: str | Path, log: logging.Logger) -> Dict[str, Any]:
    """
    Read a simple key: value text file, such as the timing file written by the runner.
    Tries to cast numeric values when possible.
    """
    path = Path(path)
    if not path.exists():
        log.warning("Text sidecar file not found: %s", path)
        return {}

    log.info("Loading text sidecar file: %s", path)
    out: Dict[str, Any] = {}

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or ":" not in line:
                continue
            key, value = line.split(":", 1)
            key = key.strip()
            value = value.strip()

            # Try int
            try:
                out[key] = int(value)
                continue
            except ValueError:
                pass

            # Try float
            try:
                out[key] = float(value)
                continue
            except ValueError:
                pass

            # Fallback to string
            out[key] = value

    return out


def _inspect_hdf5(path: str | Path, log: logging.Logger) -> Dict[str, Any]:
    path = Path(path)
    out: Dict[str, Any] = {"attrs": {}, "datasets": {}}
    log.info("Inspecting HDF5 file: %s", path)

    with h5py.File(path, "r") as f:
        out["attrs"] = dict(f.attrs)

        def visitor(name, obj):
            if isinstance(obj, h5py.Dataset):
                out["datasets"][name] = {"shape": obj.shape, "dtype": str(obj.dtype)}

        f.visititems(visitor)

    log.info("Found %d datasets", len(out["datasets"]))
    return out


def _lines_from_manifest(manifest: Dict[str, Any]) -> List[Tuple[str, int, int]]:
    return [(d["elem"], int(d["ion"]), int(d["lam"])) for d in manifest.get("lines", [])]


def _read_header_pixel_width(hdf5_path: str | Path) -> Optional[float]:
    with h5py.File(hdf5_path, "r") as f:
        if "Header" in f and "nbins" in f["Header"].attrs:
            hdr = f["Header"].attrs
            if "dvbin" in hdr:
                return float(hdr["dvbin"])
    return None


def load_saved_fake_spectra(
    *,
    hdf5_path: str,
    base: str,
    num: int,
    axis: int,
    manifest_path: Optional[str] = None,
    timing_path: Optional[str] = None,
    lines: Optional[List[Tuple[str, int, int]]] = None,
    label: str = "Loaded spectra",
    inspect_file: bool = True,
    use_res: Optional[float] = None,
    logger: Optional[logging.Logger] = None,
) -> Dict[str, Any]:
    """
    Load a saved fake_spectra file and return line-level dictionaries:
      tau[line_key], n[line_key], T[line_key], vlos[line_key]

    Also loads companion sidecar files when available:
      - manifest JSON
      - timing TXT

    Notes
    -----
    We pass res=None by default when reopening saved files to avoid fake_spectra's
    pixel-width assertion against its internal default res=1 km/s.
    """
    log = logger or _logger()

    hdf5_path = Path(hdf5_path)
    stem = hdf5_path.with_suffix("")

    manifest_path = Path(manifest_path) if manifest_path is not None else Path(str(hdf5_path) + ".json")
    timing_path = Path(timing_path) if timing_path is not None else Path(str(stem) + "_timing.txt")

    manifest = _read_manifest(manifest_path, log)
    timing_info = _read_keyval_txt(timing_path, log)

    if lines is None:
        lines = _lines_from_manifest(manifest)
    if not lines:
        raise ValueError("No lines provided and none found in manifest.")

    file_info = _inspect_hdf5(hdf5_path, log) if inspect_file else {}

    if use_res is None:
        log.info("Opening saved spectra with res=None to avoid pixel-width assertion")
    else:
        log.info("Opening saved spectra with res=%s km/s", use_res)

    sp = PlottingSpectra(
        num=num,
        base=base,
        savefile=str(hdf5_path),
        label=label,
        res=use_res,
    )

    meta = {}
    for attr in ("nbins", "dvbin", "vmax", "red"):
        if hasattr(sp, attr):
            meta[attr] = getattr(sp, attr)
            log.info("sp.%s = %s", attr, meta[attr])

    tau: Dict[str, Any] = {}
    n_ion: Dict[str, Any] = {}
    T_ion: Dict[str, Any] = {}
    vlos_ion: Dict[str, Any] = {}

    for elem, ion, lam in lines:
        key = f"{elem}{ion}_{lam}"
        log.info("Loading tau for %s", key)
        tau[key] = sp.get_tau(elem, ion, lam)
        log.info("  tau[%s].shape = %s", key, tau[key].shape)

    unique_ions = sorted({(elem, ion) for elem, ion, _ in lines})
    for elem, ion in unique_ions:
        ion_key = f"{elem}{ion}"

        try:
            log.info("Loading density for %s", ion_key)
            n_ion[ion_key] = sp.get_density(elem, ion)
            log.info("  n[%s].shape = %s", ion_key, n_ion[ion_key].shape)
        except Exception as e:
            log.warning("Could not load density for %s: %s", ion_key, e)

        try:
            log.info("Loading temperature for %s", ion_key)
            T_ion[ion_key] = sp.get_temp(elem, ion)
            log.info("  T[%s].shape = %s", ion_key, T_ion[ion_key].shape)
        except Exception as e:
            log.warning("Could not load temperature for %s: %s", ion_key, e)

        try:
            log.info("Loading LOS velocity for %s", ion_key)
            vvec = sp.get_velocity(elem, ion)
            vlos_ion[ion_key] = vvec[:, :, axis - 1]
            log.info("  vlos[%s].shape = %s", ion_key, vlos_ion[ion_key].shape)
        except Exception as e:
            log.warning("Could not load velocity for %s: %s", ion_key, e)

    n: Dict[str, Any] = {}
    T: Dict[str, Any] = {}
    vlos: Dict[str, Any] = {}

    for elem, ion, lam in lines:
        line_key = f"{elem}{ion}_{lam}"
        ion_key = f"{elem}{ion}"
        if ion_key in n_ion:
            n[line_key] = n_ion[ion_key]
        if ion_key in T_ion:
            T[line_key] = T_ion[ion_key]
        if ion_key in vlos_ion:
            vlos[line_key] = vlos_ion[ion_key]

    log.info("Finished loading file.")
    log.info("Loaded tau keys    : %s", list(tau.keys()))
    log.info("Loaded n keys      : %s", list(n.keys()))
    log.info("Loaded T keys      : %s", list(T.keys()))
    log.info("Loaded vlos keys   : %s", list(vlos.keys()))
    log.info("Loaded timing keys : %s", list(timing_info.keys()))

    return {
        "sp": sp,
        "manifest": manifest,
        "timing": timing_info,
        "file_info": file_info,
        "tau": tau,
        "n": n,
        "T": T,
        "vlos": vlos,
        "meta": meta,
        "lines": lines,
        "paths": {
            "hdf5": str(hdf5_path),
            "manifest": str(manifest_path),
            "timing": str(timing_path),
        },
    }
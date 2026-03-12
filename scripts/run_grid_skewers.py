#!/usr/bin/env python3

import argparse
import time
from pathlib import Path

from mcfs.compute_grid_fake_spectra import run_gridded_skewers


LINE_PRESETS = {
    "lya": [("H", 1, 1215)],
    "lya_si": [
        ("H", 1, 1215),
        ("Si", 3, 1206),
        ("Si", 2, 1190),
        ("Si", 2, 1193),
    ],
}


def build_savefile(base, num, nspec, axis, nbins, preset, tag=None):
    sim_name = Path(base).name
    name = f"grid_{sim_name}_snap{num:03d}_nspec{nspec}_axis{axis}_nbins{nbins}_{preset}"
    if tag:
        name += f"_{tag}"
    return name + ".hdf5"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run gridded fake_spectra skewers from terminal."
    )

    parser.add_argument("--base", required=True, help="Simulation base path")
    parser.add_argument("--num", required=True, type=int, help="Snapshot number")
    parser.add_argument("--nspec", required=True, type=int, help="Grid size")
    parser.add_argument("--axis", required=True, type=int, choices=[1, 2, 3, -1], help="LOS axis")
    parser.add_argument("--nbins", required=True, type=int, help="Number of LOS bins")
    parser.add_argument("--out-dir", required=True, help="Output directory")

    parser.add_argument(
        "--preset",
        default="lya",
        choices=LINE_PRESETS.keys(),
        help="Set of spectral lines to compute",
    )

    parser.add_argument("--savefile", default=None, help="Optional explicit output filename")
    parser.add_argument("--tag", default=None, help="Optional extra tag for filename")
    parser.add_argument("--kernel", default="tophat", help="fake_spectra kernel")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--force-recompute-tau", action="store_true")
    parser.add_argument("--compute-density", action="store_true")
    parser.add_argument("--compute-temperature", action="store_true")
    parser.add_argument("--compute-velocity-los", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    return parser.parse_args()


def main():
    args = parse_args()

    lines = LINE_PRESETS[args.preset]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    savefile = args.savefile
    if savefile is None:
        savefile = build_savefile(
            base=args.base,
            num=args.num,
            nspec=args.nspec,
            axis=args.axis,
            nbins=args.nbins,
            preset=args.preset,
            tag=args.tag,
        )

    t0 = time.time()

    gs, info = run_gridded_skewers(
        base=args.base,
        num=args.num,
        nspec=args.nspec,
        axis=args.axis,
        nbins=args.nbins,
        lines=lines,
        out_dir=str(out_dir),
        savefile=savefile,
        overwrite=args.overwrite,
        force_recompute_tau=args.force_recompute_tau,
        kernel=args.kernel,
        compute_density=args.compute_density,
        compute_temperature=args.compute_temperature,
        compute_velocity_los=args.compute_velocity_los,
        quiet=args.quiet,
    )

    elapsed = time.time() - t0

    timing_file = out_dir / (Path(savefile).stem + "_timing.txt")
    with open(timing_file, "w") as f:
        f.write(f"savefile: {savefile}\n")
        f.write(f"savepath: {info['savepath']}\n")
        f.write(f"manifest: {info['manifest']}\n")
        f.write(f"tau_keys: {info['tau_keys']}\n")
        f.write(f"elapsed_seconds: {elapsed:.3f}\n")
        f.write(f"elapsed_minutes: {elapsed/60:.3f}\n")

    print("\nRun finished.")
    print("HDF5 file :", info["savepath"])
    print("Manifest  :", info["manifest"])
    print("Tau keys  :", info["tau_keys"])
    print(f"Time [s]  : {elapsed:.3f}")
    print(f"Timing txt: {timing_file}")


if __name__ == "__main__":
    main()
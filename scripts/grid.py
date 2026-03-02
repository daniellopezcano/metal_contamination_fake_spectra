#!/usr/bin/env python3
from mcfs.fs_grid import run_gridded_tau
import os
import time
import argparse
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Run fake_spectra gridded tau extraction.")
    p.add_argument("--sim", required=True, help="Simulation name, e.g. TNG50-4")
    p.add_argument("--nspec", type=int, required=True, help="Grid nspec (skewers per transverse axis)")
    p.add_argument("--nbins", type=int, required=True, help="LOS nbins (pixels per skewer)")
    p.add_argument("--snap", type=int, default=25, help="Snapshot number (default: 25)")
    p.add_argument("--axis", type=int, default=-1, help="LOS axis (default: -1 -> concatenate x,y,z)")
    p.add_argument("--base-root", default="/home/STORAGE", help="Root where simulations live")
    p.add_argument("--out-root", default="/home/STORAGE/SKEWERS", help="Root where outputs are written")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing file")
    p.add_argument("--kernel", default="voronoi", help="Kernel (default: voronoi)")
    p.add_argument("--quiet", action="store_true", help="Less fake_spectra output")
    return p.parse_args()


def main():
    args = parse_args()

    base = os.path.join(args.base_root, args.sim)
    out_dir = os.path.join(args.out_root, args.sim, f"snapdir_{args.snap:03d}")

    lines = [
        ("H",  1, 1215),
        ("Si", 3, 1206),
        ("Si", 2, 1190),
        ("Si", 2, 1193),
    ]

    t0 = time.time()
    gs, info = run_gridded_tau(
        base=base,
        num=args.snap,
        nspec=args.nspec,
        axis=args.axis,
        nbins=args.nbins,
        lines=lines,
        out_dir=out_dir,
        overwrite=args.overwrite,
        MPI=None,
        kernel=args.kernel,
        quiet=args.quiet,
    )
    dt = time.time() - t0

    # Save a timing file next to the produced HDF5
    savepath = Path(info["savepath"])
    timing_path = savepath.with_suffix(savepath.suffix + ".timing.txt")

    timing_txt = (
        f"sim={args.sim}\n"
        f"snap={args.snap}\n"
        f"nspec={args.nspec}\n"
        f"nbins={args.nbins}\n"
        f"axis={args.axis}\n"
        f"kernel={args.kernel}\n"
        f"output_file={savepath}\n"
        f"elapsed_seconds={dt:.3f}\n"
        f"elapsed_minutes={dt/60.0:.3f}\n"
    )
    timing_path.write_text(timing_txt, encoding="utf-8")

    print(info)
    print(f"[timing] wrote: {timing_path}")
    print(f"[timing] elapsed: {dt/60.0:.2f} min")


if __name__ == "__main__":
    main()

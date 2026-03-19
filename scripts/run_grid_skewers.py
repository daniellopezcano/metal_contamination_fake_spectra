#!/usr/bin/env python3
import argparse
import os
import socket
import time
import resource
from pathlib import Path

from mcfs.compute_grid_fake_spectra import run_gridded_skewers

LINE_PRESETS = {
    "lya": [("H", 1, 1215)],
    "lya_si": [("H", 1, 1215), ("Si", 3, 1206), ("Si", 2, 1190), ("Si", 2, 1193)],
}

def get_mpi_or_none():
    try:
        from mpi4py import MPI  # type: ignore
        # If user launched with mpirun -n 1, size==1 is effectively non-parallel.
        if MPI.COMM_WORLD.Get_size() > 1:
            return MPI
    except Exception:
        pass
    return None

def resolve_base_for_fake_spectra(base: str, num: int, MPI):
    # fake_spectra path quirk: with MPI communicator it may expect base already == snapdir_###
    if MPI is None:
        return base
    p = Path(base)
    if p.name.startswith("snapdir_"):
        return base
    snapdir = p / f"snapdir_{num:03d}"
    return str(snapdir) if snapdir.exists() else base

def read_meminfo_bytes():
    mem_total = None
    mem_avail = None
    try:
        with open("/proc/meminfo", "r") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    mem_total = int(line.split()[1]) * 1024
                elif line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) * 1024
    except Exception:
        pass
    return mem_total, mem_avail

def rss_peak_mib():
    # On Linux, ru_maxrss is in kB
    try:
        kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(kb) / 1024.0
    except Exception:
        return None

def cpu_affinity_n():
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return None

def build_savefile(base, num, nspec, axis, nbins, preset, tag=None):
    sim_name = Path(base).name
    name = f"grid_{sim_name}_snap{num:03d}_nspec{nspec}_axis{axis}_nbins{nbins}_{preset}"
    if tag:
        name += f"_{tag}"
    return name + ".hdf5"

def parse_args():
    p = argparse.ArgumentParser("Run gridded fake_spectra skewers (MPI optional).")
    p.add_argument("--base", required=True)
    p.add_argument("--num", required=True, type=int)
    p.add_argument("--nspec", required=True, type=int)
    p.add_argument("--axis", required=True, type=int, choices=[1, 2, 3, -1])
    p.add_argument("--nbins", required=True, type=int)
    p.add_argument("--out-dir", required=True)

    p.add_argument("--preset", default="lya", choices=LINE_PRESETS.keys())
    p.add_argument("--savefile", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--kernel", default="tophat")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--force-recompute-tau", action="store_true")
    p.add_argument("--compute-density", action="store_true")
    p.add_argument("--compute-temperature", action="store_true")
    p.add_argument("--compute-velocity-los", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    lines = LINE_PRESETS[args.preset]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    savefile = args.savefile or build_savefile(
        base=args.base,
        num=args.num,
        nspec=args.nspec,
        axis=args.axis,
        nbins=args.nbins,
        preset=args.preset,
        tag=args.tag,
    )
    stem = Path(savefile).stem
    summary_file = out_dir / f"{stem}_summary.txt"

    MPI = get_mpi_or_none()
    if MPI is not None:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
    else:
        comm = None
        rank = 0
        size = 1

    base_fs = resolve_base_for_fake_spectra(args.base, args.num, MPI)

    # Collect static per-rank info
    host = socket.gethostname()
    aff_n = cpu_affinity_n()
    mem_total_b, mem_avail_b = read_meminfo_bytes()

    # Timers
    t0_wall = time.time()
    t0_cpu = time.process_time()

    gs, info = run_gridded_skewers(
        base=base_fs,
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
        MPI=MPI,
    )

    wall = time.time() - t0_wall
    cpu = time.process_time() - t0_cpu
    rss_mib = rss_peak_mib()

    local = {
        "rank": rank,
        "host": host,
        "wall_s": wall,
        "cpu_s": cpu,
        "aff_n": aff_n,
        "rss_mib": rss_mib,
        "mem_total_b": mem_total_b,
        "mem_avail_b": mem_avail_b,
    }

    if comm is not None:
        allr = comm.gather(local, root=0)
        comm.Barrier()
    else:
        allr = [local]

    if rank != 0:
        return

    # ---- Root writes one compact summary ----
    hosts = sorted({r["host"] for r in allr})
    wall_max = max(r["wall_s"] for r in allr)
    cpu_sum = sum(r["cpu_s"] for r in allr)
    cpu_over_wall = cpu_sum / max(wall_max, 1e-12)

    # Memory summaries (observed peak RSS, not requested)
    rss_vals = [r["rss_mib"] for r in allr if r["rss_mib"] is not None]
    rss_sum = sum(rss_vals) if rss_vals else None
    rss_max = max(rss_vals) if rss_vals else None

    # Try to infer "threads per rank" from env (best-effort)
    def env_int(name, default=1):
        v = os.environ.get(name)
        try:
            return int(v) if v is not None else default
        except Exception:
            return default

    omp = env_int("OMP_NUM_THREADS", 1)
    mkl = env_int("MKL_NUM_THREADS", 1)
    obl = env_int("OPENBLAS_NUM_THREADS", 1)

    # Approx “cores used” estimate: ranks * max(thread knobs)
    threads_est = max(omp, mkl, obl)
    cores_est = size * threads_est

    with open(summary_file, "w") as f:
        f.write("=== fake_spectra run summary ===\n")
        f.write(f"savepath: {info.get('savepath')}\n")
        f.write(f"manifest: {info.get('manifest')}\n")
        f.write(f"tau_keys: {info.get('tau_keys')}\n")
        f.write("\n")
        f.write(f"MPI ranks (processes): {size}\n")
        f.write(f"Nodes used: {len(hosts)}  ({', '.join(hosts)})\n")
        f.write(f"Thread env (per rank): OMP={omp}, MKL={mkl}, OPENBLAS={obl}\n")
        f.write(f"Estimated max threads per rank: {threads_est}\n")
        f.write(f"Estimated total compute 'cores' (ranks * threads): {cores_est}\n")
        f.write("\n")
        f.write(f"Wall time (max over ranks): {wall_max:.3f} s\n")
        f.write(f"CPU time (sum over ranks):  {cpu_sum:.3f} s\n")
        f.write(f"CPU_sum / Wall_max:         {cpu_over_wall:.3f}  (CPU-bound ideal ~ {size})\n")
        f.write("\n")

        # CPU affinity: what the OS/scheduler actually allows each rank to run on
        aff_list = [r["aff_n"] for r in allr]
        f.write("CPU affinity (allowed cores) per rank:\n")
        for r in sorted(allr, key=lambda x: x["rank"]):
            f.write(f"  rank {r['rank']:3d} @ {r['host']}: aff_n={r['aff_n']}\n")
        f.write("\n")

        # Memory (observed)
        f.write("Observed memory (peak RSS):\n")
        if rss_sum is not None:
            f.write(f"  sum over ranks: {rss_sum:.1f} MiB\n")
            f.write(f"  max rank:       {rss_max:.1f} MiB\n")
            f.write(f"  approx per rank: {(rss_sum/size):.1f} MiB\n")
        else:
            f.write("  (unavailable)\n")
        f.write("\n")

        # Node-level memory snapshot (best-effort; per-rank /proc/meminfo)
        f.write("Node memory snapshot (MemTotal / MemAvailable at start):\n")
        seen = set()
        for r in sorted(allr, key=lambda x: x["host"]):
            if r["host"] in seen:
                continue
            seen.add(r["host"])
            mt = r["mem_total_b"]
            ma = r["mem_avail_b"]
            if mt is None or ma is None:
                f.write(f"  {r['host']}: (unavailable)\n")
            else:
                f.write(f"  {r['host']}: total={mt/2**30:.2f} GiB, avail={ma/2**30:.2f} GiB\n")
        f.write("\n")

        # Per-rank timings
        f.write("Per-rank timing:\n")
        for r in sorted(allr, key=lambda x: x["rank"]):
            f.write(
                f"  rank {r['rank']:3d} @ {r['host']}: "
                f"wall={r['wall_s']:.3f}s, cpu={r['cpu_s']:.3f}s, rss_peak={r['rss_mib']}\n"
            )

    print(f"Wrote summary: {summary_file}")

if __name__ == "__main__":
    main()


# example call from terminal:
# mpiexec -n 2 python run_grid_skewers.py\
#  --base /data/desi/scratch/TNG/TNG50-4 \
#  --num 33 \
#  --nspec 4 \
#  --axis -1 \
#  --nbins 1024 \
#  --out-dir /data/desi/scratch/TNG/SKEWERS/TNG50-4/snapdir_033 \
#  --preset lya_si \
#  --overwrite \
#  --force-recompute-tau \
#  --kernel tophat \
#  --compute-density \
#  --compute-temperature \
#  --compute-velocity-los

from mcfs.compute_grid_fake_spectra import run_gridded_skewers


def main() -> None:
    lines = [
        ("H", 1, 1215),
        ("Si", 3, 1206),
        ("Si", 2, 1190),
        ("Si", 2, 1193),
    ]

    gs, info = run_gridded_skewers(
        base="/home/STORAGE/TNG50-4",
        num=33,
        nspec=2,
        axis=2,
        nbins=1024,
        lines=lines,
        out_dir="/home/STORAGE/SKEWERS/TNG50-4/snapdir_033",
        savefile="grid_TNG50-4_snap033_nspec2_axis2_nbins1024_test.hdf5",
        overwrite=True,
        force_recompute_tau=True,
        kernel="tophat",
        compute_density=True,
        compute_temperature=True,
        compute_velocity_los=True,
        quiet=False,
    )

    print("\nRun finished.")
    print("HDF5 file :", info["savepath"])
    print("Manifest  :", info["manifest"])
    print("Tau keys  :", info["tau_keys"])


if __name__ == "__main__":
    main()
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, NullFormatter
from itertools import combinations


# ============================================================
# Generic helpers
# ============================================================

def _interp_to_ref_grid(x_ref, x_in, y_in):
    """
    Interpolate y_in(x_in) onto x_ref.
    """
    x_ref = np.asarray(x_ref, dtype=float)
    x_in = np.asarray(x_in, dtype=float)
    y_in = np.asarray(y_in, dtype=float)

    idx = np.argsort(x_in)
    x_in = x_in[idx]
    y_in = y_in[idx]

    return np.interp(x_ref, x_in, y_in, left=np.nan, right=np.nan)


def _resolve_case_order(data_dict, case_order=None, fiducial_key=None):
    """
    Resolve and validate the run order.

    The first entry of `case_order` is always treated as the fiducial run.
    """
    available = list(data_dict.keys())

    if case_order is None:
        try:
            case_order = sorted(available)
        except TypeError:
            case_order = available
    else:
        case_order = list(case_order)
        missing = [k for k in case_order if k not in available]
        if missing:
            raise KeyError(f"Keys in `case_order` not found in data: {missing}")

    if fiducial_key is None:
        fiducial_key = case_order[0]
    elif fiducial_key != case_order[0]:
        raise ValueError(
            "`fiducial_key` must match the first element of `case_order`. "
            f"Got fiducial_key={fiducial_key!r}, case_order[0]={case_order[0]!r}."
        )

    return case_order, fiducial_key


def _build_case_colors(case_order, item_ids, cmap_names, *, cmin=0.30, cmax=1.0):
    """
    Build color dictionaries so that the first case in `case_order`
    receives the darkest color and later cases progressively lighter colors.
    """
    color_positions = np.linspace(cmax, cmin, len(case_order))

    item_to_cmap = {
        item_id: plt.get_cmap(cmap_names[i % len(cmap_names)])
        for i, item_id in enumerate(item_ids)
    }

    colors = {
        item_id: {
            key_case: item_to_cmap[item_id](color_positions[i_case])
            for i_case, key_case in enumerate(case_order)
        }
        for item_id in item_ids
    }

    return colors


def _subset_to_text(subset):
    """
    Convert a subset label tuple into a compact readable string.
    """
    subset = tuple(subset)
    if len(subset) == 1:
        return subset[0]
    return "·".join(subset)


def _canonical_sym_pair(A, B):
    """
    Canonical representative of the symmetric pair (A,B) ~ (B,A).
    """
    return min((A, B), (B, A))


def _build_unique_terms_by_case(avg_catalog_by_case):
    """
    From a dict of averaged catalogs keyed by case, keep only one
    representative of each symmetric pair (A,B) ~ (B,A).
    """
    unique_terms_by_case = {}
    for key_case, catalog in avg_catalog_by_case.items():
        unique_terms = {}
        for (A, B), entry in catalog.items():
            canon = _canonical_sym_pair(A, B)
            if canon not in unique_terms:
                unique_terms[canon] = entry
        unique_terms_by_case[key_case] = unique_terms
    return unique_terms_by_case


def _entry_get(entry, *keys, default=None, required=True):
    """
    Return the first matching key from `entry`.
    """
    for key in keys:
        if key in entry:
            return entry[key]
    if required:
        raise KeyError(f"None of the requested keys {keys} found in entry. Available: {list(entry.keys())}")
    return default


def _sort_xy(x, y, yerr=None):
    """
    Sort x and reorder y (and optional yerr) accordingly.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    idx = np.argsort(x)

    x = x[idx]
    y = y[idx]

    if yerr is None:
        return x, y, None

    yerr = np.asarray(yerr, dtype=float)[idx]
    return x, y, yerr


def _mask_finite_xy(x, y, yerr=None):
    """
    Keep only finite values.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if yerr is None:
        mask = np.isfinite(x) & np.isfinite(y)
        return x[mask], y[mask], None

    yerr = np.asarray(yerr, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y) & np.isfinite(yerr)
    return x[mask], y[mask], yerr[mask]


def _clean_number_formatter():
    """
    Compact formatter for linear / symlog axes.
    """
    return FuncFormatter(lambda val, pos: f"{val:g}")


def _apply_clean_symlog_y(ax, *, linthresh=0.1, labelsize=18):
    """
    Apply a clean symlog y-axis with simpler tick formatting.
    """
    ax.set_yscale("symlog", linthresh=linthresh)
    ax.yaxis.set_major_formatter(_clean_number_formatter())
    ax.yaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="y", labelsize=labelsize)


def _apply_clean_axis_formatting(ax, *, xscale="linear", yscale="linear",
                                 y_symlog_linthresh=0.1, x_symlog_linthresh=0.1,
                                 x_labelsize=18, y_labelsize=18):
    """
    Apply scales and clean tick formatting.
    """
    ax.set_xscale(xscale)
    ax.set_yscale(yscale)

    if yscale == "symlog":
        ax.set_yscale("symlog", linthresh=y_symlog_linthresh)
        ax.yaxis.set_major_formatter(_clean_number_formatter())
        ax.yaxis.set_minor_formatter(NullFormatter())

    if xscale == "symlog":
        ax.set_xscale("symlog", linthresh=x_symlog_linthresh)
        ax.xaxis.set_major_formatter(_clean_number_formatter())
        ax.xaxis.set_minor_formatter(NullFormatter())

    ax.tick_params(axis="x", labelsize=x_labelsize)
    ax.tick_params(axis="y", labelsize=y_labelsize)


def _plot_curve_with_band(ax, x, y, yerr=None, *, color, lw=1.5, alpha=0.85, band_alpha=0.12):
    """
    Robust line + fill_between plotting:
    - sorts x
    - masks NaNs / infs
    """
    x, y, yerr = _sort_xy(x, y, yerr=yerr)
    x, y, yerr = _mask_finite_xy(x, y, yerr=yerr)

    if x.size == 0:
        return

    ax.plot(x, y, lw=lw, color=color, alpha=alpha)

    if yerr is not None:
        ax.fill_between(x, y - yerr, y + yerr, color=color, alpha=band_alpha, linewidth=0.0)


# ============================================================
# Helpers for tau / flux / overflux figures
# ============================================================

def _make_tau_axes(figsize=(16, 18), height_ratios=(2, 1, 2, 1, 2, 1), hspace=0.06):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(6, 1, height_ratios=height_ratios, hspace=hspace)

    ax_tau      = fig.add_subplot(gs[0, 0])
    ax_tau_res  = fig.add_subplot(gs[1, 0], sharex=ax_tau)
    ax_flux     = fig.add_subplot(gs[2, 0], sharex=ax_tau)
    ax_flux_res = fig.add_subplot(gs[3, 0], sharex=ax_tau)
    ax_del      = fig.add_subplot(gs[4, 0], sharex=ax_tau)
    ax_del_res  = fig.add_subplot(gs[5, 0], sharex=ax_tau)

    return fig, {
        "tau": ax_tau,
        "tau_res": ax_tau_res,
        "flux": ax_flux,
        "flux_res": ax_flux_res,
        "delta": ax_del,
        "delta_res": ax_del_res,
    }


def _style_tau_axes(
    axd,
    xlabel,
    title,
    xlim,
    *,
    residual_band=0.1,
    residual_symlog_linthresh=0.1,
    main_labelsize=18,
    residual_labelsize=18,
    ylabel_fontsize=18,
):
    main_axes = [axd["tau"], axd["flux"], axd["delta"]]
    res_axes = [axd["tau_res"], axd["flux_res"], axd["delta_res"]]

    for ax in res_axes:
        ax.axhspan(-residual_band, residual_band, color="0.88", alpha=0.8, zorder=0)
        ax.axhline(0.0, color="k", lw=1.0, alpha=0.7, zorder=1)
        _apply_clean_symlog_y(ax, linthresh=residual_symlog_linthresh, labelsize=residual_labelsize)

    axd["delta"].axhline(0.0, color="k", lw=1.0, alpha=0.5)

    axd["tau"].set_ylabel(r"$\tau$", fontsize=ylabel_fontsize, labelpad=8)
    axd["tau"].set_yscale("log")
    axd["tau_res"].set_ylabel(r"$\Delta\tau$", fontsize=ylabel_fontsize - 1, labelpad=8)

    axd["flux"].set_ylabel(r"$e^{-\tau}$", fontsize=ylabel_fontsize, labelpad=8)
    axd["flux_res"].set_ylabel(r"$\Delta e^{-\tau}$", fontsize=ylabel_fontsize - 1, labelpad=8)

    axd["delta"].set_ylabel(r"$\delta$", fontsize=ylabel_fontsize, labelpad=8)
    axd["delta_res"].set_ylabel(r"$\Delta\delta$", fontsize=ylabel_fontsize - 1, labelpad=8)
    axd["delta_res"].set_xlabel(xlabel, fontsize=ylabel_fontsize)

    axd["tau"].set_title(title, fontsize=ylabel_fontsize + 1)

    for ax in res_axes:
        ax.tick_params(axis="both", labelsize=residual_labelsize)
    for ax in main_axes:
        ax.tick_params(axis="both", labelsize=main_labelsize)

    for ax in [axd["tau"], axd["tau_res"], axd["flux"], axd["flux_res"], axd["delta"]]:
        plt.setp(ax.get_xticklabels(), visible=False)

    if xlim is not None:
        axd["tau"].set_xlim(*xlim)


# ============================================================
# Helpers for Xi1D
# ===========================================================

def _compute_line_pair_markers(line_bundle, *, lag_max, positive_lags_only=True, c_kms=299792.458):
    """
    Build vertical-marker metadata for all unique unordered line pairs.

    For each unique pair {i, j}, the separation is defined as

        delta_v_ij = | c_kms * ln(lambda_i / lambda_j) |

    so that (i, j) and (j, i) are treated as the same pair and are not repeated.

    Position convention
    -------------------
    - If delta_v_ij <= lag_max, use x = delta_v_ij.
    - If delta_v_ij > lag_max, use the wrapped position
          x = lag_max - delta_v_ij
    - If `positive_lags_only=True`, any negative wrapped position is mapped back
      into the displayed [0, lag_max] interval so the marker remains visible.

    Marker style convention
    -----------------------
    - normal marker  -> dashed   ('--')
    - wrapped marker -> dotted   (':')

    The displayed text is always the original delta_v value, rounded to 2 decimals,
    even when the plotted position is wrapped.
    """
    if line_bundle is None:
        return []

    lag_max = float(lag_max)
    if lag_max <= 0:
        return []

    markers = []

    for line_i, line_j in combinations(line_bundle.lines, 2):
        delta_v = float(abs(c_kms * np.log(line_i.lambda0_A / line_j.lambda0_A)))

        wrapped = delta_v > lag_max
        x = lag_max - delta_v if wrapped else delta_v

        if positive_lags_only:
            x = x % lag_max
            if np.isclose(x, 0.0) and not np.isclose(delta_v, 0.0):
                x = lag_max

        markers.append(
            {
                "x": x,
                "delta_v": delta_v,
                "text": f"{delta_v:.2f}",
                "pair": (line_i.id, line_j.id),
                "wrapped": wrapped,
                "ls": ":" if wrapped else "--",
            }
        )

    markers.sort(key=lambda m: m["x"])
    return markers


def _draw_vertical_markers(
    ax,
    markers,
    *,
    xlim=None,
    color="k",
    lw=0.9,
    ls="--",
    alpha=0.35,
    text_fontsize=18,
    text_alpha=0.78,
    show_text=True,
):
    """
    Draw vertical markers and small rotated numeric labels on a given axis.

    Notes
    -----
    - If a marker provides its own linestyle in marker["ls"], that overrides `ls`.
    - The annotation text is taken from marker["text"] when present; otherwise
      it falls back to marker["delta_v"] formatted to 2 decimals.
    """
    if not markers:
        return

    if xlim is None:
        xmin, xmax = ax.get_xlim()
    else:
        xmin, xmax = xlim

    visible = [m for m in markers if np.isfinite(m["x"]) and xmin <= m["x"] <= xmax]
    if not visible:
        return

    visible = sorted(visible, key=lambda m: m["x"])

    y_levels = np.array([0.04, 0.18, 0.32, 0.46, 0.60, 0.74, 0.88])
    dx = 0.004 * (xmax - xmin) if np.isfinite(xmax - xmin) else 0.0

    for i, marker in enumerate(visible):
        x = marker["x"]
        marker_ls = marker.get("ls", ls)

        ax.axvline(x, color=color, lw=lw, ls=marker_ls, alpha=alpha, zorder=1)

        if show_text:
            y_frac = y_levels[i % len(y_levels)]
            x_text = min(
                max(x + dx, xmin + 0.002 * (xmax - xmin)),
                xmax - 0.002 * (xmax - xmin),
            )

            txt = marker.get("text", f"{marker.get('delta_v', np.nan):.2f}")

            ax.text(
                x_text,
                y_frac,
                txt,
                rotation=90,
                va="bottom",
                ha="left",
                transform=ax.get_xaxis_transform(),
                fontsize=text_fontsize,
                alpha=text_alpha,
                color=color,
                clip_on=True,
                bbox=dict(facecolor="white", edgecolor="none", alpha=0.55, pad=0.2),
            )

# ============================================================
# Helper for generic 1 main panel + 1 residual panel plots
# ============================================================

def _plot_main_and_residual(
    x_by_case,
    y_by_case,
    yerr_by_case,
    *,
    case_order,
    fiducial_key,
    colors_by_case,
    title,
    xlabel,
    ylabel,
    residual_ylabel,
    xscale="log",
    yscale="symlog",
    residual_xscale="log",
    residual_yscale="symlog",
    main_symlog_linthresh=0.05,
    residual_symlog_linthresh=0.05,
    residual_band=0.05,
    draw_main_zero=True,
    draw_residual_zero=True,
    figsize=(12, 8),
    main_lw=1.6,
    residual_lw=1.4,
    alpha=0.82,
    sem_alpha=0.12,
    legend_title="Runs",
    legend_loc="upper right",
    xlim=None,
    ylim=None,
    residual_ylim=None,
    residual_mode="difference",
    residual_relative_floor=None,
    vertical_markers=None,
    vertical_markers_on_residual=False,
    vertical_marker_color="k",
    vertical_marker_lw=0.9,
    vertical_marker_ls="--",
    vertical_marker_alpha=0.35,
    vertical_marker_text_fontsize=18,
    vertical_marker_text_alpha=0.78,
    show=True,
    tight_layout=True,
    main_labelsize=18,
    residual_labelsize=18,
    ylabel_fontsize=18,
):
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.05)

    ax_main = fig.add_subplot(gs[0, 0])
    ax_res = fig.add_subplot(gs[1, 0], sharex=ax_main)

    x_ref = np.asarray(x_by_case[fiducial_key], dtype=float)
    y_ref = np.asarray(y_by_case[fiducial_key], dtype=float)
    x_ref, y_ref, _ = _sort_xy(x_ref, y_ref)

    for key_case in case_order:
        x = np.asarray(x_by_case[key_case], dtype=float)
        y = np.asarray(y_by_case[key_case], dtype=float)
        yerr = None if yerr_by_case is None else np.asarray(yerr_by_case[key_case], dtype=float)
        color = colors_by_case[key_case]

        _plot_curve_with_band(
            ax_main, x, y, yerr=yerr,
            color=color, lw=main_lw, alpha=alpha, band_alpha=sem_alpha
        )

        if key_case != fiducial_key:
            x_sorted, y_sorted, _ = _sort_xy(x, y)
            y_interp = _interp_to_ref_grid(x_ref, x_sorted, y_sorted)

            if residual_mode == "difference":
                residual = y_interp - y_ref
            elif residual_mode == "relative":
                if residual_relative_floor is None:
                    denom = np.where(y_ref != 0.0, y_ref, np.nan)
                else:
                    denom = np.where(np.abs(y_ref) > residual_relative_floor, y_ref, np.nan)
                residual = (y_interp - y_ref) / denom
            else:
                raise ValueError(
                    f"Unknown residual_mode={residual_mode!r}. "
                    "Allowed: 'difference', 'relative'."
                )

            x_plot, y_plot, _ = _mask_finite_xy(x_ref, residual, None)

            if x_plot.size > 0:
                ax_res.plot(x_plot, y_plot, lw=residual_lw, color=color, alpha=alpha)

    if residual_band is not None:
        ax_res.axhspan(-residual_band, residual_band, color="0.88", alpha=0.8, zorder=0)

    if draw_main_zero:
        ax_main.axhline(0.0, color="k", lw=0.8, alpha=0.75, zorder=1)

    if draw_residual_zero:
        ax_res.axhline(0.0, color="k", lw=0.9, alpha=0.75, zorder=1)

    _apply_clean_axis_formatting(
        ax_main,
        xscale=xscale,
        yscale=yscale,
        y_symlog_linthresh=main_symlog_linthresh,
        x_labelsize=main_labelsize,
        y_labelsize=main_labelsize,
    )
    _apply_clean_axis_formatting(
        ax_res,
        xscale=residual_xscale,
        yscale=residual_yscale,
        y_symlog_linthresh=residual_symlog_linthresh,
        x_labelsize=residual_labelsize,
        y_labelsize=residual_labelsize,
    )

    ax_main.set_ylabel(ylabel, fontsize=ylabel_fontsize, labelpad=8)
    ax_res.set_ylabel(residual_ylabel, fontsize=ylabel_fontsize - 1, labelpad=8)
    ax_res.set_xlabel(xlabel, fontsize=ylabel_fontsize)
    ax_main.set_title(title, fontsize=ylabel_fontsize + 1)

    if xlim is not None:
        ax_main.set_xlim(*xlim)
        ax_res.set_xlim(*xlim)

    if ylim is not None:
        ax_main.set_ylim(*ylim)

    if residual_ylim is not None:
        ax_res.set_ylim(*residual_ylim)

    # Vertical line-pair markers
    if vertical_markers is not None:
        _draw_vertical_markers(
            ax_main,
            vertical_markers,
            xlim=xlim,
            color=vertical_marker_color,
            lw=vertical_marker_lw,
            ls=vertical_marker_ls,
            alpha=vertical_marker_alpha,
            text_fontsize=vertical_marker_text_fontsize,
            text_alpha=vertical_marker_text_alpha,
            show_text=True,
        )
        if vertical_markers_on_residual:
            _draw_vertical_markers(
                ax_res,
                vertical_markers,
                xlim=xlim,
                color=vertical_marker_color,
                lw=vertical_marker_lw,
                ls=vertical_marker_ls,
                alpha=vertical_marker_alpha,
                text_fontsize=vertical_marker_text_fontsize,
                text_alpha=vertical_marker_text_alpha,
                show_text=False,
            )

    plt.setp(ax_main.get_xticklabels(), visible=False)

    handles = [
        Line2D([0], [0], color=colors_by_case[key_case], lw=2.2, label=str(key_case))
        for key_case in case_order
    ]
    ax_main.legend(handles=handles, loc=legend_loc, fontsize=18, frameon=True, title=legend_title)

    if tight_layout:
        plt.tight_layout()
    if show:
        plt.show()

    return fig


# ============================================================
# Main plotting function: tau / flux / overflux
# ============================================================

def plot_tau_flux_overflux_resolution_comparison(
    tau_cube,
    tau_plot,
    overflux,
    overflux_total,
    v_kms,
    ii_list,
    *,
    case_order=None,
    fiducial_key=None,
    xlabel=r"$c\, \ln\left( \frac{\lambda}{\lambda_\dagger}\right)\; \left[\mathrm{km\,/\,s}\right]$",
    line_cmap_names=("Blues", "Greens", "Reds", "Purples", "Oranges", "PuBu", "YlGn", "YlOrBr", "PuRd", "Greys"),
    total_cmap_name="Greys",
    cmin=0.30,
    cmax=1.0,
    figsize=(16, 18),
    height_ratios=(2, 1, 2, 1, 2, 1),
    hspace=0.06,
    line_lw=1.0,
    residual_lw=1.0,
    alpha=0.92,
    line_ids=None,
    plot_total=True,
    residual_band=0.1,
    residual_symlog_linthresh=0.1,
    show=True,
    tight_layout=True,
):
    """
    Plot tau, transmitted flux, and overflux comparisons across multiple runs.
    """
    case_order, fiducial_key = _resolve_case_order(
        tau_cube, case_order=case_order, fiducial_key=fiducial_key
    )

    first_case = case_order[0]
    all_line_ids = tau_cube[first_case].line_ids

    if line_ids is None:
        line_ids = list(all_line_ids)
    else:
        line_ids = list(line_ids)
        missing = [lid for lid in line_ids if lid not in all_line_ids]
        if missing:
            raise KeyError(f"Requested line_ids not found in line bundle: {missing}")

    line_index = {lid: i for i, lid in enumerate(all_line_ids)}

    line_colors = _build_case_colors(
        case_order, line_ids, line_cmap_names, cmin=cmin, cmax=cmax
    )
    total_colors = _build_case_colors(
        case_order, ["total"], [total_cmap_name], cmin=cmin, cmax=cmax
    )["total"]

    figs = {}

    for ii in ii_list:
        v_fid = np.asarray(v_kms[fiducial_key], dtype=float)
        v_fid = np.sort(v_fid)
        xlim = (v_fid.min(), v_fid.max())

        tau_fid_case = tau_plot[fiducial_key]
        overflux_fid_case = overflux[fiducial_key]
        overflux_total_fid_case = overflux_total[fiducial_key]

        tau_tot_fid = np.sum(tau_fid_case[:, ii, :], axis=0)
        F_lines_fid = np.exp(-tau_fid_case[:, ii, :])
        F_tot_fid = np.prod(F_lines_fid, axis=0)
        delta_tot_fid = overflux_total_fid_case[ii]

        # Individual lines
        for line_id in line_ids:
            jj = line_index[line_id]
            fig, axd = _make_tau_axes(figsize=figsize, height_ratios=height_ratios, hspace=hspace)

            for key_case in case_order:
                v = np.asarray(v_kms[key_case], dtype=float)
                tau_case = tau_plot[key_case]
                overflux_case = overflux[key_case]
                F_lines = np.exp(-tau_case[:, ii, :])
                color = line_colors[line_id][key_case]

                _plot_curve_with_band(axd["tau"], v, tau_case[jj, ii], None, color=color, lw=line_lw, alpha=alpha)
                _plot_curve_with_band(axd["flux"], v, F_lines[jj], None, color=color, lw=line_lw, alpha=alpha)
                _plot_curve_with_band(axd["delta"], v, overflux_case[jj, ii], None, color=color, lw=line_lw, alpha=alpha)

                if key_case != fiducial_key:
                    tau_interp = _interp_to_ref_grid(v_fid, v, tau_case[jj, ii])
                    F_interp = _interp_to_ref_grid(v_fid, v, F_lines[jj])
                    delta_interp = _interp_to_ref_grid(v_fid, v, overflux_case[jj, ii])

                    x_plot, y_plot, _ = _mask_finite_xy(v_fid, tau_interp - tau_fid_case[jj, ii], None)
                    axd["tau_res"].plot(x_plot, y_plot, color=color, lw=residual_lw, ls="-", alpha=alpha)

                    x_plot, y_plot, _ = _mask_finite_xy(v_fid, F_interp - F_lines_fid[jj], None)
                    axd["flux_res"].plot(x_plot, y_plot, color=color, lw=residual_lw, ls="-", alpha=alpha)

                    x_plot, y_plot, _ = _mask_finite_xy(v_fid, delta_interp - overflux_fid_case[jj, ii], None)
                    axd["delta_res"].plot(x_plot, y_plot, color=color, lw=residual_lw, ls="-", alpha=alpha)

            _style_tau_axes(
                axd,
                xlabel=xlabel,
                title=f"Skewer index: {ii} — {line_id}",
                xlim=xlim,
                residual_band=residual_band,
                residual_symlog_linthresh=residual_symlog_linthresh,
            )

            case_handles = [
                Line2D([0], [0], color=line_colors[line_id][key_case], lw=2.2, ls="-", label=str(key_case))
                for key_case in case_order
            ]
            axd["tau"].legend(handles=case_handles, loc="upper left", frameon=True, fontsize=18, title="Runs")

            if tight_layout:
                plt.tight_layout()
            if show:
                plt.show()

            figs[(ii, line_id)] = fig

        # Total
        if plot_total:
            fig, axd = _make_tau_axes(figsize=figsize, height_ratios=height_ratios, hspace=hspace)

            for key_case in case_order:
                v = np.asarray(v_kms[key_case], dtype=float)
                delta_tot = overflux_total[key_case][ii]
                tau_case = tau_plot[key_case]
                tau_tot = np.sum(tau_case[:, ii, :], axis=0)
                F_tot = np.prod(np.exp(-tau_case[:, ii, :]), axis=0)
                color = total_colors[key_case]

                _plot_curve_with_band(axd["tau"], v, tau_tot, None, color=color, lw=line_lw, alpha=alpha)
                _plot_curve_with_band(axd["flux"], v, F_tot, None, color=color, lw=line_lw, alpha=alpha)
                _plot_curve_with_band(axd["delta"], v, delta_tot, None, color=color, lw=line_lw, alpha=alpha)

                if key_case != fiducial_key:
                    tau_tot_interp = _interp_to_ref_grid(v_fid, v, tau_tot)
                    F_tot_interp = _interp_to_ref_grid(v_fid, v, F_tot)
                    delta_tot_interp = _interp_to_ref_grid(v_fid, v, delta_tot)

                    x_plot, y_plot, _ = _mask_finite_xy(v_fid, tau_tot_interp - tau_tot_fid, None)
                    axd["tau_res"].plot(x_plot, y_plot, color=color, lw=residual_lw, ls="-", alpha=alpha)

                    x_plot, y_plot, _ = _mask_finite_xy(v_fid, F_tot_interp - F_tot_fid, None)
                    axd["flux_res"].plot(x_plot, y_plot, color=color, lw=residual_lw, ls="-", alpha=alpha)

                    x_plot, y_plot, _ = _mask_finite_xy(v_fid, delta_tot_interp - delta_tot_fid, None)
                    axd["delta_res"].plot(x_plot, y_plot, color=color, lw=residual_lw, ls="-", alpha=alpha)

            _style_tau_axes(
                axd,
                xlabel=xlabel,
                title=f"Skewer index: {ii} — Total",
                xlim=xlim,
                residual_band=residual_band,
                residual_symlog_linthresh=residual_symlog_linthresh,
            )

            case_handles = [
                Line2D([0], [0], color=total_colors[key_case], lw=2.2, ls="-", label=str(key_case))
                for key_case in case_order
            ]
            axd["tau"].legend(handles=case_handles, loc="upper left", frameon=True, fontsize=18, title="Runs")

            if tight_layout:
                plt.tight_layout()
            if show:
                plt.show()

            figs[(ii, "total")] = fig

    return figs


# ============================================================
# Main plotting function: P1D
# ============================================================

def plot_p1d_resolution_comparison(
    *,
    avg_P1D_catalog,
    omega,
    P1D_tot_mean,
    omega_tot,
    case_order=None,
    fiducial_key=None,
    term_cmap_names=("Blues", "Greens", "Reds", "Purples", "Oranges", "PuBu", "YlGn", "YlOrBr", "PuRd", "Greys"),
    total_cmap_name="Greys",
    cmin=0.30,
    cmax=1.0,
    figsize=(12, 8),
    main_lw=1.6,
    residual_lw=1.4,
    alpha=0.82,
    sem_alpha=0.12,
    xlabel=r"$k_\parallel \left[ \mathrm{km}^{-1} \mathrm{s} \right]$",
    ylabel=r"$P_{\mathrm{1D}}\,k_\parallel/\pi$",
    residual_ylabel=r"$\Delta$",
    xscale="linear",
    yscale="log",
    residual_xscale="linear",
    residual_yscale="linear",
    main_symlog_linthresh=0.05,
    residual_symlog_linthresh=0.05,
    residual_band=0.05,
    max_x=0.045,
    y_min=None,
    y_max=None,
    residual_ylim=None,
    residual_relative_floor=None,
    total_title=None,
    legend_loc="upper right",
    show=True,
    tight_layout=True,
):
    """
    Plot resolution comparisons for all canonical P1D terms and the total P1D.

    The plotted quantity is:
        y(k) = P1D(k) * k / pi

    The residual panel shows the relative difference with respect to the
    fiducial run:
        (y_case - y_ref) / y_ref
    """
    case_order, fiducial_key = _resolve_case_order(
        avg_P1D_catalog, case_order=case_order, fiducial_key=fiducial_key
    )

    unique_terms_by_case = _build_unique_terms_by_case(avg_P1D_catalog)

    sorted_items_fid = sorted(
        unique_terms_by_case[fiducial_key].items(),
        key=lambda item: (item[1]["total_order"], item[0][0], item[0][1])
    )
    canonical_terms = [term for term, _ in sorted_items_fid]

    term_colors = _build_case_colors(
        case_order, canonical_terms, term_cmap_names, cmin=cmin, cmax=cmax
    )
    total_colors = _build_case_colors(
        case_order, ["total"], [total_cmap_name], cmin=cmin, cmax=cmax
    )["total"]

    figs = {}
    xlim = (0.0, max_x) if max_x is not None else None
    ylim = None if (y_min is None and y_max is None) else (y_min, y_max)

    # One figure per canonical term
    for canon in canonical_terms:
        A, B = canon
        entry_fid = unique_terms_by_case[fiducial_key][canon]

        x_by_case = {}
        y_by_case = {}
        yerr_by_case = {}

        for key_case in case_order:
            if canon not in unique_terms_by_case[key_case]:
                continue

            entry = unique_terms_by_case[key_case][canon]

            k_par = np.asarray(omega[key_case], dtype=float)
            p1d_mean = np.real(np.asarray(_entry_get(entry, "P1D_mean", "p1d_mean")))
            p1d_sem = np.asarray(
                _entry_get(entry, "P1D_sem", "p1d_sem", default=np.zeros_like(p1d_mean), required=False)
            )

            x_by_case[key_case] = k_par
            y_by_case[key_case] = p1d_mean * k_par / np.pi
            yerr_by_case[key_case] = p1d_sem * k_par / np.pi

        case_order_term = [k for k in case_order if k in x_by_case]

        fig = _plot_main_and_residual(
            x_by_case=x_by_case,
            y_by_case=y_by_case,
            yerr_by_case=yerr_by_case,
            case_order=case_order_term,
            fiducial_key=fiducial_key,
            colors_by_case={k: term_colors[canon][k] for k in case_order_term},
            title=f"P1D[{_subset_to_text(A)}|{_subset_to_text(B)}]  " + rf"($\mathcal{{O}}({entry_fid['total_order']})$)",
            xlabel=xlabel,
            ylabel=ylabel,
            residual_ylabel=residual_ylabel,
            xscale=xscale,
            yscale=yscale,
            residual_xscale=residual_xscale,
            residual_yscale=residual_yscale,
            main_symlog_linthresh=main_symlog_linthresh,
            residual_symlog_linthresh=residual_symlog_linthresh,
            residual_band=residual_band,
            draw_main_zero=False,
            draw_residual_zero=True,
            figsize=figsize,
            main_lw=main_lw,
            residual_lw=residual_lw,
            alpha=alpha,
            sem_alpha=sem_alpha,
            legend_title="Runs",
            legend_loc=legend_loc,
            xlim=xlim,
            ylim=ylim,
            residual_ylim=residual_ylim,
            residual_mode="relative",
            residual_relative_floor=residual_relative_floor,
            show=show,
            tight_layout=tight_layout,
        )
        figs[canon] = fig

    # Total figure
    x_by_case = {}
    y_by_case = {}

    for k in case_order:
        k_par = np.asarray(omega_tot[k], dtype=float)
        x_by_case[k] = k_par
        y_by_case[k] = np.real(np.asarray(P1D_tot_mean[k])) * k_par / np.pi

    fig = _plot_main_and_residual(
        x_by_case=x_by_case,
        y_by_case=y_by_case,
        yerr_by_case=None,
        case_order=case_order,
        fiducial_key=fiducial_key,
        colors_by_case=total_colors,
        title=total_title,
        xlabel=xlabel,
        ylabel=ylabel,
        residual_ylabel=residual_ylabel,
        xscale=xscale,
        yscale=yscale,
        residual_xscale=residual_xscale,
        residual_yscale=residual_yscale,
        main_symlog_linthresh=main_symlog_linthresh,
        residual_symlog_linthresh=residual_symlog_linthresh,
        residual_band=residual_band,
        draw_main_zero=False,
        draw_residual_zero=True,
        figsize=figsize,
        main_lw=main_lw + 0.1,
        residual_lw=residual_lw + 0.1,
        alpha=alpha,
        sem_alpha=sem_alpha,
        legend_title="Runs",
        legend_loc=legend_loc,
        xlim=xlim,
        ylim=ylim,
        residual_ylim=residual_ylim,
        residual_mode="relative",
        residual_relative_floor=residual_relative_floor,
        show=show,
        tight_layout=tight_layout,
    )
    figs["total"] = fig

    return figs

# ============================================================
# Main plotting function: Xi1D
# ============================================================

def plot_xi1d_resolution_comparison(
    *,
    avg_Xi1D_catalog,
    lags,
    Xi1D_tot_mean,
    lags_tot,
    line_bundle=None,
    case_order=None,
    fiducial_key=None,
    term_cmap_names=("Blues", "Greens", "Reds", "Purples", "Oranges", "PuBu", "YlGn", "YlOrBr", "PuRd", "Greys"),
    total_cmap_name="Greys",
    cmin=0.30,
    cmax=1.0,
    figsize=(12, 8),
    main_lw=1.6,
    residual_lw=1.4,
    alpha=0.82,
    sem_alpha=0.12,
    xlabel=r"$\Delta \left[\mathrm{km\,s^{-1}}\right]$",
    ylabel=r"$\xi_{\mathrm{1D}}$",
    residual_ylabel=r"$\Delta$",
    xscale="linear",
    yscale="linear",
    residual_xscale="linear",
    residual_yscale="linear",
    main_symlog_linthresh=0.05,
    residual_symlog_linthresh=0.05,
    residual_band=0.05,
    max_x=None,
    y_min=None,
    y_max=None,
    residual_ylim=None,
    residual_relative_floor=1e-14,
    total_title=None,
    legend_loc="upper right",
    positive_lags_only=True,
    show_line_pair_markers=True,
    vertical_markers_on_residual=False,
    vertical_marker_color="k",
    vertical_marker_lw=0.9,
    vertical_marker_ls="--",
    vertical_marker_alpha=0.35,
    vertical_marker_text_fontsize=18,
    vertical_marker_text_alpha=0.78,
    show=True,
    tight_layout=True,
):
    """
    Plot resolution comparisons for all canonical Xi1D terms and the total Xi1D.

    Residual panel convention
    -------------------------
    The lower panel shows the relative difference with respect to the fiducial run:
        (y_case - y_ref) / y_ref

    Axis control
    ------------
    - If positive_lags_only=True:
          xlim = (0, max_x_use)
      where max_x_use is either `max_x` or the full positive lag extent.
    - If positive_lags_only=False and max_x is not None:
          xlim = (-max_x, max_x)
    """
    case_order, fiducial_key = _resolve_case_order(
        avg_Xi1D_catalog, case_order=case_order, fiducial_key=fiducial_key
    )

    unique_terms_by_case = _build_unique_terms_by_case(avg_Xi1D_catalog)

    sorted_items_fid = sorted(
        unique_terms_by_case[fiducial_key].items(),
        key=lambda item: (item[1]["total_order"], item[0][0], item[0][1])
    )
    canonical_terms = [term for term, _ in sorted_items_fid]

    term_colors = _build_case_colors(
        case_order, canonical_terms, term_cmap_names, cmin=cmin, cmax=cmax
    )
    total_colors = _build_case_colors(
        case_order, ["total"], [total_cmap_name], cmin=cmin, cmax=cmax
    )["total"]

    ylim = None if (y_min is None and y_max is None) else (y_min, y_max)

    figs = {}

    # One figure per canonical term
    for canon in canonical_terms:
        A, B = canon
        entry_fid = unique_terms_by_case[fiducial_key][canon]
        lags_fid = np.sort(np.asarray(lags[fiducial_key], dtype=float))

        if positive_lags_only:
            lag_max_plot = float(np.max(lags_fid)) if max_x is None else min(float(max_x), float(np.max(lags_fid)))
            xlim = (0.0, lag_max_plot)
        else:
            xlim = None if max_x is None else (-float(max_x), float(max_x))
            lag_max_plot = float(np.max(np.abs(lags_fid))) if max_x is None else float(max_x)

        x_by_case = {}
        y_by_case = {}
        yerr_by_case = {}

        for key_case in case_order:
            if canon not in unique_terms_by_case[key_case]:
                continue

            entry = unique_terms_by_case[key_case][canon]
            x_by_case[key_case] = np.asarray(lags[key_case], dtype=float)
            y_by_case[key_case] = np.real(np.asarray(_entry_get(entry, "Xi1D_mean", "xi1d_mean")))
            yerr_by_case[key_case] = np.asarray(
                _entry_get(
                    entry,
                    "Xi1D_sem",
                    "xi1d_sem",
                    default=np.zeros_like(y_by_case[key_case]),
                    required=False,
                )
            )

        case_order_term = [k for k in case_order if k in x_by_case]

        vertical_markers = None
        if show_line_pair_markers and line_bundle is not None:
            vertical_markers = _compute_line_pair_markers(
                line_bundle=line_bundle,
                lag_max=lag_max_plot,
                positive_lags_only=positive_lags_only,
            )

        fig = _plot_main_and_residual(
            x_by_case=x_by_case,
            y_by_case=y_by_case,
            yerr_by_case=yerr_by_case,
            case_order=case_order_term,
            fiducial_key=fiducial_key,
            colors_by_case={k: term_colors[canon][k] for k in case_order_term},
            title=f"ξ[{_subset_to_text(A)}|{_subset_to_text(B)}]  " + rf"($\mathcal{{O}}({entry_fid['total_order']})$)",
            xlabel=xlabel,
            ylabel=ylabel,
            residual_ylabel=residual_ylabel,
            xscale=xscale,
            yscale=yscale,
            residual_xscale=residual_xscale,
            residual_yscale=residual_yscale,
            main_symlog_linthresh=main_symlog_linthresh,
            residual_symlog_linthresh=residual_symlog_linthresh,
            residual_band=residual_band,
            draw_main_zero=True,
            draw_residual_zero=True,
            figsize=figsize,
            main_lw=main_lw,
            residual_lw=residual_lw,
            alpha=alpha,
            sem_alpha=sem_alpha,
            legend_title="Runs",
            legend_loc=legend_loc,
            xlim=xlim,
            ylim=ylim,
            residual_ylim=residual_ylim,
            residual_mode="relative",
            residual_relative_floor=residual_relative_floor,
            vertical_markers=vertical_markers,
            vertical_markers_on_residual=vertical_markers_on_residual,
            vertical_marker_color=vertical_marker_color,
            vertical_marker_lw=vertical_marker_lw,
            vertical_marker_ls=vertical_marker_ls,
            vertical_marker_alpha=vertical_marker_alpha,
            vertical_marker_text_fontsize=vertical_marker_text_fontsize,
            vertical_marker_text_alpha=vertical_marker_text_alpha,
            show=show,
            tight_layout=tight_layout,
        )
        figs[canon] = fig

    # Total figure
    lags_tot_fid = np.sort(np.asarray(lags_tot[fiducial_key], dtype=float))

    if positive_lags_only:
        lag_max_plot = float(np.max(lags_tot_fid)) if max_x is None else min(float(max_x), float(np.max(lags_tot_fid)))
        xlim = (0.0, lag_max_plot)
    else:
        xlim = None if max_x is None else (-float(max_x), float(max_x))
        lag_max_plot = float(np.max(np.abs(lags_tot_fid))) if max_x is None else float(max_x)

    x_by_case = {k: np.asarray(lags_tot[k], dtype=float) for k in case_order}
    y_by_case = {k: np.real(np.asarray(Xi1D_tot_mean[k])) for k in case_order}

    vertical_markers = None
    if show_line_pair_markers and line_bundle is not None:
        vertical_markers = _compute_line_pair_markers(
            line_bundle=line_bundle,
            lag_max=lag_max_plot,
            positive_lags_only=positive_lags_only,
        )

    fig = _plot_main_and_residual(
        x_by_case=x_by_case,
        y_by_case=y_by_case,
        yerr_by_case=None,
        case_order=case_order,
        fiducial_key=fiducial_key,
        colors_by_case=total_colors,
        title=total_title,
        xlabel=xlabel,
        ylabel=ylabel,
        residual_ylabel=residual_ylabel,
        xscale=xscale,
        yscale=yscale,
        residual_xscale=residual_xscale,
        residual_yscale=residual_yscale,
        main_symlog_linthresh=main_symlog_linthresh,
        residual_symlog_linthresh=residual_symlog_linthresh,
        residual_band=residual_band,
        draw_main_zero=True,
        draw_residual_zero=True,
        figsize=figsize,
        main_lw=main_lw + 0.1,
        residual_lw=residual_lw + 0.1,
        alpha=alpha,
        sem_alpha=sem_alpha,
        legend_title="Runs",
        legend_loc=legend_loc,
        xlim=xlim,
        ylim=ylim,
        residual_ylim=residual_ylim,
        residual_mode="relative",
        residual_relative_floor=residual_relative_floor,
        vertical_markers=vertical_markers,
        vertical_markers_on_residual=vertical_markers_on_residual,
        vertical_marker_color=vertical_marker_color,
        vertical_marker_lw=vertical_marker_lw,
        vertical_marker_ls=vertical_marker_ls,
        vertical_marker_alpha=vertical_marker_alpha,
        vertical_marker_text_fontsize=vertical_marker_text_fontsize,
        vertical_marker_text_alpha=vertical_marker_text_alpha,
        show=show,
        tight_layout=tight_layout,
    )
    figs["total"] = fig

    return figs
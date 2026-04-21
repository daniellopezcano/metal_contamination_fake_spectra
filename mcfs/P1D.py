import numpy as np
from scipy.fft import rfft, rfftfreq
from typing import TYPE_CHECKING, List, Sequence, Tuple

if TYPE_CHECKING:
    from .optical_depth import OpticalDepthCube


def subset_label(subset):
    """
    Human-readable label for a subset tuple.
    """
    subset = tuple(subset)
    if len(subset) == 1:
        return str(subset[0])
    return "(" + ",".join(map(str, subset)) + ")"


class P1DAnalyzer:
    """
    FFT-based helper for computing 1D auto/cross power spectra from
    overflux fields associated with an OpticalDepthCube workflow.
    """

    def __init__(self, lambda_or_v, optical_depth_cube: "OpticalDepthCube" = None):
        lambda_or_v = np.asarray(lambda_or_v, dtype=float)

        if lambda_or_v.ndim != 1:
            raise ValueError("`lambda_or_v` must be a 1D array.")
        if lambda_or_v.size < 2:
            raise ValueError("Need at least two spectral pixels.")

        dLambda_arr = np.diff(lambda_or_v)
        dLambda = dLambda_arr[0]
        if not np.allclose(dLambda_arr, dLambda, rtol=1e-8, atol=1e-12):
            raise ValueError("The spectral coordinate must be uniformly sampled.")

        self.lambda_or_v = lambda_or_v
        self.dLambda = float(dLambda)
        self.n_pix = lambda_or_v.size
        self.optical_depth_cube = optical_depth_cube
        self.line_bundle = None if optical_depth_cube is None else optical_depth_cube.line_bundle

    # ────────── Internal helpers ──────────
    @staticmethod
    def _normalize_subset(subset):
        """
        Ensure subset labels are stored as tuples.
        """
        if isinstance(subset, tuple):
            return subset
        if isinstance(subset, list):
            return tuple(subset)
        return (subset,)

    def _validate_field(self, field, axis=-1, name="field"):
        """
        Validate an input field against the stored spectral coordinate.
        """
        field = np.asarray(field, dtype=float)
        axis = axis % field.ndim

        if field.shape[axis] != self.n_pix:
            raise ValueError(
                f"The spectral axis of `{name}` has length {field.shape[axis]}, "
                f"but len(lambda_or_v) = {self.n_pix}."
            )
        return field, axis

    # ────────── Core FFT P1D ──────────
    def compute_cross_rfft(
        self,
        delta_A,
        delta_B,
        axis=-1,
        subtract_mean=False,
        drop_zero_mode=False,
    ):
        """
        Batched FFT-based 1D cross-power spectrum.

        Parameters
        ----------
        delta_A, delta_B : ndarray
            Arrays containing the two input fields. Typical shape:
            (N_skewers, N_pix), with the spectral axis specified by `axis`.
        axis : int, optional
            Spectral axis along which to compute the rFFT.
        subtract_mean : bool, optional
            If True, subtract the mean along the spectral axis before the FFT.
        drop_zero_mode : bool, optional
            If True, remove the zero-frequency mode from the output.

        Returns
        -------
        omega : ndarray, shape (N_freq,)
            Non-negative angular frequencies.
        P1D : ndarray
            Cross-power array with the same batch dimensions as the input and
            frequency dimension replacing the spectral one.
        """
        delta_A, axis = self._validate_field(delta_A, axis=axis, name="delta_A")
        delta_B, axis_B = self._validate_field(delta_B, axis=axis, name="delta_B")

        if axis_B != axis:
            raise ValueError("`delta_A` and `delta_B` must use the same spectral axis.")
        if delta_A.shape != delta_B.shape:
            raise ValueError("`delta_A` and `delta_B` must have the same shape.")

        x = delta_A.copy()
        y = delta_B.copy()

        if subtract_mean:
            x = x - np.mean(x, axis=axis, keepdims=True)
            y = y - np.mean(y, axis=axis, keepdims=True)

        fft_x = rfft(x, axis=axis)
        fft_y = rfft(y, axis=axis)

        omega = 2.0 * np.pi * rfftfreq(self.n_pix, d=self.dLambda)
        P1D = (self.dLambda / self.n_pix) * fft_x * np.conjugate(fft_y)

        if drop_zero_mode:
            slicer = [slice(None)] * P1D.ndim
            slicer[axis] = slice(1, None)
            P1D = P1D[tuple(slicer)]
            omega = omega[1:]

        return omega, P1D

    def compute_total_P1D(
        self,
        overflux_total,
        axis=-1,
        subtract_mean=False,
        drop_zero_mode=True,
    ):
        """
        Compute the total auto-P1D of the total overflux field.
        """
        return self.compute_cross_rfft(
            delta_A=overflux_total,
            delta_B=overflux_total,
            axis=axis,
            subtract_mean=subtract_mean,
            drop_zero_mode=drop_zero_mode,
        )

    def compute_subset_P1D_catalog(
        self,
        subset_fields,
        subset_labels,
        axis=-1,
        subtract_mean=False,
        drop_zero_mode=True,
    ):
        """
        Compute all P_AB terms for all skewers at once from subset-product fields.

        Parameters
        ----------
        subset_fields : ndarray, shape (N_subsets, N_skewers, N_pix)
            Output array from `OpticalDepthCube.build_subset_fields`.
        subset_labels : sequence
            Labels associated with `subset_fields`.
            Typical output from `OpticalDepthCube.build_subset_fields`,
            e.g. [('HI-LyA',), ('HI-LyB',), ('HI-LyA','HI-LyB'), ...]
        axis : int, optional
            Spectral axis inside each subset field. For the standard convention
            subset_fields.shape == (N_subsets, N_skewers, N_pix), use axis=-1.
        subtract_mean : bool, optional
            Whether to subtract the mean along the spectral axis.
        drop_zero_mode : bool, optional
            Whether to discard the zero mode.

        Returns
        -------
        omega : ndarray
            Frequency grid.
        catalog : dict
            catalog[(A, B)] contains:
                - 'omega'
                - 'P1D'
                - 'order_left'
                - 'order_right'
                - 'total_order'
                - 'label'
        by_total_order : dict
            Maps total order to the list of subset pairs (A, B).
        """
        subset_fields = np.asarray(subset_fields, dtype=float)

        if subset_fields.ndim != 3:
            raise ValueError(
                "`subset_fields` must have shape (N_subsets, N_skewers, N_pix)."
            )

        n_subsets = subset_fields.shape[0]

        subset_labels = [self._normalize_subset(lbl) for lbl in subset_labels]
        if len(subset_labels) != n_subsets:
            raise ValueError(
                f"`subset_labels` must have length {n_subsets}, got {len(subset_labels)}."
            )

        catalog = {}
        by_total_order = {}
        omega_out = None

        for iA, A in enumerate(subset_labels):
            for iB, B in enumerate(subset_labels):
                omega, P1D = self.compute_cross_rfft(
                    delta_A=subset_fields[iA],
                    delta_B=subset_fields[iB],
                    axis=axis,
                    subtract_mean=subtract_mean,
                    drop_zero_mode=drop_zero_mode,
                )

                total_order = len(A) + len(B)

                catalog[(A, B)] = {
                    "omega": omega,
                    "P1D": P1D,   # shape: (N_skewers, N_freq)
                    "order_left": len(A),
                    "order_right": len(B),
                    "total_order": total_order,
                    "label": f"P_{{{subset_label(A)}\\,{subset_label(B)}}}",
                }

                by_total_order.setdefault(total_order, []).append((A, B))
                omega_out = omega

        return omega_out, catalog, by_total_order

    def compute_average_P1D_catalog(
        self,
        P1D_catalog,
        average_axis=0,
        return_std=False,
        return_sem=False,
        symmetrize=False,
    ):
        """
        Average the batched P_AB terms over the skewer axis.

        Parameters
        ----------
        P1D_catalog : dict
            Output catalog from `compute_subset_P1D_catalog`.
            Each entry must contain:
                P1D_catalog[(A, B)]["P1D"] with shape (N_skewers, N_freq)
        average_axis : int, optional
            Axis over which to average. For the current convention
            P1D.shape == (N_skewers, N_freq), use average_axis=0.
        return_std : bool, optional
            If True, also return the standard deviation across skewers.
        return_sem : bool, optional
            If True, also return the standard error of the mean across skewers.
        symmetrize : bool, optional
            If True, replace each averaged term by the symmetrized combination
                P_AB,sym = P_AB + P_BA = 2 Re[P_AB]
            whenever both (A,B) and (B,A) exist in the catalog.

        Returns
        -------
        omega : ndarray
            Frequency grid associated with the catalog.
        avg_catalog : dict
            Dictionary with the same keys (A, B), each containing:
                - 'P1D'
                - optionally 'P1D_std'
                - optionally 'P1D_sem'
                - 'order_left'
                - 'order_right'
                - 'total_order'
                - 'label'
        """
        avg_catalog = {}
        omega_out = None

        for (A, B), entry in P1D_catalog.items():
            P1D = np.asarray(entry["P1D"])

            if symmetrize:
                if (B, A) not in P1D_catalog:
                    raise KeyError(f"Missing reverse pair {(B, A)} needed for symmetrization.")
                P1D_ba = np.asarray(P1D_catalog[(B, A)]["P1D"])
                values = P1D + P1D_ba
                label = f"P_{{{subset_label(A)}\\,{subset_label(B)}}}^{{sym}}"
            else:
                values = P1D
                label = entry["label"]

            P1D_mean = np.mean(values, axis=average_axis)

            out = {
                "P1D_mean": P1D_mean,
                "order_left": entry["order_left"],
                "order_right": entry["order_right"],
                "total_order": entry["total_order"],
                "label": label,
            }

            if return_std or return_sem:
                P1D_std = np.std(values, axis=average_axis, ddof=1)
                if return_std:
                    out["P1D_std"] = P1D_std
                if return_sem:
                    n_samples = values.shape[average_axis]
                    out["P1D_sem"] = P1D_std / np.sqrt(n_samples)

            avg_catalog[(A, B)] = out
            omega_out = entry["omega"]

        return omega_out, avg_catalog
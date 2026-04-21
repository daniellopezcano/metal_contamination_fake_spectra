import numpy as np
from scipy.fft import irfft
from typing import TYPE_CHECKING

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


class Xi1DAnalyzer:
    """
    FFT-based helper for converting P1D terms into periodic Xi1D terms.
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
        self.n_pix = lambda_or_v.size
        self.dLambda = float(dLambda)
        self.optical_depth_cube = optical_depth_cube
        self.line_bundle = None if optical_depth_cube is None else optical_depth_cube.line_bundle

    # ────────── Internal helpers ──────────
    @staticmethod
    def _insert_zero_mode(arr, axis=-1):
        """
        Insert a zero-valued frequency bin at index 0 along `axis`.
        """
        arr = np.asarray(arr)
        axis = axis % arr.ndim

        pad_shape = list(arr.shape)
        pad_shape[axis] = 1
        zero = np.zeros(pad_shape, dtype=arr.dtype)

        return np.concatenate([zero, arr], axis=axis)

    def _build_lag_grid(self, center_lags=False):
        """
        Build the periodic discrete lag grid associated with the original
        uniformly sampled spectral coordinate.

        Parameters
        ----------
        center_lags : bool, optional
            If True, return the lag grid in centered order
            (negative lags first, then zero, then positive lags).
            If False, return it in raw irfft order.

        Returns
        -------
        lags : ndarray, shape (N_pix,)
            Discrete lag grid.
        """
        k = np.arange(self.n_pix)

        # Raw irfft order:
        # even N: [0, 1, ..., N/2-1, -N/2, ..., -1]
        # odd  N: [0, 1, ..., (N-1)/2, -(N-1)/2, ..., -1]
        lags = np.where(k < (self.n_pix + 1) // 2, k, k - self.n_pix).astype(float) * self.dLambda

        if center_lags:
            lags = np.fft.fftshift(lags)

        return lags

    # ────────── Core Xi1D from P1D ──────────
    def compute_cross_from_P1D_irfft(
        self,
        P1D,
        axis=-1,
        P1D_zero_mode_dropped=True,
        center_lags=False,
    ):
        """
        Convert a cross-power array P_AB into its periodic 1D cross-correlation
        xi_AB by inverse real FFT, consistently with `P1DAnalyzer.compute_cross_rfft`.

        Important convention
        --------------------
        With the current P1D definition
            P_AB ~ rfft(A) * conj(rfft(B)),
        the periodic correlation with lag convention
            xi_AB(Delta) ~ < A(Lambda) B(Lambda + Delta) >
        is recovered as
            xi_AB = irfft(conj(P_AB_full)) / dLambda.

        Parameters
        ----------
        P1D : ndarray
            Cross-power array. Typical shape is (N_skewers, N_freq), with the
            frequency axis specified by `axis`.
        axis : int, optional
            Frequency axis in `P1D`.
        P1D_zero_mode_dropped : bool, optional
            If True, assume the input `P1D` does not contain the zero mode,
            and reinsert a zero before applying `irfft`.
        center_lags : bool, optional
            If True, return the correlation array in centered lag order.

        Returns
        -------
        lags : ndarray, shape (N_pix,)
            Periodic lag grid associated with the inverse transform.
        Xi1D : ndarray
            Periodic cross-correlation array with the same batch dimensions as
            `P1D`, and lag dimension replacing the frequency one.
        """
        P1D = np.asarray(P1D)
        axis = axis % P1D.ndim

        lags = self._build_lag_grid(center_lags=False)

        expected_nfreq_full = self.n_pix // 2 + 1
        expected_nfreq_input = expected_nfreq_full - 1 if P1D_zero_mode_dropped else expected_nfreq_full

        if P1D.shape[axis] != expected_nfreq_input:
            raise ValueError(
                f"Incompatible frequency-axis length. Got {P1D.shape[axis]}, "
                f"expected {expected_nfreq_input} for N_pix={self.n_pix}."
            )

        if P1D_zero_mode_dropped:
            P1D_full = self._insert_zero_mode(P1D, axis=axis)
        else:
            P1D_full = P1D

        Xi1D = irfft(np.conjugate(P1D_full), n=self.n_pix, axis=axis) / self.dLambda

        if center_lags:
            Xi1D = np.fft.fftshift(Xi1D, axes=axis)
            lags = np.fft.fftshift(lags)

        return lags, Xi1D

    def compute_total_Xi1D_from_P1D(
        self,
        P1D_total,
        axis=-1,
        P1D_zero_mode_dropped=True,
        center_lags=False,
    ):
        """
        Convenience wrapper for the total Xi1D from the total P1D field.
        """
        return self.compute_cross_from_P1D_irfft(
            P1D=P1D_total,
            axis=axis,
            P1D_zero_mode_dropped=P1D_zero_mode_dropped,
            center_lags=center_lags,
        )

    def compute_subset_Xi1D_catalog_from_P1D(
        self,
        P1D_catalog,
        axis=-1,
        P1D_zero_mode_dropped=True,
        center_lags=False,
    ):
        """
        Convert all P_AB terms stored in a P1D catalog into their periodic
        xi_AB counterparts by inverse real FFT.

        Parameters
        ----------
        P1D_catalog : dict
            Output catalog from `P1DAnalyzer.compute_subset_P1D_catalog`.
            Each entry must contain:
                P1D_catalog[(A, B)]["P1D"] with shape like (N_skewers, N_freq)
        axis : int, optional
            Frequency axis in the stored P1D arrays.
        P1D_zero_mode_dropped : bool, optional
            Whether the stored P1D arrays were computed with `drop_zero_mode=True`.
        center_lags : bool, optional
            If True, return all xi arrays in centered lag order.

        Returns
        -------
        lags : ndarray, shape (N_pix,)
            Periodic lag grid.
        Xi1D_catalog : dict
            Xi1D_catalog[(A, B)] contains:
                - 'lags'
                - 'Xi1D'
                - 'order_left'
                - 'order_right'
                - 'total_order'
                - 'label'
        by_total_order : dict
            Maps total order to the list of subset pairs (A, B).
        """
        Xi1D_catalog = {}
        by_total_order = {}
        lags_out = None

        for (A, B), entry in P1D_catalog.items():
            lags, Xi1D = self.compute_cross_from_P1D_irfft(
                P1D=entry["P1D"],
                axis=axis,
                P1D_zero_mode_dropped=P1D_zero_mode_dropped,
                center_lags=center_lags,
            )

            total_order = entry["total_order"]

            Xi1D_catalog[(A, B)] = {
                "lags": lags,
                "Xi1D": Xi1D,  # shape: (N_skewers, N_lags)
                "order_left": entry["order_left"],
                "order_right": entry["order_right"],
                "total_order": total_order,
                "label": f"\\xi_{{{subset_label(A)}\\,{subset_label(B)}}}",
            }

            by_total_order.setdefault(total_order, []).append((A, B))
            lags_out = lags

        return lags_out, Xi1D_catalog, by_total_order

    def compute_average_Xi1D_catalog(
        self,
        Xi1D_catalog,
        average_axis=0,
        return_std=False,
        return_sem=False,
        symmetrize=False,
    ):
        """
        Average the batched xi_AB terms over the skewer axis.

        Parameters
        ----------
        Xi1D_catalog : dict
            Output catalog from `compute_subset_Xi1D_catalog_from_P1D`.
            Each entry must contain:
                Xi1D_catalog[(A, B)]["Xi1D"] with shape (N_skewers, N_lags)
        average_axis : int, optional
            Axis over which to average. For the current convention
            Xi1D.shape == (N_skewers, N_lags), use average_axis=0.
        return_std : bool, optional
            If True, also return the standard deviation across skewers.
        return_sem : bool, optional
            If True, also return the standard error of the mean across skewers.
        symmetrize : bool, optional
            If True, replace each averaged term by the symmetrized combination
                xi_AB,sym = xi_AB + xi_BA
            whenever both (A,B) and (B,A) exist in the catalog.

        Returns
        -------
        lags : ndarray
            Lag grid associated with the catalog.
        avg_catalog : dict
            Dictionary with the same keys (A, B), each containing:
                - 'Xi1D_mean'
                - optionally 'Xi1D_std'
                - optionally 'Xi1D_sem'
                - 'order_left'
                - 'order_right'
                - 'total_order'
                - 'label'
        """
        avg_catalog = {}
        lags_out = None

        for (A, B), entry in Xi1D_catalog.items():
            Xi1D = np.asarray(entry["Xi1D"])

            if symmetrize:
                if (B, A) not in Xi1D_catalog:
                    raise KeyError(f"Missing reverse pair {(B, A)} needed for symmetrization.")
                Xi1D_ba = np.asarray(Xi1D_catalog[(B, A)]["Xi1D"])
                values = Xi1D + Xi1D_ba
                label = f"\\xi_{{{subset_label(A)}\\,{subset_label(B)}}}^{{sym}}"
            else:
                values = Xi1D
                label = entry["label"]

            Xi1D_mean = np.mean(values, axis=average_axis)

            out = {
                "Xi1D_mean": Xi1D_mean,
                "order_left": entry["order_left"],
                "order_right": entry["order_right"],
                "total_order": entry["total_order"],
                "label": label,
            }

            if return_std or return_sem:
                xi_std = np.std(values, axis=average_axis, ddof=1)
                if return_std:
                    out["Xi1D_std"] = xi_std
                if return_sem:
                    n_samples = values.shape[average_axis]
                    out["Xi1D_sem"] = xi_std / np.sqrt(n_samples)

            avg_catalog[(A, B)] = out
            lags_out = entry["lags"]

        return lags_out, avg_catalog
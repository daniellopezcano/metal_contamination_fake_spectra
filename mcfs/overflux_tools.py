import numpy as np
from itertools import combinations
from collections.abc import Mapping
from typing import TYPE_CHECKING, Dict, List, Sequence, Union

if TYPE_CHECKING:
    from .lines import LineBundle


class OpticalDepthCube:
    """
    Container for an optical-depth array with an associated LineBundle.

    The stored optical-depth field is assumed to have shape:
        (N_lines, N_skewers, N_spectral)

    where the first axis follows exactly the order of `line_bundle.lines`.
    """

    def __init__(self, tau: np.ndarray, line_bundle: "LineBundle"):
        tau = np.asarray(tau, dtype=float)

        if tau.ndim != 3:
            raise ValueError("`tau` must have shape (N_lines, N_skewers, N_spectral).")
        if tau.shape[0] != line_bundle.n_lines:
            raise ValueError(
                f"Mismatch between tau.shape[0]={tau.shape[0]} and "
                f"line_bundle.n_lines={line_bundle.n_lines}."
            )

        self.tau = tau
        self.line_bundle = line_bundle

    # ────────── Basic metadata ──────────
    @property
    def shape(self) -> tuple:
        return self.tau.shape

    @property
    def n_lines(self) -> int:
        return self.tau.shape[0]

    @property
    def n_skewers(self) -> int:
        return self.tau.shape[1]

    @property
    def n_spectral(self) -> int:
        return self.tau.shape[2]

    @property
    def line_ids(self) -> List[str]:
        return [line.id for line in self.line_bundle.lines]

    @property
    def species_keys(self) -> List[str]:
        return self.line_bundle.species_keys

    @property
    def ref_line_id(self) -> str:
        return self.line_bundle.ref_line_id

    @property
    def delta_v(self) -> np.ndarray:
        """
        Reference-line velocity offsets for the bundle lines.

        Defined in the LineBundle as:
            delta_v_j = c * ln(lambda_ref / lambda_j)
        """
        return np.asarray(self.line_bundle.delta_v, dtype=float)

    @property
    def delta_v_dict(self) -> Dict[str, float]:
        """Dictionary version of `delta_v`, keyed by line ID."""
        return dict(zip(self.line_ids, self.delta_v))

    # ────────── Forwarded line-bundle helpers ──────────
    def lines_for_species(self, species: str):
        return self.line_bundle.lines_for_species(species)

    def indices_for_species(self, species: str) -> List[int]:
        return self.line_bundle.indices_for_species(species)

    # ────────── Internal utilities ──────────
    @staticmethod
    def _periodic_shift_1d(y, delta_x, x, period):
        """
        Periodically shift a 1D field y(x) using the convention

            y_shifted(x) = y(x - delta_x)
        """
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
            raise ValueError("`x` and `y` must be 1D arrays of the same length.")

        idx = np.argsort(x)
        x = x[idx]
        y = y[idx]

        x0 = x[0]
        x_query = ((x - delta_x - x0) % period) + x0

        return np.interp(x_query, x, y, period=period)

    def _resolve_per_line_values(self, values, name="values") -> np.ndarray:
        """
        Convert a per-line input into a 1D array of length n_lines.

        Accepted inputs
        ---------------
        - None: defaults to `self.delta_v`
        - scalar: broadcast to all lines
        - array-like of shape (n_lines,)
        - dict-like keyed by `self.line_ids`
        """
        if values is None:
            return np.asarray(self.delta_v, dtype=float)

        if np.isscalar(values):
            return np.full(self.n_lines, float(values), dtype=float)

        if isinstance(values, Mapping):
            missing = [k for k in self.line_ids if k not in values]
            if missing:
                raise KeyError(f"Missing keys in `{name}`: {missing}")
            return np.asarray([values[k] for k in self.line_ids], dtype=float)

        arr = np.asarray(values, dtype=float)
        if arr.shape != (self.n_lines,):
            raise ValueError(
                f"`{name}` must have shape ({self.n_lines},), got {arr.shape}."
            )
        return arr

    # ────────── Main transformations ──────────
    def periodic_shift(self, delta_x=None, x=None, period=None) -> np.ndarray:
        """
        Periodically shift the line-stacked optical-depth array.

        Parameters
        ----------
        delta_x : float, array-like, dict, or None
            Shift applied independently to each line.
            Accepted forms:
              - None: defaults to `self.delta_v`
              - scalar: same shift for every line
              - array of shape (N_lines,)
              - dict keyed by bundle line IDs
        x : array, shape (N_spectral,)
            Spectral grid.
        period : float
            Period of the grid.

        Returns
        -------
        tau_shifted : ndarray, shape (N_lines, N_skewers, N_spectral)
        """
        if x is None:
            raise ValueError("`x` is required.")
        if period is None:
            raise ValueError("`period` is required.")

        x = np.asarray(x, dtype=float)
        if x.ndim != 1:
            raise ValueError("`x` must be 1D.")
        if x.size != self.n_spectral:
            raise ValueError(
                f"len(x)={x.size} must match the spectral axis size {self.n_spectral}."
            )

        delta_x = self._resolve_per_line_values(delta_x, name="delta_x")

        tau_flat = self.tau.reshape(self.n_lines, -1, self.n_spectral)
        shifted_flat = np.empty_like(tau_flat, dtype=float)

        for i_line in range(self.n_lines):
            dx = delta_x[i_line]
            for i_row in range(tau_flat.shape[1]):
                shifted_flat[i_line, i_row] = self._periodic_shift_1d(
                    y=tau_flat[i_line, i_row],
                    delta_x=dx,
                    x=x,
                    period=period,
                )

        return shifted_flat.reshape(self.tau.shape)

    def build_overflux(self, tau=None, ens_axis=0):
        """
        Build per-line overflux fields and the total overflux field.

        Definitions
        -----------
        For each line i:
            F_i = exp(-tau_i)
            delta_i = F_i / <F_i> - 1

        For the total field:
            F_tot = prod_i F_i
            delta_tot = F_tot / <F_tot> - 1

        Parameters
        ----------
        tau : array, shape (N_lines, N_skewers, N_spectral), optional
            Optical-depth array to process.
            If None, uses `self.tau`.
        ens_axis : int or tuple of int, default=0
            Axis (or axes) over which to average the total transmitted flux.
            Since F_tot has shape (N_skewers, N_spectral), using ens_axis=0
            reproduces your previous convention:
                Fbar_tot = mean(F_tot, axis=0)

        Returns
        -------
        overflux : ndarray, shape (N_lines, N_skewers, N_spectral)
            Per-line overflux fields.
        overflux_total : ndarray
            Total overflux field.
        """
        if tau is None:
            tau = self.tau

        tau = np.asarray(tau, dtype=float)

        if tau.ndim != 3:
            raise ValueError("`tau` must have shape (N_lines, N_skewers, N_spectral).")
        if tau.shape[0] != self.n_lines:
            raise ValueError(
                f"`tau.shape[0]` must match `self.n_lines={self.n_lines}`, "
                f"got {tau.shape[0]}."
            )

        F = np.exp(-tau)

        # Preserves your previous convention exactly
        Fbar = np.mean(F, axis=(1, 2), keepdims=True)
        overflux = F / Fbar - 1.0

        F_tot = np.prod(F, axis=0)

        if ens_axis is None:
            Fbar_tot = np.mean(F_tot)
        else:
            Fbar_tot = np.mean(F_tot, axis=ens_axis, keepdims=True)

        overflux_total = F_tot / Fbar_tot - 1.0

        return overflux, overflux_total

    def build_subset_fields(self, overflux, max_order=None, labels=None):
        """
        Build product fields for all non-empty subsets of the line axis.

        Parameters
        ----------
        overflux : array, shape (N_lines, N_skewers, N_spectral)
            Per-line overflux fields.
        max_order : int, optional
            Maximum subset size to include. If None, include all orders.
        labels : sequence, optional
            Labels associated with the line axis.
            If None, defaults to the bundle line IDs.

        Returns
        -------
        subset_fields : ndarray, shape (N_subsets, N_skewers, N_spectral)
            Product field for each subset.
        subset_labels : list[tuple]
            Labels describing each subset, in the same order as subset_fields.
        """
        overflux = np.asarray(overflux, dtype=float)

        if overflux.ndim != 3:
            raise ValueError("`overflux` must have shape (N_lines, N_skewers, N_spectral).")
        if overflux.shape[0] != self.n_lines:
            raise ValueError(
                f"`overflux.shape[0]` must match `self.n_lines={self.n_lines}`, "
                f"got {overflux.shape[0]}."
            )

        if max_order is None:
            max_order = self.n_lines
        if not (1 <= max_order <= self.n_lines):
            raise ValueError(
                f"`max_order` must satisfy 1 <= max_order <= {self.n_lines}."
            )

        if labels is None:
            labels = self.line_ids
        else:
            labels = list(labels)
            if len(labels) != self.n_lines:
                raise ValueError(
                    f"`labels` must have length {self.n_lines}, got {len(labels)}."
                )

        subset_arrays = []
        subset_labels = []

        for r in range(1, max_order + 1):
            for subset in combinations(range(self.n_lines), r):
                field = np.prod(overflux[list(subset)], axis=0)
                subset_arrays.append(field)
                subset_labels.append(tuple(labels[i] for i in subset))

        subset_fields = np.stack(subset_arrays, axis=0)
        return subset_fields, subset_labels
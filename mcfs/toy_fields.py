import numpy as np
from scipy.fft import rfftfreq, irfft


class ShiftedGaussianTauGenerator1D:
    """
    Generate n_fields optical-depth-like fields tau_i(v) from a single
    1D Gaussian random field tau_ref(v), using periodic shifts.

    Returns only:
        tau_fields with shape (N_samples, n_fields, Npix)

    Parameters
    ----------
    Npix : int
        Number of pixels.
    L_box : float
        Box/domain length.
    delta_v_list : array-like
        Shifts applied to the reference field.
    pk_func : callable
        Function of the form pk_func(k) returning the target 1D power spectrum.
    mean_tau : float, optional
        Mean added to the sampled GRF.
    contrast : float, optional
        Rescales the fluctuation amplitude after standardization.
    round_shifts_to_grid : bool, optional
        If True, shifts are rounded to the nearest grid pixel.
    """

    def __init__(
        self,
        Npix,
        L_box,
        delta_v_list,
        pk_func,
        mean_tau=1.0,
        contrast=0.35,
        round_shifts_to_grid=True,
    ):
        self.Npix = int(Npix)
        self.L_box = float(L_box)
        self.delta_v_list = np.asarray(delta_v_list, dtype=float)
        self.n_fields = len(self.delta_v_list)

        self.dv = self.L_box / self.Npix
        self.v = np.arange(self.Npix) * self.dv
        self.k = 2.0 * np.pi * rfftfreq(self.Npix, d=self.dv)

        self.mean_tau = float(mean_tau)
        self.contrast = float(contrast)
        self.round_shifts_to_grid = bool(round_shifts_to_grid)

        if pk_func is None:
            raise ValueError("You must provide a pk_func(k) callable.")
        self.pk_func = pk_func
        self.Pk_target = np.asarray(self.pk_func(self.k), dtype=float)

        if self.Pk_target.shape != self.k.shape:
            raise ValueError("pk_func(k) must return an array with the same shape as k.")

        if np.any(self.Pk_target < 0):
            raise ValueError("pk_func(k) must return a non-negative power spectrum.")

        if self.round_shifts_to_grid:
            self.shift_pix = np.rint(self.delta_v_list / self.dv).astype(int)
        else:
            shift_exact = self.delta_v_list / self.dv
            if not np.allclose(shift_exact, np.round(shift_exact)):
                raise ValueError(
                    "delta_v_list must be integer multiples of dv when "
                    "round_shifts_to_grid=False."
                )
            self.shift_pix = shift_exact.astype(int)

    def _sample_tau_ref(self, N_samples, rng):
        fk = np.zeros((N_samples, self.k.size), dtype=complex)

        sigma_k = np.sqrt((self.Npix / self.dv) * self.Pk_target / 2.0)

        if self.k.size > 2:
            fk[:, 1:-1] = (
                rng.normal(size=(N_samples, self.k.size - 2))
                + 1j * rng.normal(size=(N_samples, self.k.size - 2))
            ) * sigma_k[None, 1:-1]

        if self.Npix % 2 == 0:
            fk[:, -1] = (
                rng.normal(size=N_samples)
                * np.sqrt((self.Npix / self.dv) * self.Pk_target[-1])
            )

        tau_fluct = irfft(fk, n=self.Npix, axis=1)

        std = np.std(tau_fluct, axis=1, keepdims=True)
        std = np.where(std == 0.0, 1.0, std)

        tau_ref = self.mean_tau + self.contrast * tau_fluct / std
        return tau_ref

    def sample(self, N_samples, rng=None, seed=None):
        """
        Generate tau_fields of shape (N_samples, n_fields, Npix).
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        tau_ref = self._sample_tau_ref(int(N_samples), rng)

        tau_fields = np.stack(
            [np.roll(tau_ref, -shift, axis=1) for shift in self.shift_pix],
            axis=1,
        )

        return tau_fields


class ShiftedSpikeTauGenerator1D:
    """
    Generate n_fields optical-depth-like fields tau_i(v) from a common set
    of randomly sampled Gaussian spikes, using periodic shifts.

    Each field i is built as
        tau_i(v) = A_i * sum_p exp[-0.5 * (d_i / sigma_i)^2]
    where the spike centers are shifted by delta_v_i with periodic wrapping.

    Returns only:
        tau_fields with shape (N_samples, n_fields, Npix)
    """

    def __init__(
        self,
        Npix,
        L_box,
        delta_v_list,
        sigma_list,
        amplitude_list,
        n_pairs,
    ):
        self.Npix = int(Npix)
        self.L_box = float(L_box)
        self.n_pairs = int(n_pairs)

        self.dv = self.L_box / self.Npix
        self.v = np.arange(self.Npix) * self.dv

        self.delta_v_list = np.asarray(delta_v_list, dtype=float)
        self.n_fields = len(self.delta_v_list)

        self.sigma_list = np.asarray(sigma_list, dtype=float)
        self.amplitude_list = np.asarray(amplitude_list, dtype=float)

        if self.sigma_list.ndim == 0:
            self.sigma_list = np.full(self.n_fields, float(self.sigma_list))
        if self.amplitude_list.ndim == 0:
            self.amplitude_list = np.full(self.n_fields, float(self.amplitude_list))

        if len(self.sigma_list) != self.n_fields:
            raise ValueError("sigma_list must have length n_fields or be a scalar.")
        if len(self.amplitude_list) != self.n_fields:
            raise ValueError("amplitude_list must have length n_fields or be a scalar.")

    def sample(self, N_samples, rng=None, seed=None):
        """
        Generate tau_fields of shape (N_samples, n_fields, Npix).
        """
        if rng is None:
            rng = np.random.default_rng(seed)

        N_samples = int(N_samples)

        # Base random centers shared by all fields
        centers_ref = rng.uniform(0.0, self.L_box, size=(N_samples, self.n_pairs))

        # Shifted centers for each field
        centers_fields = (
            centers_ref[:, None, :] + self.delta_v_list[None, :, None]
        ) % self.L_box
        # shape: (N_samples, n_fields, n_pairs)

        # Periodic distances to every pixel
        dist = (
            (self.v[None, None, None, :] - centers_fields[:, :, :, None] + self.L_box / 2)
            % self.L_box
        ) - self.L_box / 2
        # shape: (N_samples, n_fields, n_pairs, Npix)

        # Gaussian spikes with per-field sigma
        spikes = np.exp(
            -0.5 * (dist / self.sigma_list[None, :, None, None]) ** 2
        )

        # Sum over pairs and multiply by per-field amplitude
        tau_fields = (
            self.amplitude_list[None, :, None] * np.sum(spikes, axis=2)
        )
        # shape: (N_samples, n_fields, Npix)

        return tau_fields
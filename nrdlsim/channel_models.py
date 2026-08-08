"""NR channel models from 3GPP TR 38.901.

Implements the Tapped Delay Line (TDL) models TDL-A .. TDL-E (Table 7.7.2-1..5)
with configurable RMS delay spread and maximum Doppler shift.  Time evolution
of each tap uses a sum-of-sinusoids Rayleigh/Rician fading generator.  MIMO is
obtained by applying transmit/receive antenna spatial correlation (Kronecker
model, TS 36.101 Annex B correlation coefficients) on top of the per-tap
i.i.d. fading, which is the correlation-based extension of the TDL models.
"""

from __future__ import annotations

import numpy as np

# Normalised tap delays and powers (dB).  LOS models carry a K-factor (dB).
# Source: TR 38.901 Tables 7.7.2-1 .. 7.7.2-5.
_TDL = {
    "TDL-A": {
        "k_db": None,
        "delays": [0.0000, 0.3819, 0.4025, 0.5868, 0.4610, 0.5375, 0.6708,
                   0.5750, 0.7618, 1.5375, 1.8978, 2.2242, 2.1718, 2.4942,
                   2.5119, 3.0582, 4.0810, 4.4579, 4.5695, 4.7966, 5.0066,
                   5.3043, 9.6586],
        "powers": [-13.4, 0.0, -2.2, -4.0, -6.0, -8.2, -9.9, -10.5, -7.5,
                   -15.9, -6.6, -16.7, -12.4, -15.2, -10.8, -11.3, -12.7,
                   -16.2, -18.3, -18.9, -16.6, -19.9, -29.7],
    },
    "TDL-B": {
        "k_db": None,
        "delays": [0.0000, 0.1072, 0.2155, 0.2095, 0.2870, 0.2986, 0.3752,
                   0.5055, 0.3681, 0.3697, 0.5700, 0.5283, 1.1021, 1.2756,
                   1.5474, 1.7842, 2.0169, 2.8294, 3.0219, 3.6187, 4.1067,
                   4.2790, 4.7834],
        "powers": [0.0, -2.2, -4.0, -3.2, -9.8, -1.2, -3.4, -5.2, -7.6, -3.0,
                   -8.9, -9.0, -4.8, -5.7, -7.5, -1.9, -7.6, -12.2, -9.8,
                   -11.4, -14.9, -9.2, -11.3],
    },
    "TDL-C": {
        "k_db": None,
        "delays": [0.0000, 0.2099, 0.2219, 0.2329, 0.2176, 0.6366, 0.6448,
                   0.6560, 0.6584, 0.7935, 0.8213, 0.9336, 1.2285, 1.3083,
                   2.1704, 2.7105, 4.2589, 4.6003, 5.4902, 5.6077, 6.3065,
                   6.6374, 7.0427, 8.6523],
        "powers": [-4.4, -1.2, -3.5, -5.2, -2.5, 0.0, -2.2, -3.9, -7.4, -7.1,
                   -10.7, -11.1, -5.1, -6.8, -8.7, -13.2, -13.9, -13.9, -15.8,
                   -17.1, -16.0, -15.7, -21.6, -22.8],
    },
    "TDL-D": {
        "k_db": 13.3,  # Rician K-factor of the LOS tap
        "delays": [0.0000, 0.0350, 0.6120, 1.3630, 1.4050, 1.8040, 2.5960,
                   1.7750, 4.0420, 7.9370, 9.4240, 9.7080, 12.5250],
        "powers": [-13.5, -18.8, -21.0, -22.8, -17.9, -20.1, -21.9, -22.9,
                   -27.8, -23.6, -24.8, -30.0, -27.7],
        "los_index": 0,
    },
    "TDL-E": {
        "k_db": 22.0,
        "delays": [0.0000, 0.5133, 0.5440, 0.5630, 0.5440, 0.7112, 1.9092,
                   1.9293, 1.9589, 2.6426, 3.7136, 5.4524, 12.0034, 20.6519],
        "powers": [-22.03, -15.8, -18.1, -19.8, -22.9, -22.4, -18.6, -20.8,
                   -22.6, -22.3, -25.6, -20.2, -29.8, -29.2],
        "los_index": 0,
    },
}


def _correlation_matrix(n: int, level: str) -> np.ndarray:
    """Exponential antenna correlation matrix (TS 36.101 Annex B style)."""
    alpha = {"low": 0.0, "medium": 0.3, "high": 0.9}[level]
    idx = np.arange(n)
    return alpha ** np.abs(idx[:, None] - idx[None, :])


class TDLChannel:
    """Time-varying MIMO TDL channel producing per-tap fading over a slot."""

    def __init__(self, model: str, delay_spread_ns: float, max_doppler_hz: float,
                 n_tx: int, n_rx: int, correlation: str = "low",
                 sample_rate_hz: float = 30.72e6, rng=None):
        self.model = model
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.fd = max_doppler_hz
        self.fs = sample_rate_hz
        self.rng = rng or np.random.default_rng()

        spec = _TDL[model]
        self.k_db = spec["k_db"]
        self.los_index = spec.get("los_index", None)
        delays_s = np.array(spec["delays"]) * delay_spread_ns * 1e-9
        powers_lin = 10 ** (np.array(spec["powers"]) / 10.0)

        # For LOS models, split the first tap into specular + Rayleigh parts.
        if self.k_db is not None:
            k = 10 ** (self.k_db / 10.0)
            p0 = powers_lin[self.los_index]
            # normalise so the LOS tap total keeps its listed power
            self.los_power = p0 * k / (k + 1)
            powers_lin[self.los_index] = p0 / (k + 1)
        powers_lin = powers_lin / powers_lin.sum()
        self.delays_s = delays_s
        self.powers = powers_lin
        self.n_taps = len(delays_s)

        self.r_tx = _correlation_matrix(n_tx, correlation)
        self.r_rx = _correlation_matrix(n_rx, correlation)
        self.c_tx = np.linalg.cholesky(self.r_tx + 1e-9 * np.eye(n_tx))
        self.c_rx = np.linalg.cholesky(self.r_rx + 1e-9 * np.eye(n_rx))

        # sum-of-sinusoids parameters (Jakes-like)
        self._n_sin = 20
        self._init_sos()

    def _init_sos(self):
        m = self._n_sin
        # random angles of arrival per tap, tx, rx, sinusoid
        shape = (self.n_taps, self.n_rx, self.n_tx, m)
        self._theta = self.rng.uniform(-np.pi, np.pi, size=shape)
        self._phi = self.rng.uniform(-np.pi, np.pi, size=shape)
        self._alpha = (np.arange(m) + 0.5) * (2 * np.pi / m)

    def taps(self, t: np.ndarray) -> np.ndarray:
        """Return complex tap gains h[tap, rx, tx, time] at times ``t`` (s)."""
        m = self._n_sin
        # doppler frequencies per sinusoid per tap/rx/tx
        wd = 2 * np.pi * self.fd * np.cos(
            self._alpha[None, None, None, :] + self._theta)  # broadcast
        phase = wd[..., None] * t[None, None, None, None, :] \
            + self._phi[..., None]
        # sum over sinusoids -> (tap, rx, tx, time)
        g = np.exp(1j * phase).sum(axis=3) / np.sqrt(m)

        # apply spatial correlation across the (rx, tx) plane for each tap/time
        # g has shape (tap, rx, tx, time)
        g = np.einsum('ij,tjkl->tikl', self.c_rx, g)
        g = np.einsum('kl,tmln->tmkn', self.c_tx, g)

        # scale by sqrt(tap power)
        h = g * np.sqrt(self.powers)[:, None, None, None]

        # add specular LOS component to the designated tap
        if self.k_db is not None:
            los = np.sqrt(self.los_power) * np.exp(
                1j * 2 * np.pi * self.fd * t)          # (time,)
            h[self.los_index] += los[None, None, :] \
                * np.ones((self.n_rx, self.n_tx, 1))
        return h

    def frequency_response(self, freqs_hz: np.ndarray, t: float) -> np.ndarray:
        """MIMO frequency response H[freq, rx, tx] at time ``t``.

        Computes H(f) = sum_taps h_tap * exp(-j 2 pi f tau_tap).
        """
        h = self.taps(np.array([t]))[..., 0]             # (tap, rx, tx)
        phase = np.exp(-1j * 2 * np.pi
                       * freqs_hz[:, None] * self.delays_s[None, :])  # (f, tap)
        # H[f, rx, tx] = sum_tap phase[f, tap] * h[tap, rx, tx]
        return np.einsum('ft,trx->frx', phase, h)


def awgn_frequency_response(n_freq: int, n_tx: int, n_rx: int) -> np.ndarray:
    """Flat identity-like channel for the AWGN reference model."""
    h = np.zeros((n_freq, n_rx, n_tx), dtype=complex)
    d = min(n_tx, n_rx)
    for i in range(d):
        h[:, i, i] = 1.0
    return h

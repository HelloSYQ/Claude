"""NR channel models from 3GPP TR 38.901.

Two families are implemented:

* **TDL** (Tapped Delay Line, Table 7.7.2-1..5): TDL-A..E with configurable RMS
  delay spread and maximum Doppler.  Per-tap fading uses a sum-of-sinusoids
  Rayleigh/Rician generator; MIMO is the correlation-based extension (Kronecker
  antenna correlation on i.i.d. per-tap fading).

* **CDL** (Clustered Delay Line, Table 7.7.1-1..5): CDL-A..E with per-cluster
  angles (AoD/AoA/ZoD/ZoA), intra-cluster ray offsets (Table 7.5-3), per-cluster
  angle spreads and a Ricean specular path for the LOS models (CDL-D/E).  The
  spatial MIMO channel is synthesised from uniform-linear-array steering vectors,
  so it captures the true angular structure rather than a correlation surrogate.

Both channel classes expose ``frequency_response(freqs_hz, t) -> [n_freq,
n_rx, n_tx]`` so they are interchangeable in the link simulator.
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


# ===========================================================================
# CDL (Clustered Delay Line) models, TR 38.901 Table 7.7.1-1 .. 7.7.1-5
# ---------------------------------------------------------------------------
# Each cluster row is [normalised delay, power(dB), AOD, AOA, ZOD, ZOA] with
# angles in degrees.  'spread' = (C_ASD, C_ASA, C_ZSD, C_ZSA) per-cluster rms
# angle spreads (deg).  LOS models (CDL-D/E) additionally carry a specular ray
# ('los_power_db' at the cluster-1 angles) giving the Ricean K-factor.
# ===========================================================================

# Intra-cluster ray angular offsets, TR 38.901 Table 7.5-3 (the +/- basis).
_RAY_OFFSETS = np.array([
    0.0447, -0.0447, 0.1413, -0.1413, 0.2492, -0.2492, 0.3715, -0.3715,
    0.5129, -0.5129, 0.6797, -0.6797, 0.8844, -0.8844, 1.1481, -1.1481,
    1.5195, -1.5195, 2.1551, -2.1551])

_CDL = {
    "CDL-A": {
        "spread": (5.0, 11.0, 3.0, 3.0), "xpr_db": 10.0,
        "clusters": [
            [0.0000, -13.4, -178.1,   51.3,  50.2, 125.4],
            [0.3819,   0.0,   -4.2, -152.7,  93.2,  91.3],
            [0.4025,  -2.2,   -4.2, -152.7,  93.2,  91.3],
            [0.5868,  -4.0,   -4.2, -152.7,  93.2,  91.3],
            [0.4610,  -6.0,   90.2,   76.6, 122.0,  94.0],
            [0.5375,  -8.2,   90.2,   76.6, 122.0,  94.0],
            [0.6708,  -9.9,   90.2,   76.6, 122.0,  94.0],
            [0.5750, -10.5,  121.5,   -1.8, 150.2,  47.1],
            [0.7618,  -7.5,  -81.7,  -41.9,  55.2,  56.0],
            [1.5375, -15.9,  158.4,   94.2,  26.4,  30.1],
            [1.8978,  -6.6,  -83.0,   51.9, 126.4,  58.8],
            [2.2242, -16.7,  134.8, -115.9, 171.6,  26.0],
            [2.1718, -12.4, -153.0,   26.6, 151.4,  49.2],
            [2.4942, -15.2, -172.0,   76.6, 157.2, 143.1],
            [2.5119, -10.8, -129.9,   -7.0,  47.2, 117.4],
            [3.0582, -11.3, -136.0,  -23.0,  40.4, 122.7],
            [4.0810, -12.7,  165.4,  -47.2,  43.3, 123.2],
            [4.4579, -16.2,  148.4,  110.4, 161.8,  32.6],
            [4.5695, -18.3,  132.7,  144.5,  10.8,  27.2],
            [4.7966, -18.9, -118.6,  155.3,  16.7,  15.2],
            [5.0066, -16.6, -154.1,  102.0, 171.7, 146.0],
            [5.3043, -19.9,  126.5, -151.8,  22.7, 150.7],
            [9.6586, -29.7,  -56.2,   55.2, 144.9, 156.1],
        ],
    },
    "CDL-B": {
        "spread": (10.0, 22.0, 3.0, 7.0), "xpr_db": 8.0,
        "clusters": [
            [0.0000,   0.0,   9.3, -173.3, 105.8,  78.9],
            [0.1072,  -2.2,   9.3, -173.3, 105.8,  78.9],
            [0.2155,  -4.0,   9.3, -173.3, 105.8,  78.9],
            [0.2095,  -3.2, -34.1,  125.5, 115.3,  63.3],
            [0.2870,  -9.8, -65.4,  -88.0, 119.0,  59.9],
            [0.2986,  -1.2, -11.4,  155.1, 103.7,  67.5],
            [0.3752,  -3.4, -11.4,  155.1, 103.7,  67.5],
            [0.5055,  -5.2, -11.4,  155.1, 103.7,  67.5],
            [0.3681,  -7.6, -67.2,  -89.8, 118.2,  82.6],
            [0.3697,  -3.0,  52.5,  132.1, 102.0,  66.3],
            [0.5700,  -8.9, -72.0,  -83.6, 100.4,  61.6],
            [0.5283,  -9.0,  74.3,   95.3,  98.3,  58.0],
            [1.1021,  -4.8, -52.2,  103.7, 103.4,  78.2],
            [1.2756,  -5.7, -50.5,  -87.8, 102.5,  82.0],
            [1.5474,  -7.5,  61.4,  -92.5, 101.4,  62.4],
            [1.7842,  -1.9,  30.6, -139.1, 103.0,  62.5],
            [2.0169,  -7.6, -72.5,  -90.6, 100.0,  71.6],
            [2.8294, -12.2, -90.6,   58.6, 115.2,  55.8],
            [3.0219,  -9.8, -77.6,  -79.0, 100.5,  60.1],
            [3.6187, -11.4, -82.6,   65.8, 119.6,  60.6],
            [4.1067, -14.9, -103.6,  52.7, 118.5,  60.0],
            [4.2790,  -9.2,  75.6,   88.7, 102.5,  61.3],
            [4.7834, -11.3, -77.6,  -60.4, 101.8,  70.5],
        ],
    },
    "CDL-C": {
        "spread": (2.0, 15.0, 3.0, 7.0), "xpr_db": 7.0,
        "clusters": [
            [0.0000,  -4.4,  -46.6, -101.0,  97.2,  87.6],
            [0.2099,  -1.2,  -22.8,  120.0,  98.6,  72.1],
            [0.2219,  -3.5,  -22.8,  120.0,  98.6,  72.1],
            [0.2329,  -5.2,  -22.8,  120.0,  98.6,  72.1],
            [0.2176,  -2.5,  -40.7, -127.5, 100.6,  70.1],
            [0.6366,   0.0,    0.3,  170.4,  99.2,  75.3],
            [0.6448,  -2.2,    0.3,  170.4,  99.2,  75.3],
            [0.6560,  -3.9,    0.3,  170.4,  99.2,  75.3],
            [0.6584,  -7.4,   73.1,   55.4, 105.2,  67.4],
            [0.7935,  -7.1,  -64.5,   66.5,  95.3,  63.8],
            [0.8213, -10.7,   80.2,  -48.1, 106.1,  71.4],
            [0.9336, -11.1,  -97.1,   46.9,  93.5,  60.5],
            [1.2285,  -5.1,  -55.3,   68.1, 103.7,  90.6],
            [1.3083,  -6.8,  -64.3,  -68.7, 104.2,  60.1],
            [2.1704,  -8.7,  -78.5,   81.5,  93.0,  61.0],
            [2.7105, -13.2,  102.7,   30.7, 104.2, 100.7],
            [4.2589, -13.9,   99.2,  -16.4,  94.9,  62.3],
            [4.6003, -13.9,   88.8,    3.8,  93.1,  66.7],
            [5.4902, -15.8,   -1.9,  -13.7, 106.0,  52.9],
            [5.6077, -17.1,  -45.6,    9.7,  93.8,  61.8],
            [6.3065, -16.0,  -89.8,    5.6, 108.8,  51.9],
            [6.6374, -15.7,   85.6,  -33.4,  95.6,  61.7],
            [7.0427, -21.6,   86.4,  -47.9,  95.9,  57.8],
            [8.6523, -22.8, -104.7, -132.6, 108.4,  57.3],
        ],
    },
    "CDL-D": {
        "spread": (5.0, 8.0, 3.0, 3.0), "xpr_db": 11.0,
        "los_power_db": -0.2, "los_angles": (0.0, -180.0, 98.5, 81.5),
        "clusters": [
            [0.0000, -13.5,    0.0, -180.0,  98.5,  81.5],
            [0.0350, -18.8,   89.2,   89.2,  85.5,  86.9],
            [0.6120, -21.0,   89.2,   89.2,  85.5,  86.9],
            [1.3630, -22.8,   89.2,   89.2,  85.5,  86.9],
            [1.4050, -17.9,   13.0,  163.0,  97.5,  79.4],
            [1.8040, -20.1,   13.0,  163.0,  97.5,  79.4],
            [2.5960, -21.9,   13.0,  163.0,  97.5,  79.4],
            [1.7750, -22.9,   34.6, -137.0,  98.5,  78.2],
            [4.0420, -27.8,  -64.5,   74.5,  88.4,  73.6],
            [7.9370, -23.6,  -32.9,  127.7,  91.3,  78.3],
            [9.4240, -24.8,   52.6, -119.6, 103.8,  87.0],
            [9.7080, -30.0, -132.1,   -9.1,  80.3,  70.6],
            [12.525, -27.7,   77.2,  -83.8, 107.8,  73.1],
        ],
    },
    "CDL-E": {
        "spread": (5.0, 11.0, 3.0, 7.0), "xpr_db": 8.0,
        "los_power_db": -0.03, "los_angles": (0.0, -180.0, 99.6, 80.4),
        "clusters": [
            [0.0000, -22.03,   0.0, -180.0,  99.6,  80.4],
            [0.5133, -15.8,   57.5,   18.2, 104.2,  80.4],
            [0.5440, -18.1,   57.5,   18.2, 104.2,  80.4],
            [0.5630, -19.8,   57.5,   18.2, 104.2,  80.4],
            [0.5440, -22.9,  -20.1,  101.8,  99.4,  80.8],
            [0.7112, -22.4,   16.2,  112.9, 100.8,  86.3],
            [1.9092, -18.6,    9.3, -155.5,  98.8,  82.7],
            [1.9293, -20.8,    9.3, -155.5,  98.8,  82.7],
            [1.9589, -22.6,    9.3, -155.5,  98.8,  82.7],
            [2.6426, -22.3,   19.0, -143.3, 100.8,  82.9],
            [3.7136, -25.6,   32.9,  -94.7,  96.4,  88.0],
            [5.4524, -20.2,   12.7,  147.0,  98.5,  79.6],
            [12.003, -29.8,   -1.0, -132.4,  99.7,  80.0],
            [20.652, -29.2,  -25.3,  147.2, 101.6,  76.6],
        ],
    },
}


def _ula_steering(n_ant: int, az_deg: np.ndarray, zen_deg: np.ndarray,
                  spacing: float = 0.5) -> np.ndarray:
    """Uniform linear array (horizontal) steering vectors.

    Returns array of shape (..., n_ant) giving the per-element phase response
    exp(j 2*pi d k sin(zenith) cos(azimuth)) for a horizontal ULA along x.
    Zenith still modulates the effective aperture; it also drives Doppler.
    """
    az = np.deg2rad(az_deg)
    zen = np.deg2rad(zen_deg)
    # directional cosine along the array axis (x): sin(theta)*cos(phi)
    u = np.sin(zen) * np.cos(az)
    k = np.arange(n_ant)
    phase = 2 * np.pi * spacing * np.multiply.outer(u, k)   # (..., n_ant)
    return np.exp(1j * phase)


def _dir_cosines(az_deg, zen_deg):
    az = np.deg2rad(az_deg)
    zen = np.deg2rad(zen_deg)
    return np.stack([np.sin(zen) * np.cos(az),
                     np.sin(zen) * np.sin(az),
                     np.cos(zen)], axis=-1)


class CDLChannel:
    """Time-varying MIMO CDL channel (TR 38.901 clause 7.7.1).

    The channel for each cluster is the sum of ``rays_per_cluster`` sub-rays
    whose angles are the cluster mean plus the Table 7.5-3 offsets scaled by the
    per-cluster angle spreads.  Each ray contributes an outer product of the
    receive- and transmit-array steering vectors, a random initial phase, and a
    Doppler term set by the ray's arrival direction and the UE velocity.
    """

    RAYS = 20

    def __init__(self, model: str, delay_spread_ns: float, max_doppler_hz: float,
                 n_tx: int, n_rx: int, carrier_freq_hz: float = 3.5e9,
                 tx_spacing: float = 0.5, rx_spacing: float = 0.5,
                 travel_az_deg: float = 0.0, travel_zen_deg: float = 90.0,
                 rng=None):
        if model not in _CDL:
            raise ValueError(f"unknown CDL model {model}")
        self.model = model
        self.n_tx = n_tx
        self.n_rx = n_rx
        self.fd = max_doppler_hz
        self.tx_spacing = tx_spacing
        self.rx_spacing = rx_spacing
        self.rng = rng or np.random.default_rng()

        spec = _CDL[model]
        clusters = np.array(spec["clusters"], float)
        self.delays_s = clusters[:, 0] * delay_spread_ns * 1e-9
        powers_lin = 10 ** (clusters[:, 1] / 10.0)
        c_asd, c_asa, c_zsd, c_zsa = spec["spread"]

        n_clu = len(clusters)
        self.n_clu = n_clu
        # per-cluster, per-ray angles (deg): shape (n_clu, RAYS)
        self.aod = clusters[:, 2][:, None] + c_asd * _RAY_OFFSETS[None, :]
        self.zod = clusters[:, 4][:, None] + c_zsd * _RAY_OFFSETS[None, :]
        # random coupling of AoA/ZoA rays vs AoD/ZoD rays (TR 38.901 step 7)
        perm_a = np.array([self.rng.permutation(self.RAYS)
                           for _ in range(n_clu)])
        perm_z = np.array([self.rng.permutation(self.RAYS)
                           for _ in range(n_clu)])
        aoa_base = clusters[:, 3][:, None] + c_asa * _RAY_OFFSETS[None, :]
        zoa_base = clusters[:, 5][:, None] + c_zsa * _RAY_OFFSETS[None, :]
        self.aoa = np.take_along_axis(aoa_base, perm_a, axis=1)
        self.zoa = np.take_along_axis(zoa_base, perm_z, axis=1)

        # LOS specular path (Ricean K) for CDL-D/E
        self.has_los = "los_power_db" in spec
        if self.has_los:
            p_los = 10 ** (spec["los_power_db"] / 10.0)
            # Ricean scaling: normalise so specular + diffuse total power = 1
            total = p_los + powers_lin.sum()
            self.p_los = p_los / total
            powers_lin = powers_lin / total
            self.los_angles = spec["los_angles"]
        else:
            powers_lin = powers_lin / powers_lin.sum()
        self.powers = powers_lin

        # random initial phases per ray
        self.phase0 = self.rng.uniform(-np.pi, np.pi, size=(n_clu, self.RAYS))

        # precompute steering vectors (angle-only): (n_clu, RAYS, n_ant)
        self.a_tx = _ula_steering(n_tx, self.aod, self.zod, tx_spacing)
        self.a_rx = _ula_steering(n_rx, self.aoa, self.zoa, rx_spacing)

        # Doppler per ray from arrival direction . travel direction
        v_hat = _dir_cosines(np.array(travel_az_deg), np.array(travel_zen_deg))
        rx_dir = _dir_cosines(self.aoa, self.zoa)             # (n_clu, RAYS, 3)
        self.doppler = self.fd * (rx_dir @ v_hat)             # (n_clu, RAYS)
        if self.has_los:
            los_rx = _dir_cosines(np.array(self.los_angles[1]),
                                  np.array(self.los_angles[3]))
            self.los_doppler = self.fd * float(los_rx @ v_hat)
            self.los_a_tx = _ula_steering(
                n_tx, np.array(self.los_angles[0]),
                np.array(self.los_angles[2]), tx_spacing)
            self.los_a_rx = _ula_steering(
                n_rx, np.array(self.los_angles[1]),
                np.array(self.los_angles[3]), rx_spacing)

    def _cluster_spatial(self, t: float) -> np.ndarray:
        """Per-cluster spatial matrices at time ``t``: (n_clu, n_rx, n_tx)."""
        # ray coefficient: sqrt(P_c / M) * exp(j(phase0 + 2*pi*doppler*t))
        coef = np.sqrt(self.powers[:, None] / self.RAYS) \
            * np.exp(1j * (self.phase0 + 2 * np.pi * self.doppler * t))
        # H_c[r,t] = sum_m coef[c,m] * a_rx[c,m,r] * a_tx[c,m,t]
        Hc = np.einsum('cm,cmr,cmt->crt', coef, self.a_rx, self.a_tx)
        if self.has_los:
            los = np.sqrt(self.p_los) * np.exp(1j * 2 * np.pi
                                               * self.los_doppler * t)
            Hc[0] += los * np.outer(self.los_a_rx, self.los_a_tx)
        return Hc

    def frequency_response(self, freqs_hz: np.ndarray, t: float) -> np.ndarray:
        """MIMO frequency response H[freq, rx, tx] at time ``t``."""
        Hc = self._cluster_spatial(t)                         # (n_clu, rx, tx)
        phase = np.exp(-1j * 2 * np.pi
                       * freqs_hz[:, None] * self.delays_s[None, :])  # (f, c)
        return np.einsum('fc,crt->frt', phase, Hc)

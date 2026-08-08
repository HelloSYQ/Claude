"""Time-domain OFDM waveform generation and reception (TS 38.211 clause 5.3).

This module lifts the simulator from a frequency-domain (per-resource-element)
abstraction to a true sample-level waveform:

    resource grid --IFFT + cyclic prefix--> time samples
       --> RF impairments (CFO, timing offset) + multipath + AWGN -->
    --remove CP + FFT--> received resource grid

Because it operates on samples it captures effects the frequency-domain path
cannot: inter-carrier interference from a carrier frequency offset, the loss of
orthogonality when the delay spread exceeds the cyclic prefix, and the phase
ramp/ISI from a symbol-timing error.  A one-tap (per-subcarrier) equaliser then
recovers the grid, and the residual error (EVM) quantifies the impairment.
"""

from __future__ import annotations

import numpy as np

SYMBOLS_PER_SLOT = 14


def _next_pow2(n: int) -> int:
    return 1 << (int(np.ceil(np.log2(n))))


class OFDMModulator:
    """NR CP-OFDM modulator/demodulator for one slot.

    FFT size is the smallest power of two that holds the occupied subcarriers.
    Normal cyclic prefix lengths follow TS 38.211 5.3.1 (the first symbol of a
    0.5 ms boundary carries the longer CP).
    """

    def __init__(self, n_rb: int, scs_hz: float, mu: int = 1,
                 fft_size: int | None = None):
        self.n_sc = n_rb * 12
        self.n_fft = fft_size or max(128, _next_pow2(self.n_sc))
        self.scs_hz = scs_hz
        self.mu = mu
        self.fs = self.n_fft * scs_hz                 # sample rate (Hz)
        self.n_sym = SYMBOLS_PER_SLOT
        self.cp = self._cp_lengths()
        self.sc_idx = self._subcarrier_indices()

    # -- structure -----------------------------------------------------------
    def _cp_lengths(self) -> np.ndarray:
        short = int(round(144 * self.n_fft / 2048))
        long = int(round((144 + 16) * self.n_fft / 2048))
        cp = np.full(self.n_sym, short, dtype=int)
        cp[0] = long                                  # 0.5 ms boundary symbol
        if self.mu == 0:
            cp[7] = long
        return cp

    def _subcarrier_indices(self) -> np.ndarray:
        """Occupied subcarriers centred around DC, in FFT bin order."""
        half = self.n_sc // 2
        return (np.arange(self.n_sc) - half) % self.n_fft

    @property
    def slot_length(self) -> int:
        return int(self.n_sym * self.n_fft + self.cp.sum())

    # -- transmit / receive --------------------------------------------------
    def modulate(self, grid: np.ndarray) -> np.ndarray:
        """Resource grid [n_sc, n_sym] -> time-domain sample vector."""
        assert grid.shape == (self.n_sc, self.n_sym), grid.shape
        out = []
        for l in range(self.n_sym):
            spec = np.zeros(self.n_fft, dtype=complex)
            spec[self.sc_idx] = grid[:, l]
            t = np.fft.ifft(spec) * np.sqrt(self.n_fft)   # unit-power symbols
            out.append(np.concatenate([t[self.n_fft - self.cp[l]:], t]))
        return np.concatenate(out)

    def demodulate(self, samples: np.ndarray, timing_offset: int = 0
                   ) -> np.ndarray:
        """Time samples -> resource grid, sampling FFT windows after each CP.

        ``timing_offset`` shifts the FFT window (in samples); a small negative
        offset stays inside the CP (recoverable phase ramp), a large or positive
        offset leaks the next/previous symbol into the window (ISI).
        """
        grid = np.zeros((self.n_sc, self.n_sym), dtype=complex)
        pos = 0
        for l in range(self.n_sym):
            start = pos + self.cp[l] + timing_offset
            start = max(0, min(start, samples.size - self.n_fft))
            win = samples[start:start + self.n_fft]
            spec = np.fft.fft(win) / np.sqrt(self.n_fft)
            grid[:, l] = spec[self.sc_idx]
            pos += self.cp[l] + self.n_fft
        return grid

    def channel_frequency(self, h_time: np.ndarray) -> np.ndarray:
        """Per-subcarrier channel response for a time-domain impulse response."""
        H = np.fft.fft(h_time, self.n_fft)
        return H[self.sc_idx]


# ---------------------------------------------------------------------------
# RF impairments (applied on the time-domain sample stream)
# ---------------------------------------------------------------------------

def apply_cfo(samples: np.ndarray, cfo_hz: float, fs: float,
              phase0: float = 0.0) -> np.ndarray:
    """Apply a carrier frequency offset: x[n] * exp(j 2*pi f_cfo n / fs)."""
    n = np.arange(samples.size)
    return samples * np.exp(1j * (2 * np.pi * cfo_hz * n / fs + phase0))


def apply_multipath(samples: np.ndarray, h_time: np.ndarray) -> np.ndarray:
    """Linear convolution with a time-domain channel impulse response."""
    return np.convolve(samples, h_time)[:samples.size]


def add_awgn(samples: np.ndarray, snr_db: float, rng, signal_power=None
             ) -> np.ndarray:
    if signal_power is None:
        signal_power = np.mean(np.abs(samples) ** 2)
    noise_power = signal_power / (10 ** (snr_db / 10.0))
    noise = np.sqrt(noise_power / 2) * (rng.standard_normal(samples.shape)
                                        + 1j * rng.standard_normal(samples.shape))
    return samples + noise


# ---------------------------------------------------------------------------
# End-to-end single-stream waveform analysis (EVM / effective SINR)
# ---------------------------------------------------------------------------

def waveform_evm(mod: OFDMModulator, grid_tx: np.ndarray, snr_db: float,
                 cfo_hz: float = 0.0, timing_offset: int = 0,
                 h_time: np.ndarray | None = None, track_phase: bool = True,
                 rng=None):
    """Run a grid through the full waveform chain and measure EVM.

    Returns (evm_rms, effective_sinr_db).  A one-tap equaliser inverts the known
    channel response.  When ``track_phase`` is set, the per-symbol common phase
    error (the removable, DM-RS/PT-RS-tracked part of a CFO) is estimated and
    corrected, so the residual EVM reflects the irreducible inter-carrier
    interference plus noise rather than the accumulating phase rotation.
    """
    rng = rng or np.random.default_rng(0)
    x = mod.modulate(grid_tx)
    sig_power = np.mean(np.abs(x) ** 2)
    if h_time is not None:
        x = apply_multipath(x, h_time)
    if cfo_hz:
        x = apply_cfo(x, cfo_hz, mod.fs)
    x = add_awgn(x, snr_db, rng, signal_power=sig_power)
    grid_rx = mod.demodulate(x, timing_offset=timing_offset)

    # Joint channel / phase estimation (data-aided).  A front-loaded reference
    # gives an initial per-subcarrier estimate; the per-symbol common phase
    # error (the removable part of a CFO) is then estimated and the received
    # symbols are de-rotated before averaging the channel estimate.  This
    # absorbs the actual channel and any timing-induced phase ramp without the
    # CFO phase drift corrupting the estimate, so:
    #   * a timing error inside the CP is recoverable, beyond it leaves ISI;
    #   * a CFO leaves only the irreducible inter-carrier interference.
    tx = grid_tx
    H0 = grid_rx[:, 0] / (tx[:, 0] + 1e-12)
    eq0 = grid_rx / H0[:, None]
    if track_phase:
        cpe = np.sum(eq0 * np.conj(tx), axis=0)
        cpe /= np.abs(cpe) + 1e-12
    else:
        cpe = np.ones(mod.n_sym, dtype=complex)
    rx_derot = grid_rx * np.conj(cpe)[None, :]
    num = np.sum(rx_derot * np.conj(tx), axis=1)
    den = np.sum(np.abs(tx) ** 2, axis=1) + 1e-12
    H_est = (num / den)[:, None]
    grid_eq = rx_derot / H_est

    err = grid_eq - grid_tx
    evm = np.sqrt(np.mean(np.abs(err) ** 2) / np.mean(np.abs(grid_tx) ** 2))
    sinr_db = -20 * np.log10(evm + 1e-12)
    return float(evm), float(sinr_db)


def cfo_ici_sinr_db(cfo_hz: float, scs_hz: float) -> float:
    """Analytic ICI-limited SINR ceiling from a residual CFO (small-offset).

    For a normalised offset eps = f_cfo / SCS, the inter-carrier interference
    power is approximately (pi*eps)^2/3, giving SINR ~ 3/(pi*eps)^2.
    """
    eps = cfo_hz / scs_hz
    ici = (np.pi * eps) ** 2 / 3.0
    return float(10 * np.log10(1.0 / max(ici, 1e-12)))

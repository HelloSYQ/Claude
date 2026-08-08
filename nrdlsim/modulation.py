"""QAM modulation and soft (LLR) demodulation per TS 38.211 clause 5.1.

Supports QPSK (Qm=2), 16QAM (Qm=4), 64QAM (Qm=6) and 256QAM (Qm=8) using
the Gray-mapped constellations defined in the specification.  Constellations
are normalised to unit average energy.
"""

from __future__ import annotations

import numpy as np

# Normalisation factors 1/sqrt(mean power) per TS 38.211 5.1.
_NORM = {2: 1 / np.sqrt(2), 4: 1 / np.sqrt(10), 6: 1 / np.sqrt(42),
         8: 1 / np.sqrt(170)}


def _build_constellation(qm: int) -> np.ndarray:
    """Build the size-2^Qm constellation indexed by the integer symbol.

    The bit-to-symbol mapping reproduces the TS 38.211 expressions where the
    in-phase and quadrature components are each Gray-coded PAM levels.
    """
    m = qm // 2                      # bits per dimension
    levels = _pam_levels(m)          # Gray-mapped PAM amplitude per bit group
    const = np.zeros(2 ** qm, dtype=complex)
    for sym in range(2 ** qm):
        bits = [(sym >> (qm - 1 - i)) & 1 for i in range(qm)]
        i_bits = bits[0::2]
        q_bits = bits[1::2]
        const[sym] = (levels[_bits_to_index(i_bits)]
                      + 1j * levels[_bits_to_index(q_bits)])
    return const * _NORM[qm]


def _bits_to_index(bits) -> int:
    idx = 0
    for b in bits:
        idx = (idx << 1) | b
    return idx


def _pam_levels(m: int) -> np.ndarray:
    """Gray-coded PAM amplitude for each m-bit label (NR sign convention).

    NR maps bit 0 -> +amplitude.  Amplitudes are the odd integers
    {..., -3, -1, +1, +3, ...} arranged by Gray code.
    """
    n = 2 ** m
    amps = np.zeros(n)
    for label in range(n):
        gray = label ^ (label >> 1)
        level = 2 * gray - (n - 1)         # spread to +/- (n-1)
        amps[label] = -level               # NR: bit 0 -> positive
    return amps


class Modulator:
    def __init__(self, qm: int):
        self.qm = qm
        self.constellation = _build_constellation(qm)
        # precompute bit labels for each constellation point
        self.bit_labels = np.array(
            [[(s >> (qm - 1 - i)) & 1 for i in range(qm)]
             for s in range(2 ** qm)])

    def modulate(self, bits: np.ndarray) -> np.ndarray:
        """Map a bit vector (length multiple of Qm) to QAM symbols."""
        bits = np.asarray(bits).reshape(-1, self.qm)
        weights = 1 << np.arange(self.qm - 1, -1, -1)
        idx = bits.dot(weights)
        return self.constellation[idx]

    def demodulate_llr(self, rx: np.ndarray, noise_var: np.ndarray) -> np.ndarray:
        """Max-log-MAP soft demapping returning LLRs (one per coded bit).

        LLR sign convention: positive LLR favours bit = 0, matching the
        modulate() mapping.  ``noise_var`` may be a scalar or per-symbol array
        (post-equalisation effective noise variance).
        """
        rx = np.asarray(rx).reshape(-1)
        noise_var = np.broadcast_to(np.asarray(noise_var, float), rx.shape)
        # distances to every constellation point: shape (Nsym, 2^Qm)
        d2 = np.abs(rx[:, None] - self.constellation[None, :]) ** 2
        metric = -d2 / noise_var[:, None]
        llr = np.empty((rx.size, self.qm))
        for b in range(self.qm):
            mask0 = self.bit_labels[:, b] == 0
            m0 = metric[:, mask0].max(axis=1)
            m1 = metric[:, ~mask0].max(axis=1)
            llr[:, b] = m0 - m1
        return llr.reshape(-1)


def bits_per_symbol(qm: int) -> int:
    return qm

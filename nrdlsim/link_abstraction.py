"""Link-to-system abstraction: BICM capacity, MIESM effective SINR and an
NR-LDPC block-error model.

Rather than decoding every code block, the fast simulation path predicts the
block-error rate from the set of per-resource-element post-equaliser SINRs.
The chain is:

  1.  Map each RE SINR to its bit-interleaved coded-modulation (BICM) mutual
      information for the active modulation order.
  2.  Average the mutual information across REs (MIESM / RBIR compression) and
      invert to an equivalent AWGN "effective SINR".
  3.  Map the effective SINR to BLER using a code-rate-dependent threshold and
      a waterfall whose sharpness grows with the code-block length, matching
      the behaviour of NR LDPC codes.

This is the standard link-abstraction methodology used to build spectral
efficiency curves; a bit-true LDPC path is available separately in ``ldpc.py``.
"""

from __future__ import annotations

import numpy as np
from functools import lru_cache

from .modulation import _build_constellation


@lru_cache(maxsize=None)
def _bicm_table(qm: int, n_snr: int = 121):
    """Precompute BICM capacity (bits/symbol) vs SNR (dB) for a modulation."""
    snr_db = np.linspace(-20, 40, n_snr)
    const = _build_constellation(qm)
    labels = np.array([[(s >> (qm - 1 - i)) & 1 for i in range(qm)]
                       for s in range(2 ** qm)])
    rng = np.random.default_rng(12345)
    n_mc = 2000
    cap = np.zeros(n_snr)
    for si, sdb in enumerate(snr_db):
        snr = 10 ** (sdb / 10.0)
        noise = np.sqrt(1.0 / (2 * snr))
        idx = rng.integers(0, 2 ** qm, size=n_mc)
        x = const[idx]
        n = noise * (rng.standard_normal(n_mc) + 1j * rng.standard_normal(n_mc))
        y = x + n
        d2 = np.abs(y[:, None] - const[None, :]) ** 2
        metric = -d2 / (2 * noise ** 2)
        total = 0.0
        for b in range(qm):
            mask0 = labels[:, b] == 0
            m0 = metric[:, mask0]
            m1 = metric[:, ~mask0]
            # per-bit mutual information via LLR
            max0 = m0.max(axis=1)
            max1 = m1.max(axis=1)
            llr = max0 - max1
            sent = labels[idx, b]
            # I(bit) = 1 - E[log2(1+exp(-(1-2c)*llr))]
            s = 1 - 2 * sent
            total += 1 - np.mean(np.log2(1 + np.exp(-s * llr)))
        cap[si] = total
    return snr_db, cap


def bicm_capacity(snr_db, qm: int) -> np.ndarray:
    """BICM capacity in bits/symbol for given SNR(dB) and modulation order."""
    grid_db, cap = _bicm_table(qm)
    return np.interp(snr_db, grid_db, cap)


def effective_sinr_miesm(sinr_lin: np.ndarray, qm: int) -> float:
    """MIESM compression of per-RE SINRs to a single effective SINR (dB)."""
    sinr_lin = np.asarray(sinr_lin, float).reshape(-1)
    sinr_lin = np.clip(sinr_lin, 1e-6, 1e6)
    per_re_mi = bicm_capacity(10 * np.log10(sinr_lin), qm)   # bits/symbol
    mean_mi = np.clip(per_re_mi.mean(), 1e-6, qm - 1e-6)
    # invert: find SNR whose BICM capacity == mean_mi
    grid_db, cap = _bicm_table(qm)
    eff_db = np.interp(mean_mi, cap, grid_db)
    return float(eff_db)


def required_snr_db(qm: int, code_rate: float) -> float:
    """AWGN SNR at which BICM capacity equals the coded bits/symbol (Qm*R)."""
    grid_db, cap = _bicm_table(qm)
    target = qm * code_rate
    if target >= cap[-1]:
        return grid_db[-1]
    return float(np.interp(target, cap, grid_db))


def bler_from_effective_sinr(eff_sinr_db: float, qm: int, code_rate: float,
                             tb_bits: int) -> float:
    """NR-LDPC-like BLER waterfall vs effective SINR.

    The threshold is the capacity-achieving SNR plus a coding gap; the slope
    sharpens with block length (longer LDPC blocks decode more steeply).
    """
    snr_req = required_snr_db(qm, code_rate)
    # implementation/coding gap of an NR LDPC code from capacity (dB)
    gap = 1.0 + 0.6 * code_rate
    threshold = snr_req + gap
    # waterfall slope: sharper for longer blocks
    slope = 1.2 + 0.25 * np.log2(max(tb_bits, 64) / 64.0)
    x = slope * (eff_sinr_db - threshold)
    # logistic waterfall in (0,1)
    return float(1.0 / (1.0 + np.exp(x)))

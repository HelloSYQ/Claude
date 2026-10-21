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

import os
import numpy as np
from functools import lru_cache

from .modulation import _build_constellation

_CACHE_DIR = os.path.join(os.path.dirname(__file__), "_cache")


@lru_cache(maxsize=None)
def _bicm_table(qm: int, n_snr: int = 121):
    """BICM capacity (bits/symbol) vs SNR (dB) for a modulation.

    The Monte-Carlo build is ~1 s per modulation; the result is deterministic
    (fixed seed) so it is cached both in-process (lru_cache) and on disk, making
    repeat process invocations start instantly.
    """
    cache_file = os.path.join(_CACHE_DIR, f"bicm_qm{qm}_{n_snr}.npz")
    if os.path.exists(cache_file):
        try:
            d = np.load(cache_file)
            return d["snr_db"], d["cap"]
        except Exception:
            pass
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
    try:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        np.savez(cache_file, snr_db=snr_db, cap=cap)
    except Exception:
        pass
    return snr_db, cap


def bicm_capacity(snr_db, qm: int) -> np.ndarray:
    """BICM capacity in bits/symbol for given SNR(dB) and modulation order."""
    grid_db, cap = _bicm_table(qm)
    return np.interp(snr_db, grid_db, cap)


def effective_sinr_miesm(sinr_lin: np.ndarray, qm: int,
                         weights: np.ndarray | None = None) -> float:
    """MIESM compression of per-RE SINRs to a single effective SINR (dB).

    ``weights`` (same shape as ``sinr_lin``) weight the per-sample mutual
    information, e.g. the detection-order samples of :func:`rml_sinr_samples`.
    """
    sinr_lin = np.asarray(sinr_lin, float).reshape(-1)
    sinr_lin = np.clip(sinr_lin, 1e-6, 1e6)
    per_re_mi = bicm_capacity(10 * np.log10(sinr_lin), qm)   # bits/symbol
    if weights is None:
        mean_mi = per_re_mi.mean()
    else:
        w = np.broadcast_to(np.asarray(weights, float), per_re_mi.shape).reshape(-1)
        mean_mi = np.sum(per_re_mi * w) / np.sum(w)
    mean_mi = np.clip(mean_mi, 1e-6, qm - 1e-6)
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


def _waterfall(qm: int, code_rate: float, tb_bits: int):
    """(threshold_db, slope_per_db) of the BLER waterfall; BLER = 0.5 at threshold.

    The threshold is the capacity-achieving SNR plus a coding gap; the slope
    sharpens with block length (longer LDPC blocks decode more steeply).
    """
    # implementation/coding gap of an NR LDPC code from capacity (dB)
    gap = 1.0 + 0.6 * code_rate
    threshold = required_snr_db(qm, code_rate) + gap
    slope = 1.2 + 0.25 * np.log2(max(tb_bits, 64) / 64.0)
    return threshold, slope


def bler_from_effective_sinr(eff_sinr_db: float, qm: int, code_rate: float,
                             tb_bits: int) -> float:
    """NR-LDPC-like BLER waterfall vs effective SINR (logistic in (0,1))."""
    threshold, slope = _waterfall(qm, code_rate, tb_bits)
    return float(1.0 / (1.0 + np.exp(slope * (eff_sinr_db - threshold))))


def required_eff_sinr_db(qm: int, code_rate: float, tb_bits: int,
                         target_bler: float) -> float:
    """Effective SINR (dB) at which the waterfall reaches ``target_bler``.

    Exact inverse of :func:`bler_from_effective_sinr`, shared by CQI selection
    (UE) and MCS selection (gNB) so both operate at the same BLER target.
    """
    threshold, slope = _waterfall(qm, code_rate, tb_bits)
    p = min(max(target_bler, 1e-6), 1 - 1e-6)
    return float(threshold + np.log((1 - p) / p) / slope)


@lru_cache(maxsize=None)
def _amc_table(mcs_table: int, n_layers: int, n_re_per_rb: int, n_prb: int):
    """Per-MCS (Qm, TBS, BLER-threshold, slope) arrays for :func:`ideal_amc`."""
    from . import mcs_tables, tbs as tbs_mod
    rows = []
    for idx in range(mcs_tables.num_mcs(mcs_table)):
        info = mcs_tables.get_mcs(idx, mcs_table)
        tb = tbs_mod.compute_tbs(n_re_per_rb, n_prb, info.modulation_order,
                                 info.target_code_rate, n_layers)
        thr, slope = _waterfall(info.modulation_order, info.target_code_rate, tb)
        rows.append((info.modulation_order, tb, thr, slope))
    a = np.array(rows, float)
    return a[:, 0].astype(int), a[:, 1], a[:, 2], a[:, 3]


def ideal_amc(sinr_lin, n_layers: int, n_re_per_rb: int, n_prb: int,
              mcs_table: int = 2, weights=None):
    """Ideal AMC for one codeword: the MCS with the largest expected TB bits.

    ``sinr_lin`` are the post-equaliser SINRs of every RE (or RB) the
    codeword occupies, across all of its ``n_layers`` layers.  Each MCS is
    scored with its MIESM effective SINR at the MCS's modulation order and
    the LDPC waterfall, expected bits = TBS * (1 - BLER).  Returns
    (mcs_index, tbs, bler, expected_bits).
    """
    qm_of, tbs, thr, slope = _amc_table(mcs_table, n_layers, n_re_per_rb, n_prb)
    eff = {qm: effective_sinr_miesm(sinr_lin, qm, weights) for qm in np.unique(qm_of)}
    e = np.array([eff[q] for q in qm_of])
    bler = 1.0 / (1.0 + np.exp(slope * (e - thr)))
    bits = tbs * (1.0 - bler)
    i = int(np.argmax(bits))
    return i, int(tbs[i]), float(bler[i]), float(bits[i])


# ---------------------------------------------------------------------------
# Code-block-aware TB error model (TS 38.212 clause 5.2.2 segmentation)
# ---------------------------------------------------------------------------
# A transport block is CRC-attached and split into C LDPC code blocks; the TB
# fails if any code block fails, so TB BLER = 1 - prod_c (1 - BLER_cb,c).
# Each code block follows the waterfall of its own (payload) size and sees
# its own effective SINR (the REs it is mapped to).  ``slope`` overrides the
# per-CB waterfall slope (1/dB) for sensitivity studies; None keeps the
# default size-dependent slope of _waterfall.

def cb_segmentation(tbs: int, code_rate: float):
    """(C, K') = number of code blocks and bits per code block (incl. CRCs).

    TS 38.212 5.2.2 via :func:`nrdlsim.ldpc.segment_transport_block`.
    """
    from .ldpc import segment_transport_block
    s = segment_transport_block(int(tbs), code_rate)
    return s.c, s.k_prime


def _cb_waterfall(qm, code_rate, k_cb, slope):
    threshold, s = _waterfall(qm, code_rate, k_cb)
    return threshold, (s if slope is None else slope)


def tb_bler_from_cb_sinrs(cb_eff_sinr_db, qm: int, code_rate: float, tbs: int,
                          slope: float | None = None) -> float:
    """TB BLER from the effective SINR of each code block."""
    c, k_cb = cb_segmentation(tbs, code_rate)
    e = np.broadcast_to(np.asarray(cb_eff_sinr_db, float), (c,))
    threshold, s = _cb_waterfall(qm, code_rate, k_cb, slope)
    p_cb = 1.0 / (1.0 + np.exp(s * (e - threshold)))
    return float(1.0 - np.prod(1.0 - p_cb))


def required_eff_sinr_db_tb(qm: int, code_rate: float, tbs: int,
                            target_bler: float, slope: float | None = None) -> float:
    """Common per-CB effective SINR at which the TB BLER equals ``target_bler``.

    Inverse of :func:`tb_bler_from_cb_sinrs` when every code block sees the
    same effective SINR: each CB must reach 1 - (1 - target)^(1/C).
    """
    c, k_cb = cb_segmentation(tbs, code_rate)
    p = 1.0 - (1.0 - min(max(target_bler, 1e-6), 1 - 1e-6)) ** (1.0 / c)
    threshold, s = _cb_waterfall(qm, code_rate, k_cb, slope)
    return float(threshold + np.log((1 - p) / p) / s)


def mi_to_eff_sinr_db(mean_mi, qm: int) -> np.ndarray:
    """Invert the BICM curve: effective SINR (dB) for a mean MI (bits/symbol)."""
    grid_db, cap = _bicm_table(qm)
    return np.interp(np.clip(mean_mi, 1e-6, qm - 1e-6), cap, grid_db)


def cb_re_weights(n_rb: int, n_cb: int, n_re_per_rb: int) -> np.ndarray:
    """Fraction of each code block's REs that fall in each RB, shape (C, n_rb).

    Modulation symbols are layer-mapped round-robin, then mapped to REs
    frequency-first (subcarrier, then OFDM symbol) over the allocation
    (TS 38.211 7.3.1), so code block c occupies one contiguous slice of every
    layer's RE stream, sweeping across the band and wrapping into the next
    OFDM symbol.  The per-code-block lengths follow the rate-matching split
    of TS 38.212 5.4.2.1; data REs are taken as n_re_per_rb / 12 full
    symbols, without VRB-to-PRB interleaving.  Built on
    :mod:`nrdlsim.re_mapping`.
    """
    from .ldpc import rate_match_lengths
    from .re_mapping import (code_block_spans, code_block_prb_fraction,
                             vrb_to_prb_non_interleaved)
    # the split depends only on the per-layer RE count, so one layer and
    # one bit per RE give the per-layer spans directly
    e_r = rate_match_lengths(n_re_per_rb * n_rb, n_cb, 1, 1)
    return code_block_prb_fraction(code_block_spans(e_r, 1, 1),
                                   vrb_to_prb_non_interleaved(n_rb))


def mcs_bler(sinr_lin, n_layers: int, mcs: int, n_re_per_rb: int, n_prb: int,
             mcs_table: int = 2, weights=None) -> float:
    """BLER of a codeword at a given MCS on the SINRs ``sinr_lin``."""
    qm_of, tbs, thr, slope = _amc_table(mcs_table, n_layers, n_re_per_rb, n_prb)
    e = effective_sinr_miesm(sinr_lin, int(qm_of[mcs]), weights)
    return float(1.0 / (1.0 + np.exp(slope[mcs] * (e - thr[mcs]))))


def sic_amc(sinr_after, groups, n_re_per_rb: int, n_prb: int,
            mcs_table: int = 2):
    """Ideal AMC and expected bits for codeword-level MMSE-SIC.

    ``groups`` are the layers of each codeword in decoding order and
    ``sinr_after(cancelled)`` returns the per-layer SINRs [n, n_layers] with
    the layers ``cancelled`` (a tuple) removed, e.g. via
    :func:`nrdlsim.receiver.cancelled_sinr`.  Codeword k's MCS is chosen
    (most expected bits) assuming codewords 1..k-1 were decoded and
    cancelled.  At decoding, only codewords whose CRC passed are cancelled,
    so a failure propagates: the expected bits sum over all success
    patterns of the earlier codewords.  Returns (mcs list, tbs list,
    expected bits, P(success) per codeword).
    """
    cache = {}

    def sinr(cancel):
        key = tuple(sorted(cancel))
        if key not in cache:
            cache[key] = sinr_after(key)
        return cache[key]

    mcs, tbs = [], []
    done = []
    for g in groups:
        i, tb, _, _ = ideal_amc(sinr(done)[:, g].ravel(), len(g), n_re_per_rb,
                                n_prb, mcs_table)
        mcs.append(i)
        tbs.append(tb)
        done = done + list(g)

    p_ok = np.zeros(len(groups))

    def walk(k, cancelled, prob):
        if k == len(groups) or prob == 0.0:
            return
        g = groups[k]
        bl = mcs_bler(sinr(cancelled)[:, g].ravel(), len(g), mcs[k], n_re_per_rb,
                      n_prb, mcs_table)
        p_ok[k] += prob * (1.0 - bl)
        walk(k + 1, cancelled + list(g), prob * (1.0 - bl))
        walk(k + 1, cancelled, prob * bl)

    walk(0, [], 1.0)
    return mcs, tbs, float(np.dot(tbs, p_ok)), p_ok

"""CSI feedback: rank (RI), precoding (PMI) and channel-quality (CQI).

Follows the reporting philosophy of TS 38.214 clause 5.2.  From an estimated
MIMO channel the UE:

  * selects the transmission rank that maximises predicted throughput,
  * derives the SVD-based precoder (a proxy for the PMI codebook selection),
  * computes per-layer post-equaliser SINRs and maps the wideband effective
    SINR to the highest CQI meeting the BLER target.

Reports are delayed by a configurable number of slots to model the CSI
acquisition/feedback loop.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections import deque
import numpy as np

from . import mcs_tables
from .link_abstraction import effective_sinr_miesm, bicm_capacity
from .layer_mapping import svd_precoder
from .receiver import batch_mmse_sinr


@dataclass
class CSIReport:
    rank: int
    cqi: int
    precoder: np.ndarray            # W[f, n_tx, rank]
    eff_sinr_db: float


def _mmse_layer_sinr(H_eff: np.ndarray, noise_var: float) -> np.ndarray:
    """Post-MMSE per-layer SINR for effective channel H_eff = H @ W.

    H_eff: [n_rx, n_layers].  Returns SINR per layer (linear).
    """
    n_rx, n_layers = H_eff.shape
    A = H_eff.conj().T @ H_eff + noise_var * np.eye(n_layers)
    Ainv = np.linalg.inv(A)
    # MMSE receiver: G = (H^H H + N0 I)^-1 H^H
    G = Ainv @ H_eff.conj().T
    sinr = np.zeros(n_layers)
    for k in range(n_layers):
        gk = G[k]
        sig = np.abs(gk @ H_eff[:, k]) ** 2
        interf = 0.0
        for j in range(n_layers):
            if j != k:
                interf += np.abs(gk @ H_eff[:, j]) ** 2
        noise = noise_var * np.abs(gk @ gk.conj())
        sinr[k] = sig / (interf + noise + 1e-12)
    return sinr


def compute_csi(H_freq: np.ndarray, noise_var: float, max_rank: int,
                mcs_table: int, target_bler: float = 0.1) -> CSIReport:
    """Compute the best (rank, precoder, CQI) for the estimated channel."""
    n_f, n_rx, n_tx = H_freq.shape
    max_rank = min(max_rank, n_rx, n_tx)
    cqi_table = mcs_tables.MCS_TO_CQI_TABLE[mcs_table]

    best = None
    best_tput = -1.0
    for rank in range(1, max_rank + 1):
        W = svd_precoder(H_freq, rank)
        # batched post-MMSE SINR across all RBs at once
        H_eff = H_freq @ W                      # [n_f, n_rx, rank]
        sinrs = batch_mmse_sinr(H_eff, noise_var).reshape(-1)
        # choose a representative modulation order for compression (64QAM)
        eff_db = effective_sinr_miesm(sinrs, qm=6)
        cqi = _sinr_to_cqi(eff_db, cqi_table, target_bler)
        _, rate, eff = mcs_tables.get_cqi(cqi, cqi_table)
        tput = rank * eff * (1.0 if cqi > 0 else 0.0)
        if tput > best_tput:
            best_tput = tput
            best = CSIReport(rank, cqi, W, eff_db)
    return best


from functools import lru_cache


@lru_cache(maxsize=None)
def _cqi_required_snr(cqi_table: int, target_bler: float):
    """Required effective SINR (dB) per CQI index — computed once and cached."""
    from .link_abstraction import required_snr_db
    margin = 0.5 * np.log10(0.1 / max(target_bler, 1e-3) + 1.0)
    reqs = []
    for cqi in range(1, 16):
        qm, rate, _ = mcs_tables.get_cqi(cqi, cqi_table)
        reqs.append(required_snr_db(qm, rate) + 1.0 + 0.6 * rate + margin)
    return np.array(reqs)


def _sinr_to_cqi(eff_sinr_db: float, cqi_table: int, target_bler: float) -> int:
    """Highest CQI whose required SNR (for target BLER) <= effective SINR."""
    reqs = _cqi_required_snr(cqi_table, target_bler)
    # CQIs must be satisfied contiguously from index 1 upward
    best = 0
    for i in range(len(reqs)):
        if eff_sinr_db >= reqs[i]:
            best = i + 1
        else:
            break
    return best


class CSIFeedbackChannel:
    """Models CSI report delay by buffering reports for ``delay`` slots."""

    def __init__(self, delay_slots: int):
        self.delay = max(0, delay_slots)
        self.buffer: deque = deque()
        self.latest: CSIReport | None = None

    def push(self, report: CSIReport):
        self.buffer.append(report)

    def get(self) -> CSIReport | None:
        if len(self.buffer) > self.delay:
            self.latest = self.buffer.popleft()
        return self.latest

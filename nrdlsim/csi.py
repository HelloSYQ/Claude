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

from dataclasses import dataclass, field
from collections import deque
import numpy as np

from . import mcs_tables
from . import tbs as tbs_mod
from .link_abstraction import effective_sinr_miesm, required_eff_sinr_db
from .layer_mapping import svd_precoder
from .receiver import batch_mmse_sinr


@dataclass
class CSIReport:
    rank: int
    cqi: int
    precoder: np.ndarray            # W[f, n_tx, rank] (unit-norm columns)
    eff_sinr_db: float
    # precoders for every evaluated rank, so a HARQ retransmission that must
    # keep its original rank can still use this (delayed) report's precoder
    precoders: dict = field(default_factory=dict)


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


# CSI reference resource used when the caller does not supply one:
# 13 PDSCH symbols minus 12 DM-RS RE per PRB (TS 38.214 5.2.2.5).
DEFAULT_REF_RE_PER_RB = 144


def compute_csi(H_freq: np.ndarray, noise_var: float, max_rank: int,
                mcs_table: int, target_bler: float = 0.1,
                n_re_per_rb: int | None = None,
                n_rb: int | None = None) -> CSIReport:
    """Compute the best (rank, precoder, CQI) for the estimated channel.

    ``n_re_per_rb``/``n_rb`` describe the CSI reference resource; they set the
    TBS, hence the BLER-waterfall slope, used to place each CQI at the target
    BLER. They default to 144 RE/PRB over ``H_freq.shape[0]`` PRBs.
    """
    n_f, n_rx, n_tx = H_freq.shape
    max_rank = min(max_rank, n_rx, n_tx)
    cqi_table = mcs_tables.MCS_TO_CQI_TABLE[mcs_table]
    n_re_per_rb = n_re_per_rb or DEFAULT_REF_RE_PER_RB
    n_rb = n_rb or n_f
    qms = sorted({mcs_tables.get_cqi(i, cqi_table)[0] for i in range(1, 16)})

    best = None
    best_tput = -1.0
    precoders = {}
    for rank in range(1, max_rank + 1):
        W = svd_precoder(H_freq, rank)
        precoders[rank] = W
        # evaluate with the transmit power split across layers, exactly as
        # the gNB will transmit (W / sqrt(rank)); otherwise the SINR of a
        # rank-r hypothesis is overstated by 10*log10(r) dB
        H_eff = H_freq @ (W / np.sqrt(rank))    # [n_f, n_rx, rank]
        sinrs = batch_mmse_sinr(H_eff, noise_var).reshape(-1)
        # each CQI is judged with the MI curve of its own modulation order
        eff = {qm: effective_sinr_miesm(sinrs, qm) for qm in qms}
        reqs = cqi_required_sinr_db(cqi_table, rank, target_bler,
                                    n_re_per_rb, n_rb)
        cqi = 0
        for i in range(15, 0, -1):              # highest CQI meeting target
            qm_i = mcs_tables.get_cqi(i, cqi_table)[0]
            if eff[qm_i] >= reqs[i - 1]:
                cqi = i
                break
        _, _, se = mcs_tables.get_cqi(cqi, cqi_table)
        tput = rank * se
        if tput > best_tput:
            best_tput = tput
            qm_rep = mcs_tables.get_cqi(cqi, cqi_table)[0] if cqi else qms[0]
            best = CSIReport(rank, cqi, W, eff[qm_rep])
    best.precoders = precoders
    return best


from functools import lru_cache


@lru_cache(maxsize=None)
def cqi_required_sinr_db(cqi_table: int, rank: int, target_bler: float,
                         n_re_per_rb: int, n_rb: int) -> np.ndarray:
    """Effective SINR (dB) at which each CQI 1..15 meets ``target_bler``.

    Uses the TBS the CQI's (Qm, R) would give on the reference resource at
    this rank.  The gNB uses the same table to turn a reported CQI back into
    an SINR, so UE and gNB share one definition of the BLER target.
    """
    reqs = []
    for cqi in range(1, 16):
        qm, rate, _ = mcs_tables.get_cqi(cqi, cqi_table)
        tb = tbs_mod.compute_tbs(n_re_per_rb, n_rb, qm, rate, rank)
        reqs.append(required_eff_sinr_db(qm, rate, tb, target_bler))
    return np.array(reqs)


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

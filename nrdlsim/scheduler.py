"""Resource scheduling and MCS link adaptation.

The scheduler decides, per slot, the frequency-domain resource allocation
(number of PRBs), the transmission rank and the MCS index, driven by the most
recent (delayed) CSI report.

MCS selection works in the SINR domain: the reported CQI is turned back into
the effective SINR at which it meets the BLER target (the same table the UE
used to pick it), an outer-loop link-adaptation (OLLA) offset in dB is
subtracted, and the highest MCS whose own target-BLER SINR fits is chosen.
OLLA is driven by first-transmission HARQ ACK/NACKs and moves in both
directions, so it can correct a CQI that is either optimistic or pessimistic.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import numpy as np

from . import mcs_tables
from . import tbs as tbs_mod
from .csi import CSIReport, cqi_required_sinr_db, DEFAULT_REF_RE_PER_RB
from .link_abstraction import required_eff_sinr_db


@dataclass
class SchedulingDecision:
    num_rb: int
    start_rb: int
    num_layers: int
    mcs_index: int


@lru_cache(maxsize=None)
def _mcs_required_sinr_db(mcs_table: int, rank: int, target_bler: float,
                          n_re_per_rb: int, n_rb: int) -> np.ndarray:
    """Effective SINR (dB) at which each MCS index meets ``target_bler``."""
    reqs = []
    for idx in range(mcs_tables.num_mcs(mcs_table)):
        info = mcs_tables.get_mcs(idx, mcs_table)
        tb = tbs_mod.compute_tbs(n_re_per_rb, n_rb, info.modulation_order,
                                 info.target_code_rate, rank)
        reqs.append(required_eff_sinr_db(info.modulation_order,
                                         info.target_code_rate, tb, target_bler))
    return np.array(reqs)


class Scheduler:
    def __init__(self, total_rb: int, mcs_table: int, target_bler: float = 0.1,
                 link_adaptation: bool = True, olla_step_db: float = 0.5,
                 n_re_per_rb: int = DEFAULT_REF_RE_PER_RB,
                 olla_limit_db: float = 10.0):
        self.total_rb = total_rb
        self.mcs_table = mcs_table
        self.target_bler = target_bler
        self.link_adaptation = link_adaptation
        self.n_re_per_rb = n_re_per_rb
        self.cqi_table = mcs_tables.MCS_TO_CQI_TABLE[mcs_table]
        # OLLA back-off in dB subtracted from the CQI-implied SINR.
        # Positive = more conservative, negative = more aggressive.
        self.olla_offset = 0.0
        self.olla_limit_db = olla_limit_db
        # NACK raises the back-off by step_nack, ACK lowers it by step_ack.
        # Zero drift requires BLER*step_nack = (1-BLER)*step_ack, so this ratio
        # makes the long-run first-transmission BLER converge to the target.
        self.olla_step_nack = olla_step_db
        self.olla_step_ack = olla_step_db * target_bler / (1 - target_bler)

    def schedule(self, csi: CSIReport | None, fixed_mcs: int | None,
                 fixed_rank: int) -> SchedulingDecision:
        num_rb = self.total_rb
        if not self.link_adaptation or csi is None:
            rank = fixed_rank
            mcs = fixed_mcs if fixed_mcs is not None else 0
            return SchedulingDecision(num_rb, 0, rank, mcs)

        rank = max(1, csi.rank)
        if csi.cqi < 1:                     # CQI 0: out of range
            return SchedulingDecision(num_rb, 0, rank, 0)
        cqi_sinr = cqi_required_sinr_db(self.cqi_table, rank, self.target_bler,
                                        self.n_re_per_rb, num_rb)[csi.cqi - 1]
        mcs = self._sinr_to_mcs(cqi_sinr - self.olla_offset, rank)
        return SchedulingDecision(num_rb, 0, rank, mcs)

    def _sinr_to_mcs(self, sinr_db: float, rank: int) -> int:
        """Highest MCS whose target-BLER SINR does not exceed ``sinr_db``."""
        reqs = _mcs_required_sinr_db(self.mcs_table, rank, self.target_bler,
                                     self.n_re_per_rb, self.total_rb)
        ok = np.nonzero(reqs <= sinr_db + 1e-9)[0]
        return int(ok[-1]) if ok.size else 0

    def update_olla(self, ack: bool):
        """OLLA update from a first-transmission HARQ ACK/NACK."""
        if not self.link_adaptation:
            return
        self.olla_offset += -self.olla_step_ack if ack else self.olla_step_nack
        self.olla_offset = float(np.clip(self.olla_offset,
                                         -self.olla_limit_db, self.olla_limit_db))

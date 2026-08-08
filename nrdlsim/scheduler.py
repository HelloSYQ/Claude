"""Resource scheduling and MCS link adaptation.

The scheduler decides, per slot, the frequency-domain resource allocation
(number of PRBs), the transmission rank and the MCS index, driven by the most
recent (delayed) CSI report.  An outer-loop link-adaptation (OLLA) offset
tracks the ACK/NACK history to keep the operating BLER at the target, as in
TS 38.214-based scheduler implementations.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from . import mcs_tables
from .csi import CSIReport


@dataclass
class SchedulingDecision:
    num_rb: int
    start_rb: int
    num_layers: int
    mcs_index: int


class Scheduler:
    def __init__(self, total_rb: int, mcs_table: int, target_bler: float = 0.1,
                 link_adaptation: bool = True, olla_step_up: float = 0.1):
        self.total_rb = total_rb
        self.mcs_table = mcs_table
        self.target_bler = target_bler
        self.link_adaptation = link_adaptation
        # OLLA: subtract olla_offset (in CQI-eff units) when NACKs pile up
        self.olla_offset = 0.0
        self.olla_step_up = olla_step_up
        # step_down chosen so long-run BLER converges to target
        self.olla_step_down = olla_step_up * (1 - target_bler) / target_bler

    def schedule(self, csi: CSIReport | None, fixed_mcs: int | None,
                 fixed_rank: int) -> SchedulingDecision:
        num_rb = self.total_rb
        if not self.link_adaptation or csi is None:
            rank = fixed_rank
            mcs = fixed_mcs if fixed_mcs is not None else 0
            return SchedulingDecision(num_rb, 0, rank, mcs)

        rank = max(1, csi.rank)
        # map reported CQI to MCS, then apply OLLA back-off
        cqi_table = mcs_tables.MCS_TO_CQI_TABLE[self.mcs_table]
        _, _, eff = mcs_tables.get_cqi(max(csi.cqi, 1), cqi_table)
        eff_adj = max(eff - self.olla_offset, 0.0)
        mcs = self._eff_to_mcs(eff_adj)
        return SchedulingDecision(num_rb, 0, rank, mcs)

    def _eff_to_mcs(self, target_eff: float) -> int:
        best = 0
        for idx in range(mcs_tables.num_mcs(self.mcs_table)):
            info = mcs_tables.get_mcs(idx, self.mcs_table)
            if info.spectral_efficiency <= target_eff + 1e-9:
                best = idx
            else:
                break
        return best

    def update_olla(self, ack: bool):
        """Outer-loop link adaptation update from HARQ ACK/NACK feedback."""
        if not self.link_adaptation:
            return
        if ack:
            self.olla_offset = max(0.0, self.olla_offset - self.olla_step_up)
        else:
            self.olla_offset += self.olla_step_down

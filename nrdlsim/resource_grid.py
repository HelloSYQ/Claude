"""Resource grid construction and DM-RS resource-element accounting.

Follows the PDSCH resource mapping in TS 38.211 clause 7.3.1.6 and the
DM-RS structure in clause 7.4.1.1.  The grid is the standard NR
[subcarrier x OFDM symbol x antenna port] container.
"""

from __future__ import annotations

import numpy as np

from .config import CarrierConfig, PDSCHConfig, DMRSConfig


class ResourceGrid:
    """A single-slot NR resource grid for one antenna layer/port."""

    def __init__(self, carrier: CarrierConfig):
        self.carrier = carrier
        self.n_sc = carrier.n_subcarriers
        self.n_sym = carrier.symbols_per_slot
        self.grid = np.zeros((self.n_sc, self.n_sym), dtype=complex)

    def reset(self):
        self.grid[:] = 0.0


def dmrs_symbol_positions(pdsch: PDSCHConfig) -> list[int]:
    """OFDM symbols (relative to slot) that carry PDSCH DM-RS.

    Simplified single-front-load mapping (TS 38.211 Table 7.4.1.1.2-3):
    first DM-RS at the PDSCH start, additional DM-RS spread across the
    allocation.  Returns absolute symbol indices in the slot.
    """
    l0 = pdsch.start_symbol
    dur = pdsch.num_symbols
    add = pdsch.dmrs.additional_positions
    positions = [l0]
    if add >= 1 and dur >= 8:
        # spread additional DM-RS roughly evenly across the allocation
        for k in range(1, add + 1):
            positions.append(l0 + int(round(k * (dur - 1) / (add + 1))))
    return sorted(set(positions))


def dmrs_re_per_rb(pdsch: PDSCHConfig) -> int:
    """Total DM-RS resource elements per PRB across the whole allocation.

    Accounts for the CDM groups without data (those subcarriers cannot
    carry PDSCH even when they are not used for DM-RS on this port).
    """
    dmrs = pdsch.dmrs
    n_dmrs_sym = len(dmrs_symbol_positions(pdsch))
    if dmrs.config_type == 1:
        re_per_group = 6            # 6 subcarriers per CDM group column pattern
    else:
        re_per_group = 4
    cdm_groups = dmrs.num_cdm_groups_without_data
    if dmrs.config_type == 1:
        occupied_sc = min(12, re_per_group * cdm_groups)
    else:
        occupied_sc = min(12, re_per_group * cdm_groups)
    return occupied_sc * n_dmrs_sym


def data_re_per_rb(pdsch: PDSCHConfig) -> int:
    """PDSCH data resource elements per PRB in the slot."""
    total = 12 * pdsch.num_symbols
    return total - dmrs_re_per_rb(pdsch) - pdsch.n_oh


def num_data_re(pdsch: PDSCHConfig) -> int:
    """Total PDSCH data REs across all allocated PRBs (single layer)."""
    return data_re_per_rb(pdsch) * pdsch.num_rb


def dmrs_overhead_fraction(pdsch: PDSCHConfig) -> float:
    total = 12 * pdsch.num_symbols
    return dmrs_re_per_rb(pdsch) / total

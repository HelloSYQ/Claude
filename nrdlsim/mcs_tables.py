"""MCS and CQI tables from 3GPP TS 38.214.

- MCS Table 1  : TS 38.214 Table 5.1.3.1-1  (up to 64QAM)
- MCS Table 2  : TS 38.214 Table 5.1.3.1-2  (up to 256QAM)
- MCS Table 3  : TS 38.214 Table 5.1.3.1-3  (low spectral efficiency / URLLC)
- CQI Table 1  : TS 38.214 Table 5.2.2.1-2  (up to 64QAM)
- CQI Table 2  : TS 38.214 Table 5.2.2.1-3  (up to 256QAM)

Each MCS entry is (modulation order Qm, target code rate x1024).
Each CQI entry is (modulation order Qm, target code rate x1024, spectral eff).
"""

from __future__ import annotations

from dataclasses import dataclass

# ---------------------------------------------------------------------------
# MCS tables: index -> (Qm, R x 1024)
# ---------------------------------------------------------------------------

MCS_TABLE_1 = [
    (2, 120), (2, 157), (2, 193), (2, 251), (2, 308), (2, 379), (2, 449),
    (2, 526), (2, 602), (2, 679), (4, 340), (4, 378), (4, 434), (4, 490),
    (4, 553), (4, 616), (4, 658), (6, 438), (6, 466), (6, 517), (6, 567),
    (6, 616), (6, 666), (6, 719), (6, 772), (6, 822), (6, 873), (6, 910),
    (6, 948),
]

MCS_TABLE_2 = [
    (2, 120), (2, 193), (2, 308), (2, 449), (2, 602), (4, 378), (4, 434),
    (4, 490), (4, 553), (4, 616), (4, 658), (6, 466), (6, 517), (6, 567),
    (6, 616), (6, 666), (6, 719), (6, 772), (6, 822), (6, 873), (8, 682.5),
    (8, 711), (8, 754), (8, 797), (8, 841), (8, 885), (8, 916.5), (8, 948),
]

MCS_TABLE_3 = [
    (2, 30), (2, 40), (2, 50), (2, 64), (2, 78), (2, 99), (2, 120), (2, 157),
    (2, 193), (2, 251), (2, 308), (2, 379), (2, 449), (2, 526), (2, 602),
    (4, 340), (4, 378), (4, 434), (4, 490), (4, 553), (4, 616), (6, 438),
    (6, 466), (6, 517), (6, 567), (6, 616), (6, 666), (6, 719), (6, 772),
]

_MCS_TABLES = {1: MCS_TABLE_1, 2: MCS_TABLE_2, 3: MCS_TABLE_3}

# ---------------------------------------------------------------------------
# CQI tables: index -> (Qm, R x 1024, spectral efficiency)
# Index 0 is "out of range".
# ---------------------------------------------------------------------------

CQI_TABLE_1 = [
    (0, 0, 0.0),
    (2, 78, 0.1523), (2, 120, 0.2344), (2, 193, 0.3770), (2, 308, 0.6016),
    (2, 449, 0.8770), (2, 602, 1.1758), (4, 378, 1.4766), (4, 490, 1.9141),
    (4, 616, 2.4063), (6, 466, 2.7305), (6, 567, 3.3223), (6, 666, 3.9023),
    (6, 772, 4.5234), (6, 873, 5.1152), (6, 948, 5.5547),
]

CQI_TABLE_2 = [
    (0, 0, 0.0),
    (2, 78, 0.1523), (2, 193, 0.3770), (2, 449, 0.8770), (4, 378, 1.4766),
    (4, 490, 1.9141), (4, 616, 2.4063), (6, 466, 2.7305), (6, 567, 3.3223),
    (6, 666, 3.9023), (6, 772, 4.5234), (6, 873, 5.1152), (8, 711, 5.5547),
    (8, 797, 6.2266), (8, 885, 6.9141), (8, 948, 7.4063),
]

_CQI_TABLES = {1: CQI_TABLE_1, 2: CQI_TABLE_2}

# Map MCS table -> associated CQI table for link adaptation.
MCS_TO_CQI_TABLE = {1: 1, 2: 2, 3: 1}


@dataclass
class MCSInfo:
    index: int
    modulation_order: int      # bits per QAM symbol (Qm)
    target_code_rate: float    # R (0..1)
    modulation_name: str
    spectral_efficiency: float  # Qm * R (bits/RE, before overhead)


_MOD_NAME = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}


def get_mcs(index: int, table: int = 2) -> MCSInfo:
    """Return modulation order and code rate for an MCS index."""
    tbl = _MCS_TABLES[table]
    if not 0 <= index < len(tbl):
        raise ValueError(f"MCS index {index} out of range for table {table}")
    qm, r1024 = tbl[index]
    r = r1024 / 1024.0
    return MCSInfo(index, qm, r, _MOD_NAME[qm], qm * r)


def num_mcs(table: int = 2) -> int:
    return len(_MCS_TABLES[table])


def get_cqi(index: int, table: int = 1):
    """Return (Qm, code_rate, spectral_efficiency) for a CQI index."""
    tbl = _CQI_TABLES[table]
    qm, r1024, eff = tbl[index]
    return qm, r1024 / 1024.0, eff


def cqi_to_mcs(cqi_index: int, mcs_table: int = 2) -> int:
    """Map a reported CQI to an MCS index with a matching spectral efficiency.

    Chooses the highest MCS whose (Qm * R) does not exceed the CQI efficiency.
    This mirrors the outer-loop-free link adaptation used by many schedulers.
    """
    cqi_table = MCS_TO_CQI_TABLE[mcs_table]
    _, _, target_eff = get_cqi(max(cqi_index, 1), cqi_table)
    best = 0
    for idx in range(num_mcs(mcs_table)):
        info = get_mcs(idx, mcs_table)
        if info.spectral_efficiency <= target_eff + 1e-9:
            best = idx
        else:
            break
    return best

"""Transport block size determination (TS 38.214 clause 5.1.3.2).

Implements the full NR TBS procedure including the <=3824 bit quantisation
table and the >3824 bit closed-form formula.
"""

from __future__ import annotations

import math

# TBS quantisation table for N_info <= 3824 (TS 38.214 Table 5.1.3.2-1)
_TBS_TABLE = [
    24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144,
    152, 160, 168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320,
    336, 352, 368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640,
    672, 704, 736, 768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160,
    1192, 1224, 1256, 1288, 1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736,
    1800, 1864, 1928, 2024, 2088, 2152, 2216, 2280, 2408, 2472, 2536, 2600,
    2664, 2728, 2792, 2856, 2976, 3104, 3240, 3368, 3496, 3624, 3752, 3824,
]


def _ceil_pow2_quant(n_info: float) -> int:
    n = max(3, math.floor(math.log2(n_info)) - 6)
    return int(2 ** n * round(n_info / (2 ** n)))


def re_per_rb(n_pdsch_symbols: int, n_dmrs_re_per_rb: int, n_oh: int) -> int:
    """N'_RE per PRB, capped at 156 (TS 38.214 5.1.3.2 step 1-2)."""
    n_re_prime = 12 * n_pdsch_symbols - n_dmrs_re_per_rb - n_oh
    return min(156, n_re_prime)


def compute_tbs(n_re_per_rb: int, n_prb: int, qm: int, code_rate: float,
                num_layers: int) -> int:
    """Return transport block size in bits following TS 38.214 5.1.3.2."""
    n_re = n_re_per_rb * n_prb
    n_info = n_re * code_rate * qm * num_layers

    if n_info <= 3824:
        n_info_q = max(24, _ceil_pow2_quant(n_info))
        # find smallest TBS >= n_info_q
        for tbs in _TBS_TABLE:
            if tbs >= n_info_q:
                return tbs
        return _TBS_TABLE[-1]

    # N_info > 3824
    n = math.floor(math.log2(n_info - 24)) - 5
    n_info_q = max(3840, 2 ** n * round((n_info - 24) / (2 ** n)))

    if code_rate <= 0.25:
        c = math.ceil((n_info_q + 24) / 3816)
        tbs = 8 * c * math.ceil((n_info_q + 24) / (8 * c)) - 24
    else:
        if n_info_q > 8424:
            c = math.ceil((n_info_q + 24) / 8424)
            tbs = 8 * c * math.ceil((n_info_q + 24) / (8 * c)) - 24
        else:
            tbs = 8 * math.ceil((n_info_q + 24) / 8) - 24
    return int(tbs)

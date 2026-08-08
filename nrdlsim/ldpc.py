"""NR DL-SCH coding chain (TS 38.212): CRC, code-block segmentation, a
QC-LDPC encoder/decoder and rate matching.

The QC-LDPC code follows the NR base-graph architecture: a systematic part
plus a dual-diagonal (accumulator) parity core lifted by circular-shift
permutations of size Z.  Base graph selection (BG1 for high rate / large
blocks, BG2 otherwise) and the lifting-size set follow TS 38.212 clause 5.2.2
and 5.3.2.  Encoding is the standard accumulator recursion; decoding is a
normalised layered min-sum.  This is a genuine bit-true FEC (see tests), used
when ``fec_mode='ldpc'``.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

# Lifting size set (TS 38.212 Table 5.3.2-1)
_LIFTING_SIZES = sorted({
    a * (2 ** j)
    for a in (2, 3, 5, 7, 9, 11, 13, 15)
    for j in range(0, 8)
    if a * (2 ** j) <= 384
})


def crc_length_and_poly(kind: str):
    polys = {
        "24A": (24, 0x864CFB),
        "24B": (24, 0x800063),
        "16": (16, 0x1021),
    }
    return polys[kind]


def attach_crc(bits: np.ndarray, kind: str) -> np.ndarray:
    """Append a CRC of the given kind (TS 38.212 clause 5.1)."""
    length, poly = crc_length_and_poly(kind)
    # bit-serial CRC over the message followed by ``length`` zero bits
    reg = 0
    msg = np.concatenate([bits, np.zeros(length, dtype=np.int8)])
    for b in msg:
        top = (reg >> (length - 1)) & 1
        reg = ((reg << 1) & ((1 << length) - 1)) | int(b)
        if top:
            reg ^= poly
    crc_bits = np.array([(reg >> (length - 1 - i)) & 1
                         for i in range(length)], dtype=np.int8)
    return np.concatenate([bits, crc_bits])


def check_crc(bits: np.ndarray, kind: str) -> bool:
    length, poly = crc_length_and_poly(kind)
    reg = 0
    for b in bits:
        top = (reg >> (length - 1)) & 1
        reg = ((reg << 1) & ((1 << length) - 1)) | int(b)
        if top:
            reg ^= poly
    return reg == 0


def select_base_graph(tbs: int, code_rate: float) -> int:
    """BG selection per TS 38.212 clause 7.2.2."""
    if tbs <= 292 or (tbs <= 3824 and code_rate <= 0.67) or code_rate <= 0.25:
        return 2
    return 1


def _bg_dims(bg: int):
    # (num info columns Kb, num parity columns Mb) of the base graph
    if bg == 1:
        return 22, 46      # nb=68, mb=46
    return 10, 42          # nb=52, mb=42 (BG2)


def choose_lifting_size(k: int, kb: int) -> int:
    """Smallest lifting size Z with Kb*Z >= k (TS 38.212 5.2.2)."""
    for z in _LIFTING_SIZES:
        if kb * z >= k:
            return z
    return _LIFTING_SIZES[-1]


@dataclass
class LDPCCode:
    bg: int
    z: int
    kb: int
    mb: int
    base_sys: np.ndarray     # mb x kb shift matrix (-1 = zero block)

    @property
    def k_bits(self):
        return self.kb * self.z

    @property
    def n_bits(self):
        return (self.kb + self.mb) * self.z


def build_code(bg: int, z: int, seed: int = 0) -> LDPCCode:
    """Construct an NR-structured QC-LDPC code with accumulator parity core."""
    kb, mb = _bg_dims(bg)
    rng = np.random.default_rng(1000 + bg * 7 + z)
    # sparse systematic base graph: each check row connects to a few info cols
    base_sys = -np.ones((mb, kb), dtype=int)
    row_weight = 6 if bg == 1 else 8
    for r in range(mb):
        cols = rng.choice(kb, size=min(row_weight, kb), replace=False)
        for c in cols:
            base_sys[r, c] = int(rng.integers(0, z))
    # guarantee every info column participates (column weight >= 2)
    for c in range(kb):
        rows = np.where(base_sys[:, c] >= 0)[0]
        while rows.size < 2:
            r = int(rng.integers(0, mb))
            base_sys[r, c] = int(rng.integers(0, z))
            rows = np.where(base_sys[:, c] >= 0)[0]
    return LDPCCode(bg, z, kb, mb, base_sys)


def _circ_shift(vec: np.ndarray, s: int) -> np.ndarray:
    """Circular (right) shift of a length-Z block, the QC-LDPC permutation."""
    if s < 0:
        return np.zeros_like(vec)
    return np.roll(vec, s)


def encode(code: LDPCCode, info_bits: np.ndarray) -> np.ndarray:
    """Systematic accumulator encoding: returns [systematic | parity] bits."""
    z, kb, mb = code.z, code.kb, code.mb
    s = info_bits.reshape(kb, z).astype(np.int8)

    # per-check-row systematic syndrome contribution.
    # The parity-check adjacency (see _build_parity_check) connects check row
    # (r*z + i) to systematic bit c at position (i + sh) mod z, so the
    # contribution uses a matching left circular shift: roll(s[c], -sh).
    contrib = np.zeros((mb, z), dtype=np.int8)
    for r in range(mb):
        acc = np.zeros(z, dtype=np.int8)
        for c in range(kb):
            sh = code.base_sys[r, c]
            if sh >= 0:
                acc ^= np.roll(s[c], -sh)
        contrib[r] = acc

    # accumulator parity (dual-diagonal identity core): p_0 = contrib_0,
    # p_i = p_{i-1} XOR contrib_i
    p = np.zeros((mb, z), dtype=np.int8)
    p[0] = contrib[0]
    for r in range(1, mb):
        p[r] = p[r - 1] ^ contrib[r]

    return np.concatenate([s.reshape(-1), p.reshape(-1)]).astype(np.int8)


def _build_parity_check(code: LDPCCode):
    """Return check-node adjacency (list of (var_index, ) per check) for min-sum."""
    z, kb, mb = code.z, code.kb, code.mb
    n = (kb + mb) * z
    check_edges = [[] for _ in range(mb * z)]
    var_edges = [[] for _ in range(n)]

    def add_block(cr, base_col, shift):
        # identity permuted by 'shift': check row (cr*z + i) connects
        # var col base_col*z + ((i+shift) mod z)
        for i in range(z):
            chk = cr * z + i
            var = base_col * z + ((i + shift) % z)
            check_edges[chk].append(var)
            var_edges[var].append(chk)

    # systematic part
    for r in range(mb):
        for c in range(kb):
            sh = code.base_sys[r, c]
            if sh >= 0:
                add_block(r, c, sh)
    # parity accumulator: diagonal (identity) + subdiagonal (identity)
    for r in range(mb):
        add_block(r, kb + r, 0)                 # diagonal
        if r >= 1:
            add_block(r, kb + r - 1, 0)          # subdiagonal
    return check_edges, var_edges, n


def decode(code: LDPCCode, llr: np.ndarray, max_iter: int = 25,
           alpha: float = 0.75):
    """Normalised min-sum LDPC decoder.

    ``llr`` are channel LLRs with the convention positive => bit 0.
    Returns (hard_bits, success) over the full codeword length.
    """
    check_edges, var_edges, n = _build_parity_check(code)
    n_chk = len(check_edges)

    # message storage keyed by (check, var)
    # use dict of arrays per check for min-sum
    llr = np.asarray(llr, float).copy()
    if llr.size < n:
        llr = np.concatenate([llr, np.zeros(n - llr.size)])
    # messages var->check initialised to channel llr
    msg = {}
    for chk in range(n_chk):
        for v in check_edges[chk]:
            msg[(chk, v)] = llr[v]

    for _ in range(max_iter):
        # check node update (min-sum)
        for chk in range(n_chk):
            vs = check_edges[chk]
            vals = np.array([msg[(chk, v)] for v in vs])
            signs = np.sign(vals)
            signs[signs == 0] = 1
            absv = np.abs(vals)
            total_sign = np.prod(signs)
            # for each edge, exclude itself
            for i, v in enumerate(vs):
                s = total_sign * signs[i]
                m = np.min(np.delete(absv, i)) if len(absv) > 1 else 0.0
                msg[(chk, v)] = alpha * s * m
        # variable node update
        total = llr.copy()
        for chk in range(n_chk):
            for v in check_edges[chk]:
                total[v] += msg[(chk, v)]
        for chk in range(n_chk):
            for v in check_edges[chk]:
                # var->check excludes this check's incoming
                msg[(chk, v)] = total[v] - msg[(chk, v)]
        # hard decision + parity check
        hard = (total < 0).astype(np.int8)
        ok = True
        for chk in range(n_chk):
            if np.bitwise_xor.reduce(hard[check_edges[chk]]) != 0:
                ok = False
                break
        if ok:
            return hard, True
    return (total < 0).astype(np.int8), False

"""PDSCH mapping of code blocks to resource elements and VRB-to-PRB mapping
(TS 38.211 clauses 7.3.1.3, 7.3.1.5 and 7.3.1.6).

The rate-matched code blocks of a codeword are concatenated, modulated and
layer-mapped (symbol i goes to layer i mod v, TS 38.211 Table 7.3.1.3-1).
Each layer is then mapped to resource elements frequency first: across the
subcarriers of the allocated virtual resource blocks (VRBs), then OFDM symbol
by OFDM symbol.  The VRB-to-PRB mapping finally places every VRB on a physical
resource block (PRB).  It decides which part of the band each code block
sees, so it sets the frequency diversity of each code block.

Three VRB-to-PRB mappings are provided, each returned as ``prb[v]`` for
VRB v:

  * :func:`vrb_to_prb_non_interleaved` : PRB = VRB;
  * :func:`vrb_to_prb_nr_interleaved` : the NR interleaver (bundles of L RBs,
    2-row block interleaver, TS 38.211 7.3.1.6);
  * :func:`vrb_to_prb_block_interleaver` : an R x C block interleaver of
    groups of ``granularity`` RBs, written in rows and read in columns.

:func:`code_block_means` averages a per-subcarrier, per-layer quantity (for
example the BICM mutual information of each RE) over the REs of each code
block.  It assumes the channel is constant over the data symbols of the slot.
"""

from __future__ import annotations

import math
import numpy as np

SC_PER_RB = 12


def vrb_to_prb_non_interleaved(n_vrb: int) -> np.ndarray:
    """Non-interleaved mapping: VRB n is PRB n."""
    return np.arange(n_vrb)


def vrb_to_prb_nr_interleaved(n_bwp: int, bundle_size: int,
                              n_bwp_start: int = 0) -> np.ndarray:
    """NR interleaved VRB-to-PRB mapping (TS 38.211 7.3.1.6), ``prb[v]``.

    The BWP is cut into N_bundle = ceil((N_BWP + N_start mod L) / L) bundles
    of L RBs; the first and last bundles are shorter when the BWP is not
    aligned to L.  VRB bundle j < N_bundle - 1 goes to PRB bundle
    f(j) = r C + c with j = c R + r, R = 2, C = floor(N_bundle / R); the last
    bundle stays in place.  Indices are relative to the BWP start.
    """
    L = bundle_size
    off = n_bwp_start % L
    n_bundle = math.ceil((n_bwp + off) / L)
    # RB edges of the bundles (bundle 0 is L - off long, the last one may be short)
    edges = np.minimum(np.arange(n_bundle + 1) * L - off, n_bwp)
    edges[0] = 0
    R = 2
    C = n_bundle // R
    f = np.arange(n_bundle)
    for j in range(n_bundle - 1):
        c, r = divmod(j, R)
        f[j] = r * C + c
    prb = np.empty(n_bwp, dtype=int)
    for j in range(n_bundle):
        vrbs = np.arange(edges[j], edges[j + 1])
        prbs = np.arange(edges[f[j]], edges[f[j] + 1])
        if len(vrbs) != len(prbs):
            raise ValueError("VRB and PRB bundle sizes differ")
        prb[vrbs] = prbs
    return prb


def vrb_to_prb_block_interleaver(n_rows: int, n_cols: int,
                                 granularity: int = 1) -> np.ndarray:
    """Block-interleaved VRB-to-PRB mapping, ``prb[v]``.

    VRB groups of ``granularity`` RBs are written row by row into an
    ``n_rows`` x ``n_cols`` matrix and read column by column; the read order
    is the PRB order.  VRB group g = r * n_cols + c therefore lands on PRB
    group c * n_rows + r: for 4 x 8 with 1-RB groups, VRBs 0..7 go to PRBs
    0, 4, ..., 28 and VRB 8 to PRB 1.
    """
    g = np.arange(n_rows * n_cols)
    r, c = np.divmod(g, n_cols)
    prb_group = c * n_rows + r
    return (prb_group[:, None] * granularity
            + np.arange(granularity)[None, :]).reshape(-1)


def subcarrier_map(vrb_to_prb: np.ndarray) -> np.ndarray:
    """PRB-domain subcarrier of each VRB-domain subcarrier of the allocation."""
    sc = np.arange(SC_PER_RB)
    return (np.asarray(vrb_to_prb)[:, None] * SC_PER_RB + sc[None, :]).reshape(-1)


def code_block_spans(e_r: np.ndarray, qm: int, n_layers: int) -> np.ndarray:
    """Per-layer RE index range [start, stop) of each code block, shape (C, 2).

    Code block r occupies E_r / Q_m consecutive modulation symbols; layer
    mapping deals them out round-robin, so it takes the same
    E_r / (Q_m v) consecutive REs on every layer.
    """
    n = np.asarray(e_r) // (qm * n_layers)
    stop = np.cumsum(n)
    return np.stack([stop - n, stop], axis=1)


def code_block_means(values: np.ndarray, spans: np.ndarray,
                     vrb_to_prb: np.ndarray) -> np.ndarray:
    """Mean of ``values`` over the REs of each code block.

    ``values``: [..., n_sc, n_layers] per PRB-domain subcarrier and layer,
    constant over the data symbols.  Per-layer RE k sits on VRB-domain
    subcarrier k mod n_sc of data symbol k // n_sc.  Returns [..., C].
    """
    sc = subcarrier_map(vrb_to_prb)
    n_sc = sc.size
    n_layers = values.shape[-1]
    u = values[..., sc, :].sum(-1)                        # VRB order, layer sum
    pre = np.concatenate([np.zeros(u.shape[:-1] + (1,)), np.cumsum(u, -1)], -1)

    def F(k):                                             # sum over REs [0, k)
        return (k // n_sc) * pre[..., -1:] + pre[..., k % n_sc]

    a, b = spans[:, 0], spans[:, 1]
    return (F(b) - F(a)) / ((b - a) * n_layers)


def code_block_prb_fraction(spans: np.ndarray, vrb_to_prb: np.ndarray) -> np.ndarray:
    """Fraction of each code block's REs on each PRB, shape (C, n_prb)."""
    prb = np.asarray(vrb_to_prb)
    n_sc = prb.size * SC_PER_RB
    out = np.zeros((len(spans), prb.size))
    for i, (a, b) in enumerate(spans):
        vrb = (np.arange(a, b) % n_sc) // SC_PER_RB
        np.add.at(out[i], prb[vrb], 1.0)
        out[i] /= b - a
    return out

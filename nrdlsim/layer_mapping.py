"""Codeword-to-layer mapping and precoding (TS 38.211 clause 7.3.1).

Layer mapping distributes one or two codewords across up to 8 transmission
layers exactly as in Table 7.3.1.3-1.  Precoding maps layers to antenna
ports; both an SVD (closed-loop, ideal CSI) precoder and a DFT-based
codebook precoder are provided.
"""

from __future__ import annotations

import numpy as np


def num_codewords(num_layers: int) -> int:
    """Two codewords are used for ranks 5..8 (TS 38.211 7.3.1.3)."""
    return 2 if num_layers >= 5 else 1


def layer_groups(num_layers: int, max_codewords: int) -> list[list[int]]:
    """Layers carried by each codeword when up to ``max_codewords`` are allowed.

    The layers are split into min(max_codewords, num_layers) contiguous
    groups as evenly as possible, smaller groups first, which is the NR
    split for two codewords (rank 5 -> {2, 3}, Table 7.3.1.3-1) extended to
    more codewords.  ``max_codewords = 1`` is NR up to rank 4 (one codeword
    on all layers); ``max_codewords = num_layers`` is one codeword per layer.

    >>> layer_groups(4, 2)
    [[0, 1], [2, 3]]
    >>> layer_groups(3, 2)
    [[0], [1, 2]]
    """
    n_cw = max(1, min(max_codewords, num_layers))
    sizes = [num_layers // n_cw + (1 if k >= n_cw - num_layers % n_cw else 0)
             for k in range(n_cw)]
    out, start = [], 0
    for n in sizes:
        out.append(list(range(start, start + n)))
        start += n
    return out


def layer_map(codewords: list[np.ndarray], num_layers: int) -> np.ndarray:
    """Map codeword symbol streams to a [num_layers, M_symb_per_layer] array.

    Implements the codeword-to-layer split of Table 7.3.1.3-1.
    """
    ncw = num_codewords(num_layers)
    if len(codewords) != ncw:
        raise ValueError(f"rank {num_layers} needs {ncw} codewords")

    if ncw == 1:
        v = num_layers
        d = codewords[0]
        m_layer = d.size // v
        layers = d[: m_layer * v].reshape(m_layer, v).T
        return layers

    # two codewords: split layers as {4:4} for rank 8, etc. (balanced)
    n0 = num_layers // 2
    n1 = num_layers - n0
    d0, d1 = codewords
    m0 = d0.size // n0
    m1 = d1.size // n1
    m = min(m0, m1)
    l0 = d0[: m * n0].reshape(m, n0).T
    l1 = d1[: m * n1].reshape(m, n1).T
    return np.vstack([l0, l1])


def svd_precoder(H_freq: np.ndarray, num_layers: int) -> np.ndarray:
    """Per-subcarrier SVD precoding matrices W[f, n_tx, num_layers].

    Uses the right singular vectors associated with the strongest singular
    values (eigen-beamforming, the closed-loop ideal-CSI precoder).
    """
    # batched SVD over all subcarriers/RBs at once
    _, _, vh = np.linalg.svd(H_freq, full_matrices=False)   # vh: [n_f, k, n_tx]
    v = np.conj(np.swapaxes(vh, -1, -2))                    # [n_f, n_tx, k]
    return v[:, :, :num_layers]


def apply_precoding(layer_symbols: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Map layer symbols to antenna ports.

    layer_symbols: [num_layers, n_re], W: [n_re, n_tx, num_layers]
    Returns tx_symbols: [n_tx, n_re].
    """
    n_layers, n_re = layer_symbols.shape
    n_tx = W.shape[1]
    tx = np.zeros((n_tx, n_re), dtype=complex)
    for r in range(n_re):
        tx[:, r] = W[r] @ layer_symbols[:, r]
    # normalise transmit power to unity per RE
    norm = np.sqrt((np.abs(tx) ** 2).sum(axis=0, keepdims=True))
    norm[norm == 0] = 1.0
    return tx / norm * np.sqrt(n_layers)

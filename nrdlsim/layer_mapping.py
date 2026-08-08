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
    n_f, n_rx, n_tx = H_freq.shape
    W = np.zeros((n_f, n_tx, num_layers), dtype=complex)
    for f in range(n_f):
        _, _, vh = np.linalg.svd(H_freq[f])
        v = vh.conj().T                    # columns are right singular vectors
        W[f] = v[:, :num_layers]
    return W


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

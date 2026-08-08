"""Receiver processing: MIMO channel estimation, MMSE equalisation and
per-resource-element post-equaliser SINR.

The equaliser inverts the effective per-subcarrier channel H_eff = H * W where
W is the transmit precoder, producing per-layer soft symbols and the
post-equaliser SINR that feeds both the LLR computation (bit-true path) and
the MIESM link abstraction (fast path).
"""

from __future__ import annotations

import numpy as np


def mmse_equalize(y: np.ndarray, H_eff: np.ndarray, noise_var: float):
    """MMSE equalise one RE.

    y: [n_rx] received, H_eff: [n_rx, n_layers].
    Returns (x_hat[n_layers], sinr[n_layers]).
    """
    n_rx, n_layers = H_eff.shape
    Hh = H_eff.conj().T
    A = Hh @ H_eff + noise_var * np.eye(n_layers)
    Ainv = np.linalg.inv(A)
    G = Ainv @ Hh                          # [n_layers, n_rx]
    x_hat = G @ y
    # per-layer SINR from the MMSE filter
    sinr = np.zeros(n_layers)
    for k in range(n_layers):
        gk = G[k]
        sig = np.abs(gk @ H_eff[:, k]) ** 2
        interf = sum(np.abs(gk @ H_eff[:, j]) ** 2
                     for j in range(n_layers) if j != k)
        noise = noise_var * np.real(gk @ gk.conj())
        sinr[k] = sig / (interf + noise + 1e-12)
    # bias-correct the MMSE estimate for use in LLRs
    return x_hat, sinr


def per_re_sinr(H_freq: np.ndarray, W: np.ndarray, noise_var: float,
                re_to_sc: np.ndarray) -> np.ndarray:
    """Compute post-MMSE SINR for every data RE.

    H_freq: [n_sc, n_rx, n_tx] channel, W: [n_sc, n_tx, n_layers] precoder,
    re_to_sc: subcarrier index of each data RE.  Returns flat SINR array
    of length len(re_to_sc)*n_layers (linear).
    """
    n_layers = W.shape[2]
    out = np.empty((len(re_to_sc), n_layers))
    # cache per unique subcarrier
    for i, sc in enumerate(re_to_sc):
        H_eff = H_freq[sc] @ W[sc]
        _, sinr = mmse_equalize(np.zeros(H_eff.shape[0]), H_eff, noise_var)
        out[i] = sinr
    return out.reshape(-1)


def estimate_channel_from_dmrs(H_true: np.ndarray, noise_var: float,
                               rng) -> np.ndarray:
    """Practical DM-RS channel estimate: true response + estimation noise.

    Models least-squares DM-RS estimation followed by frequency-domain
    averaging; the residual error variance scales with the pilot SNR.
    """
    n_sc, n_rx, n_tx = H_true.shape
    # DM-RS density gives roughly a 6x estimation SNR gain via averaging
    est_var = noise_var / 6.0
    err = np.sqrt(est_var / 2) * (rng.standard_normal(H_true.shape)
                                  + 1j * rng.standard_normal(H_true.shape))
    return H_true + err

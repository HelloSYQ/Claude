"""Receiver processing: MIMO channel estimation, MMSE equalisation and
per-resource-element post-equaliser SINR.

The equaliser inverts the effective per-subcarrier channel H_eff = H * W where
W is the transmit precoder, producing per-layer soft symbols and the
post-equaliser SINR that feeds both the LLR computation (bit-true path) and
the MIESM link abstraction (fast path).
"""

from __future__ import annotations

import numpy as np


def batch_mmse_sinr(H_eff: np.ndarray, noise_var: float) -> np.ndarray:
    """Post-MMSE per-layer SINR for a stack of effective channels.

    H_eff: [..., n_rx, n_layers] (batched over resource blocks / REs).
    Returns SINR [..., n_layers] (linear).  Vectorised equivalent of a per-RB
    ``mmse_equalize`` loop — uses batched matrix ops so the whole allocation is
    processed in a handful of BLAS calls.
    """
    n_layers = H_eff.shape[-1]
    Hh = np.conj(np.swapaxes(H_eff, -1, -2))          # [..., nl, nrx]
    A = Hh @ H_eff + noise_var * np.eye(n_layers)     # [..., nl, nl]
    G = np.linalg.solve(A, Hh)                        # (H^H H + N0 I)^-1 H^H
    GH = G @ H_eff                                    # [..., nl, nl]
    diag = np.abs(np.diagonal(GH, axis1=-2, axis2=-1)) ** 2       # signal
    total = np.sum(np.abs(GH) ** 2, axis=-1)          # signal + interference
    interf = total - diag
    noise = noise_var * np.sum(np.abs(G) ** 2, axis=-1)
    return diag / (interf + noise + 1e-12)


def batch_mmse_sinr_est(H_eff: np.ndarray, H_est: np.ndarray,
                        noise_var: float, extra_cov: np.ndarray | None = None
                        ) -> np.ndarray:
    """Post-MMSE per-layer SINR when the filter is built on a channel estimate.

    The LMMSE filter G = (Ĥ^H Ĥ + N0 I)^-1 Ĥ^H uses the estimate ``H_est``;
    signal, inter-layer leakage and noise are then evaluated on the true
    ``H_eff``.  With ``H_est = H_eff`` this equals :func:`batch_mmse_sinr`.
    ``extra_cov`` [..., n_rx, n_rx] is interference the filter does not know
    about (e.g. the residual of an imperfect cancellation); it is added to
    the noise.  Shapes as in :func:`batch_mmse_sinr`.
    """
    n_layers = H_eff.shape[-1]
    Eh = np.conj(np.swapaxes(H_est, -1, -2))
    G = np.linalg.solve(Eh @ H_est + noise_var * np.eye(n_layers), Eh)
    GH = G @ H_eff
    diag = np.abs(np.diagonal(GH, axis1=-2, axis2=-1)) ** 2
    interf = np.sum(np.abs(GH) ** 2, axis=-1) - diag
    noise = noise_var * np.sum(np.abs(G) ** 2, axis=-1)
    if extra_cov is not None:
        noise = noise + np.real(np.einsum("...kr,...rs,...ks->...k",
                                          np.conj(G), extra_cov, G))
    return diag / (interf + noise + 1e-12)


def cancelled_sinr(H_eff: np.ndarray, H_est: np.ndarray, noise_var: float,
                   cancelled) -> np.ndarray:
    """Per-layer MMSE SINR after the layers ``cancelled`` are removed.

    A cancelled layer is reconstructed from its decoded symbols and the
    channel estimate, so (H_eff - H_est) x of every cancelled layer stays in
    the signal as unknown interference.  Returns [..., n_layers] with NaN on
    the cancelled layers.  ``cancelled`` = () is plain MMSE.
    """
    n_layers = H_eff.shape[-1]
    c = sorted(set(cancelled))
    keep = [k for k in range(n_layers) if k not in c]
    out = np.full(H_eff.shape[:-2] + (n_layers,), np.nan)
    if not keep:
        return out
    extra = None
    if c:
        E = (H_eff - H_est)[..., c]
        if np.any(E):
            extra = E @ np.conj(np.swapaxes(E, -1, -2))
    out[..., keep] = batch_mmse_sinr_est(H_eff[..., keep], H_est[..., keep],
                                         noise_var, extra)
    return out


def rml_sinr_samples(H_eff: np.ndarray, H_est: np.ndarray, noise_var: float):
    """Per-layer SINR samples and weights of the (R-)ML receiver abstraction.

    A maximum-likelihood detector reaches the joint mutual information of
    all layers.  By the chain rule that is the sum of MMSE-SIC terms for any
    detection order; per layer this abstraction takes the term of each
    order, averaged over all orders.  Layer k's samples are the MMSE SINRs
    of k with every subset T of the other layers removed, with weight
    1 / (r * C(r-1, |T|)), so that sum_k E_w[log2(1 + SINR)] equals
    log2 det(I + H^H H / N0) for Gaussian inputs.  The BICM capacity
    applied per sample then caps each layer at its constellation.  Removed
    layers leave their channel-estimation error as interference.

    Returns (samples [..., r, n_s], weights [r, n_s]).  For r = 1 this is
    MMSE.  This models an ideal (R-)ML detector; a reduced-complexity
    implementation loses some of the gain.
    """
    from itertools import combinations
    from math import comb
    r = H_eff.shape[-1]
    cache = {}
    samples, weights = [], []
    for k in range(r):
        others = [j for j in range(r) if j != k]
        s_k, w_k = [], []
        for m in range(r):
            for T in combinations(others, m):
                if T not in cache:
                    cache[T] = cancelled_sinr(H_eff, H_est, noise_var, T)
                s_k.append(cache[T][..., k])
                w_k.append(1.0 / (r * comb(r - 1, m)))
        samples.append(np.stack(s_k, axis=-1))
        weights.append(w_k)
    return np.stack(samples, axis=-2), np.array(weights)


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
    # batched effective channel H_eff[sc] = H_freq[sc] @ W[sc]
    H_eff = H_freq[re_to_sc] @ W[re_to_sc]            # [n, n_rx, n_layers]
    return batch_mmse_sinr(H_eff, noise_var).reshape(-1)


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

"""Power-delay-profile (PDP) verification of the CDL channel generator.

A CDL frequency response is a sum of cluster exponentials,

    H(f) = sum_c H_c exp(-j 2 pi f tau_c),     tau_c = tau_c,norm * DS,

with E|H_c|^2 = P_c (TR 38.901 Tables 7.7.1-1..5).  This module checks the
*generated* channel against that definition in three independent ways:

1. Cluster powers recovered from H(f).  Sampling H(f) at random frequencies
   over a very wide synthetic band makes the exponentials of distinct delays
   nearly orthogonal, so least squares on the table delays recovers every
   H_c.  E|H_c|^2 must reproduce the table powers, and the LS residual must
   be ~0: all energy sits at the table delays scaled by DS.
2. Frequency correlation on a realistic band: E[H(f) H*(f+df)] must equal
   sum_c P_c exp(+j 2 pi df tau_c), the Fourier transform of the PDP.
3. The RMS delay spread of the recovered PDP must equal the configured DS,
   and for CDL-D/E the recovered Ricean K-factor must equal the table's.

Clusters that share a delay (the LOS ray and cluster 1 of CDL-D/E, clusters
3 and 5 of CDL-E) cannot be separated by delay, so their powers are compared
as one delay bin; the K-factor check separates the LOS ray instead by its
deterministic (non-zero-mean) phase.
"""

from __future__ import annotations

import numpy as np

from .channel_models import CDLChannel, _CDL


def table_pdp(model: str):
    """Normalised table PDP: (tau_norm, power) incl. the LOS ray, sum(power)=1."""
    spec = _CDL[model]
    cl = np.array(spec["clusters"], float)
    tau, p = cl[:, 0], 10 ** (cl[:, 1] / 10.0)
    if "los_power_db" in spec:
        tau = np.r_[0.0, tau]
        p = np.r_[10 ** (spec["los_power_db"] / 10.0), p]
    return tau, p / p.sum()


def merge_equal_delays(tau, p, tol=1e-12):
    """Sum the powers of entries with identical delay; returns sorted bins."""
    order = np.argsort(tau, kind="stable")
    tau, p = np.asarray(tau)[order], np.asarray(p)[order]
    keep = np.r_[True, np.diff(tau) > tol]
    idx = np.cumsum(keep) - 1
    return tau[keep], np.bincount(idx, weights=p)


def rms_delay_spread(tau, p) -> float:
    p = np.asarray(p, float) / np.sum(p)
    mean = np.sum(p * tau)
    return float(np.sqrt(max(np.sum(p * np.asarray(tau) ** 2) - mean ** 2, 0.0)))


def los_k_factor_db(model: str) -> float:
    """Table K-factor: LOS ray vs the Laplacian cluster at the same delay."""
    spec = _CDL[model]
    return spec["los_power_db"] - spec["clusters"][0][1]


def _drops(model, ds_ns, n_drops, seed, ch_kw):
    for d in range(n_drops):
        yield CDLChannel(model, ds_ns, 100.0, rng=np.random.default_rng(seed + d),
                         **ch_kw)


def estimate_cluster_pdp(model: str, ds_ns: float = 100.0, n_drops: int = 200,
                         n_times: int = 4, seed: int = 0, n_freq: int | None = None,
                         **ch_kw):
    """Recover the per-delay-bin PDP from generated frequency responses.

    Returns a dict with the delay bins (ns), the recovered normalised powers
    ``p_est``, the table powers ``p_ref`` of the same bins, the mean total
    power per port and the worst LS residual (fraction of energy not
    explained by the table delays).
    """
    ch_kw = {"n_tx": 2, "n_rx": 2, **ch_kw}
    tau_n, p_ref = merge_equal_delays(*table_pdp(model))
    tau_s = tau_n * ds_ns * 1e-9
    # synthetic band wide enough to resolve the closest pair of delay bins
    b_hz = 40.0 / np.min(np.diff(tau_s))
    n_freq = n_freq or max(400, 16 * tau_s.size)
    rng = np.random.default_rng(12345)
    f = rng.uniform(-b_hz / 2, b_hz / 2, n_freq)
    A = np.exp(-2j * np.pi * f[:, None] * tau_s[None, :])       # (f, bins)
    A_pinv = np.linalg.pinv(A)

    acc = np.zeros(tau_s.size)
    n_obs, worst_res = 0, 0.0
    times = np.arange(n_times) * 10e-3          # fd=100 Hz -> decorrelated
    for ch in _drops(model, ds_ns, n_drops, seed, ch_kw):
        for t in times:
            H = ch.frequency_response(f, t).reshape(n_freq, -1)   # (f, ports)
            C = A_pinv @ H                                        # (bins, ports)
            res = np.linalg.norm(H - A @ C) ** 2 / np.linalg.norm(H) ** 2
            worst_res = max(worst_res, res)
            acc += np.sum(np.abs(C) ** 2, axis=1)
            n_obs += H.shape[1]
    p_abs = acc / n_obs
    # p_abs: mean |H_c|^2 per port, not renormalised.  The generator fixes the
    # mean per-port power at 1, so p_abs should equal p_ref bin by bin; a
    # single wrong cluster then shows up in its own bin instead of being
    # spread over all bins by renormalisation.
    return dict(tau_ns=tau_n * ds_ns, p_est=p_abs / p_abs.sum(), p_abs=p_abs,
                p_ref=p_ref, total_power=float(p_abs.sum()),
                worst_residual=worst_res)


def estimate_k_factor_db(model: str, n_drops: int = 400, seed: int = 0) -> float:
    """Ricean K recovered from the delay-0 bin: |mean|^2 / variance.

    The LOS ray has the same phase in every drop while the co-delayed
    Laplacian cluster has random ray phases, so across drops the delay-0
    coefficient's mean is the LOS term and its variance the diffuse power.
    Single-polarised omni elements keep both couplings at unity.
    """
    tau_s = table_pdp(model)[0] * 100e-9
    tau_s = np.unique(tau_s)
    b_hz = 40.0 / np.min(np.diff(tau_s))
    f = np.random.default_rng(12345).uniform(-b_hz / 2, b_hz / 2,
                                             max(400, 16 * tau_s.size))
    A_pinv = np.linalg.pinv(np.exp(-2j * np.pi * f[:, None] * tau_s[None, :]))
    c0 = np.array([(A_pinv @ ch.frequency_response(f, 0.0).reshape(f.size, -1))[0]
                   for ch in _drops(model, 100.0, n_drops, seed,
                                    {"n_tx": 2, "n_rx": 2})])   # (drops, ports)
    mean = c0.mean(axis=0)
    var = np.mean(np.abs(c0 - mean) ** 2, axis=0)
    return float(10 * np.log10(np.mean(np.abs(mean) ** 2) / np.mean(var)))


def frequency_correlation(model: str, ds_ns: float = 100.0, n_drops: int = 200,
                          n_times: int = 4, seed: int = 0, n_rb: int = 273,
                          scs_hz: float = 30e3, max_lag: int = 100, **ch_kw):
    """Empirical vs analytic frequency correlation on an RB-sampled band.

    Returns (lag_hz, R_emp, R_th), both normalised by the analytic R(0).
    """
    ch_kw = {"n_tx": 2, "n_rx": 2, **ch_kw}
    df = 12 * scs_hz
    f = (np.arange(n_rb) - n_rb / 2) * df
    lags = np.arange(max_lag + 1)
    acc = np.zeros(lags.size, complex)
    n = 0
    times = np.arange(n_times) * 10e-3
    for ch in _drops(model, ds_ns, n_drops, seed, ch_kw):
        for t in times:
            H = ch.frequency_response(f, t).reshape(n_rb, -1)
            for k in lags:
                acc[k] += np.mean(H[: n_rb - k] * np.conj(H[k:]))
            n += 1
    tau, p = table_pdp(model)
    lag_hz = lags * df
    R_th = np.exp(2j * np.pi * lag_hz[:, None] * tau[None, :] * ds_ns * 1e-9) @ p
    R_emp = acc / n
    scale = R_emp[0].real            # mean power per port (unity by design)
    return lag_hz, R_emp / scale, R_th

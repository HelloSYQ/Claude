"""Alternative receivers: codeword-level MMSE-SIC and the (R-)ML abstraction."""

import os
import sys
from math import comb

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim.link_abstraction import (effective_sinr_miesm, ideal_amc,  # noqa: E402
                                      mcs_bler, sic_amc)
from nrdlsim.receiver import (batch_mmse_sinr, batch_mmse_sinr_est,      # noqa: E402
                              cancelled_sinr, rml_sinr_samples)

RNG = np.random.default_rng(7)


def _channel(n=30, n_rx=4, r=4, scale=None):
    H = (RNG.standard_normal((n, n_rx, r)) + 1j * RNG.standard_normal((n, n_rx, r))) / np.sqrt(2)
    if scale is not None:
        H = H * np.asarray(scale)[None, None, :]
    return H


def test_cancellation_reduces_to_mmse_and_mrc():
    H, n0 = _channel(), 0.1
    assert np.allclose(cancelled_sinr(H, H, n0, ()), batch_mmse_sinr(H, n0))
    s = cancelled_sinr(H, H, n0, (1, 2, 3))           # only layer 0 left: MRC
    assert np.all(np.isnan(s[:, 1:]))
    mrc = np.sum(np.abs(H[:, :, 0]) ** 2, axis=-1) / n0
    assert np.allclose(s[:, 0], mrc / (1 + 1e-12 * mrc), rtol=1e-6)


def test_cancellation_with_estimation_error_leaves_a_residual():
    H, n0 = _channel(), 0.01
    E = 0.2 * (RNG.standard_normal(H.shape) + 1j * RNG.standard_normal(H.shape))
    perfect = cancelled_sinr(H, H + E, n0, (0,))
    # same estimate for the remaining layers, but a perfect cancellation
    no_res = np.full_like(perfect, np.nan)
    no_res[:, 1:] = batch_mmse_sinr_est(H[..., 1:], (H + E)[..., 1:], n0)
    assert np.all(perfect[:, 1:] < no_res[:, 1:])


def test_rml_weights_and_chain_rule_identity():
    for r in (1, 2, 3, 4):
        H, n0 = _channel(r=r), 0.05
        s, w = rml_sinr_samples(H, H, n0)
        assert s.shape == (30, r, 2 ** (r - 1)) and np.allclose(w.sum(axis=1), 1.0)
        tot = (np.log2(1 + s) * w).sum(axis=(-1, -2))
        HH = np.conj(np.swapaxes(H, -1, -2)) @ H
        ref = np.log2(np.real(np.linalg.det(np.eye(r) + HH / n0)))
        assert np.allclose(tot, ref, atol=1e-8)
        if r == 1:
            assert np.allclose(s[..., 0, 0], batch_mmse_sinr(H, n0)[..., 0])
    # each weight is 1 / (r C(r-1, |T|)): the smallest sample (nothing removed)
    # is MMSE, the largest (all others removed) the interference-free bound
    H, n0 = _channel(), 0.05
    s, w = rml_sinr_samples(H, H, n0)
    assert np.isclose(w[0, 0], 1 / (4 * comb(3, 0)))
    assert np.allclose(s[..., 0], batch_mmse_sinr(H, n0))


def test_rml_helps_a_single_codeword_most_when_layers_interfere():
    H, n0, n_re, n_prb = _channel(scale=[1, 0.7, 0.4, 0.2]), 0.02, 120, 30
    mmse = ideal_amc(batch_mmse_sinr(H, n0).ravel(), 4, n_re, n_prb)[3]
    s, w = rml_sinr_samples(H, H, n0)
    ml = ideal_amc(s.ravel(), 4, n_re, n_prb, weights=np.broadcast_to(w, s.shape).ravel())[3]
    assert ml > 1.05 * mmse
    # orthogonal layers: no interference, ML = MMSE
    Q = np.linalg.qr(_channel(n_rx=4, r=4))[0] * np.array([3, 2, 1, 0.5])
    s, w = rml_sinr_samples(Q, Q, n0)
    assert np.allclose(s, s[..., :1], rtol=1e-6)


def test_weighted_miesm():
    x = 10 ** (RNG.uniform(-5, 25, 200) / 10)
    assert effective_sinr_miesm(x, 6) == effective_sinr_miesm(x, 6, np.ones(200))
    two = np.concatenate([x, x[:50]])
    w = np.concatenate([np.ones(200), np.zeros(50)])
    assert np.isclose(effective_sinr_miesm(two, 6, w), effective_sinr_miesm(x, 6))


def test_sic_one_codeword_is_mmse_and_per_layer_sic_gains():
    H, n0, n_re, n_prb = _channel(scale=[1, 0.8, 0.5, 0.3]), 0.05, 120, 30
    after = lambda c: cancelled_sinr(H, H, n0, c)           # noqa: E731
    mmse = batch_mmse_sinr(H, n0)
    one = sic_amc(after, [[0, 1, 2, 3]], n_re, n_prb)
    ref = ideal_amc(mmse.ravel(), 4, n_re, n_prb)
    assert one[0] == [ref[0]] and np.isclose(one[2], ref[3])
    per = sic_amc(after, [[0], [1], [2], [3]], n_re, n_prb)
    no_sic = sum(ideal_amc(mmse[:, k], 1, n_re, n_prb)[3] for k in range(4))
    assert per[2] > 1.05 * no_sic
    # error propagation: later codewords succeed less often than their own
    # BLER with the earlier ones always cancelled would suggest
    k = 3
    genie = 1 - mcs_bler(after((0, 1, 2))[:, [k]].ravel(), 1, per[0][k], n_re, n_prb)
    assert per[3][k] <= genie + 1e-12

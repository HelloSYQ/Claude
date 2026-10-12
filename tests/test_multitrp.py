"""Multi-TRP channel (coherent joint transmission)."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim.channel_models import CDLChannel, MultiTRPChannel   # noqa: E402

F = np.arange(12) * 360e3


def _links():
    a = CDLChannel("CDL-C", 100, 50, n_tx=8, n_rx=2, rng=np.random.default_rng(1))
    b = CDLChannel("CDL-A", 100, 50, n_tx=4, n_rx=2, rng=np.random.default_rng(2))
    return a, b


def test_stacks_trps_along_the_transmit_ports():
    a, b = _links()
    m = MultiTRPChannel([a, b])
    H = m.frequency_response(F, 0.02)
    assert H.shape == (12, 2, 12) and m.n_tx == 12 and m.n_rx == 2
    assert m.port_slices == [slice(0, 8), slice(8, 12)]
    assert np.allclose(H[..., m.port_slices[0]], a.frequency_response(F, 0.02))
    assert np.allclose(H[..., m.port_slices[1]], b.frequency_response(F, 0.02))


def test_gain_delay_phase_and_frequency_offsets():
    a, b = _links()
    m = MultiTRPChannel([a, b], gains_db=[0, -6], delays_ns=[0, 250],
                        phase_deg=[0, 30], freq_offset_hz=[0, 100])
    t = 0.01
    r = m.frequency_response(F, t)[..., 8:] / b.frequency_response(F, t)
    assert np.allclose(np.abs(r), 10 ** (-6 / 20))
    expected = np.deg2rad(30) + 2 * np.pi * 100 * t - 2 * np.pi * F * 250e-9
    assert np.allclose(np.angle(r[:, 0, 0] * np.exp(-1j * expected)), 0, atol=1e-9)

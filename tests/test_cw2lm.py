"""Codeword-to-layer mapping study: TXRU sub-array, handset UE element list,
MMSE with a channel estimate, codeword layer groups and ideal AMC.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim.channel_models import (CDLChannel, build_panel, rotation_matrix,  # noqa: E402
                                    subarray_factor, ut_handset_array,
                                    _element_list_field)
from nrdlsim.layer_mapping import layer_groups                                # noqa: E402
from nrdlsim.link_abstraction import ideal_amc                                # noqa: E402
from nrdlsim.receiver import batch_mmse_sinr, batch_mmse_sinr_est             # noqa: E402

FREQS = (np.arange(24) - 12) * 360e3


def test_subarray_factor_peaks_at_the_electrical_tilt():
    R = rotation_matrix()
    zen = np.linspace(1, 179, 357)
    for tilt in (90.0, 102.0):
        af = subarray_factor(np.zeros_like(zen), zen, R, 3, 0.7, tilt)
        g = np.abs(af) ** 2
        assert abs(zen[np.argmax(g)] - tilt) <= 0.5
        assert abs(g.max() - 3.0) < 1e-3            # K = 3 coherent elements
    assert np.allclose(subarray_factor(0.0, 60.0, R, 1, 0.7), 1.0)


def test_subarray_of_one_and_panel_element_list_change_nothing():
    kw = dict(n_tx=16, n_rx=4, tx_pol=2, rx_pol=2, tx_layout=(2, 4), rx_layout=(1, 2),
              tx_pattern="38.901", rx_pattern="38.901")
    ref = CDLChannel("CDL-A", 100, 10, rng=np.random.default_rng(3), **kw)
    sub1 = CDLChannel("CDL-A", 100, 10, rng=np.random.default_rng(3), tx_subarray=1, **kw)
    pos, slant = build_panel(4, 2, (1, 2), 0.5, 0.5)
    el = {"positions": pos, "rotation_deg": np.zeros((4, 3)),
          "pol_slant_deg": np.rad2deg(slant)}
    lst = CDLChannel("CDL-A", 100, 10, rng=np.random.default_rng(3), rx_elements=el,
                     **{k: v for k, v in kw.items() if k not in ("rx_pol", "rx_layout")})
    H = ref.frequency_response(FREQS, 0.0)
    assert np.array_equal(H, sub1.frequency_response(FREQS, 0.0))
    assert np.allclose(H, lst.frequency_response(FREQS, 0.0), atol=1e-12)


def test_txru_subarray_spaces_the_port_rows():
    ch = CDLChannel("CDL-A", 100, 10, n_tx=64, n_rx=2, tx_pol=2, tx_layout=(4, 8),
                    spacing_v=0.7, tx_subarray=3, rng=np.random.default_rng(0))
    z = np.unique(np.round(ch.pos_tx[:, 2], 6))
    assert np.allclose(np.diff(z), 2.1)                # 3 elements x 0.7 wavelength
    assert ch.frequency_response(FREQS, 0.0).shape == (24, 2, 64)


def test_handset_elements_point_out_of_their_edges():
    ue = ut_handset_array(3.5e9)
    assert ue["positions"].shape == (8, 3)
    el = dict(g_max_dbi=5.4, hpbw_deg=125.0, front_back_db=22.5)
    # (azimuth, zenith) of the boresight of each edge: top, right, bottom, left
    expect = {0: (0.0, 0.0), 2: (90.0, 90.0), 4: (0.0, 180.0), 6: (-90.0, 90.0)}
    for i, (az, zen) in expect.items():
        one = {k: v[i:i + 1] for k, v in ue.items()}
        _, A = _element_list_field(np.array(az), np.array(zen), rotation_matrix(), one,
                                   "38.901", el)
        assert abs(10 * np.log10(A[0]) - 5.4) < 1e-6
        # opposite direction is in the back lobe
        _, B = _element_list_field(np.array(az + 180.0), np.array(180.0 - zen),
                                   rotation_matrix(), one, "38.901", el)
        assert 10 * np.log10(B[0]) < 5.4 - 20
    sub = ut_handset_array(3.5e9, indices=(1, 3, 5, 7))
    assert len({tuple(r) for r in sub["rotation_deg"]}) == 4      # one per edge
    assert np.all(sub["pol_slant_deg"] == 0)


def test_handset_spreads_receive_power_across_elements():
    p = np.zeros(8)
    for seed in range(40):
        ch = CDLChannel("CDL-A", 100, 10, n_tx=8, n_rx=8, tx_pol=2, tx_layout=(1, 4),
                        tx_pattern="38.901", rx_elements=ut_handset_array(3.5e9),
                        rx_pattern="38.901", rx_element_max_gain_dbi=5.4,
                        rx_element_hpbw_deg=125.0, rx_element_front_back_db=22.5,
                        rng=np.random.default_rng(seed))
        p += (np.abs(ch.frequency_response(FREQS, 0.0)) ** 2).mean((0, 2)) / 40
    assert np.isclose(p.mean(), 1.0, rtol=0.1)          # normalised per port on average
    assert 10 * np.log10(p.max() / p.min()) > 3.0       # directive elements differ


def test_mmse_with_perfect_estimate_matches_ideal_and_errors_hurt():
    rng = np.random.default_rng(0)
    H = (rng.standard_normal((50, 8, 3)) + 1j * rng.standard_normal((50, 8, 3))) / np.sqrt(2)
    n0 = 0.05
    assert np.allclose(batch_mmse_sinr_est(H, H, n0), batch_mmse_sinr(H, n0))
    E = np.sqrt(n0 / 2) * (rng.standard_normal(H.shape) + 1j * rng.standard_normal(H.shape))
    assert batch_mmse_sinr_est(H, H + E, n0).mean() < batch_mmse_sinr(H, n0).mean()


def test_layer_groups():
    assert layer_groups(4, 1) == [[0, 1, 2, 3]]
    assert layer_groups(4, 2) == [[0, 1], [2, 3]]
    assert layer_groups(3, 2) == [[0], [1, 2]]
    assert layer_groups(5, 2) == [[0, 1], [2, 3, 4]]           # TS 38.211 Table 7.3.1.3-1
    assert layer_groups(4, 4) == [[0], [1], [2], [3]]
    assert layer_groups(2, 4) == [[0], [1]]
    for r in range(1, 9):
        for c in (1, 2, 4):
            g = layer_groups(r, c)
            assert sum(g, []) == list(range(r)) and len(g) == min(r, c)


def test_ideal_amc_codeword_per_layer_gains_only_with_imbalance():
    n_re, n_prb = 120, 96
    def joint_vs_split(sinr_db):
        s = 10 ** (np.repeat(np.asarray(sinr_db, float)[None, :], n_prb, 0) / 10)
        joint = ideal_amc(s.ravel(), s.shape[1], n_re, n_prb)[3]
        split = sum(ideal_amc(s[:, k], 1, n_re, n_prb)[3] for k in range(s.shape[1]))
        return joint, split
    j, s = joint_vs_split([12, 12, 12, 12])
    assert abs(s / j - 1) < 0.03                     # equal layers: same SE
    j, s = joint_vs_split([24, 14, 4, -6])
    assert s > 1.1 * j                               # strong imbalance: per-layer wins
    lo = ideal_amc(np.full(96, 10 ** 0.5), 1, n_re, n_prb)
    hi = ideal_amc(np.full(96, 10 ** 2.0), 1, n_re, n_prb)
    assert hi[0] > lo[0] and hi[3] > lo[3]

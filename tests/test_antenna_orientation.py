"""Regression tests: TR 38.901 clause 7.1.3 antenna orientation in the CDL model.

The implementation rotates local unit vectors with R = Rz(alpha) Ry(beta)
Rx(gamma); these tests check it against the closed-form expressions of the
spec and against physical expectations.
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim.channel_models import (rotation_matrix, to_local_angles,
                                    element_field, element_power_gain,
                                    build_panel, _location_phase, CDLChannel)


def test_matches_tr38901_closed_forms():
    rng = np.random.default_rng(1)
    for _ in range(2000):
        al, be, ga = rng.uniform(-180, 180), rng.uniform(-90, 90), rng.uniform(-180, 180)
        th, ph = rng.uniform(1, 179), rng.uniform(-180, 180)
        a, b, g, t, p = np.deg2rad([al, be, ga, th, ph])
        d = p - a
        th_cf = np.rad2deg(np.arccos(np.cos(b) * np.cos(g) * np.cos(t) + (
            np.sin(b) * np.cos(g) * np.cos(d) - np.sin(g) * np.sin(d)) * np.sin(t)))
        ph_cf = np.rad2deg(np.angle(
            (np.cos(b) * np.sin(t) * np.cos(d) - np.sin(b) * np.cos(t))
            + 1j * (np.cos(b) * np.sin(g) * np.cos(t)
                    + (np.sin(b) * np.sin(g) * np.cos(d) + np.cos(g) * np.sin(d))
                    * np.sin(t))))                                   # eq. 7.1-8
        psi_cf = np.angle(
            (np.sin(g) * np.cos(t) * np.sin(d)
             + np.cos(g) * (np.cos(b) * np.sin(t) - np.sin(b) * np.cos(t) * np.cos(d)))
            + 1j * (np.sin(g) * np.cos(d) + np.sin(b) * np.cos(g) * np.sin(d)))  # 7.1-15
        R = rotation_matrix(al, be, ga)
        az_l, zen_l = to_local_angles(ph, th, R)
        assert abs(zen_l - th_cf) < 1e-9
        assert abs((az_l - ph_cf + 180) % 360 - 180) < 1e-9
        F, A = element_field(ph, th, np.array([0.0]), R, "38.901")
        psi = np.arctan2(F[0, 1], F[0, 0])            # zeta=0: F = sqrt(A)(cos, sin)psi
        assert abs(np.angle(np.exp(1j * (psi - psi_cf)))) < 1e-9
        assert abs(np.sum(F ** 2) - A) < 1e-12        # rotation preserves power
    print("rotation == TR 38.901 eq. 7.1-7 / 7.1-8 / 7.1-15, power preserved: OK")


def test_pattern_and_array_turn_together():
    pos, _ = build_panel(8, 2, (2, 2), 0.5, 0.5)
    for bearing, tilt in [(0, 0), (73, 0), (-120, 25), (40, -15)]:
        R = rotation_matrix(bearing, tilt, 0)
        g = element_power_gain(bearing, 90 + tilt, bearing, tilt)
        assert abs(10 * np.log10(g) - 8.0) < 1e-9      # pattern peak at boresight
        ph = _location_phase(pos @ R.T, np.array(bearing), np.array(90 + tilt))
        assert np.ptp(np.angle(ph)) < 1e-9             # array broadside there too
    # rolling a vertical element by 90 deg makes it horizontally polarized
    F, _ = element_field(0.0, 90.0, np.array([0.0]), rotation_matrix(0, 0, 90))
    assert np.allclose(F[0], [0.0, 1.0], atol=1e-12)
    print("pattern peak, array broadside and polarization rotate together: OK")


def test_zero_orientation_is_identity():
    ch = CDLChannel("CDL-C", 100, 10, n_tx=8, n_rx=4, tx_pol=2, rx_pol=2,
                    tx_layout=(2, 2), rx_layout=(1, 2), rng=np.random.default_rng(0))
    pos, _ = build_panel(8, 2, (2, 2), 0.5, 0.5)
    assert np.allclose(ch.R_tx, np.eye(3)) and np.allclose(ch.pos_tx, pos)
    print("zero orientation leaves the panel unrotated: OK")


def test_unit_power_under_random_orientation():
    f = (np.arange(24) - 12) * 360e3
    rng = np.random.default_rng(7)
    p = []
    for s in range(150):
        ch = CDLChannel("CDL-C", 30, 10, n_tx=16, n_rx=4, tx_pol=2, rx_pol=2,
                        tx_layout=(2, 4), rx_layout=(1, 2),
                        tx_pattern="38.901", downtilt_deg=rng.uniform(0, 12),
                        rx_pattern="38.901",
                        rx_boresight_az_deg=rng.uniform(0, 360),
                        rx_downtilt_deg=rng.uniform(-20, 20),
                        rx_slant_deg=rng.uniform(-180, 180),
                        rx_element_max_gain_dbi=5.0, rx_element_hpbw_deg=90.0,
                        rng=np.random.default_rng(s))
        p.append(np.mean(np.abs(ch.frequency_response(f, 0.0)) ** 2))
    assert abs(np.mean(p) - 1.0) < 0.05, np.mean(p)
    print(f"unit per-port power with random panel orientation ({np.mean(p):.3f}): OK")


def test_separate_ue_element():
    kw = dict(n_tx=4, n_rx=2, tx_pattern="38.901", rx_pattern="38.901",
              rng=np.random.default_rng(0))
    a = CDLChannel("CDL-B", 100, 10, **kw)
    kw["rng"] = np.random.default_rng(0)
    b = CDLChannel("CDL-B", 100, 10, rx_element_hpbw_deg=120.0,
                   rx_element_max_gain_dbi=3.0, **kw)
    # same seed -> same rays; the tx element is unchanged, so the per-ray
    # power-gain ratio must equal the ratio of the two rx elements' gains
    r = (b.ray_gain / a.ray_gain) ** 2
    exp = (element_power_gain(b.aoa, b.zoa, 0, 0, 3.0, 120.0)
           / element_power_gain(a.aoa, a.zoa, 0, 0, 8.0, 65.0))
    assert np.allclose(r, exp)
    print("UE element parameters independent of the gNB element: OK")


if __name__ == "__main__":
    test_matches_tr38901_closed_forms()
    test_pattern_and_array_turn_together()
    test_zero_orientation_is_identity()
    test_unit_power_under_random_orientation()
    test_separate_ue_element()
    print("\nAll antenna-orientation tests passed.")

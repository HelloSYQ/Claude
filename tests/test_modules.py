"""Sanity/self tests for the NR simulator building blocks."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim.modulation import Modulator, _build_constellation
from nrdlsim import mcs_tables, tbs as tbs_mod
from nrdlsim import ldpc as ldpc_mod
from nrdlsim.channel_models import TDLChannel
from nrdlsim.link_abstraction import bicm_capacity


def test_constellation_unit_energy():
    for qm in (2, 4, 6, 8):
        c = _build_constellation(qm)
        assert abs(np.mean(np.abs(c) ** 2) - 1.0) < 1e-9, qm
    print("constellation unit-energy: OK")


def test_modulation_roundtrip_no_noise():
    for qm in (2, 4, 6, 8):
        mod = Modulator(qm)
        bits = np.random.default_rng(0).integers(0, 2, size=qm * 500)
        sym = mod.modulate(bits)
        llr = mod.demodulate_llr(sym, 1e-3)
        rx_bits = (llr < 0).astype(int)
        assert np.array_equal(bits, rx_bits), qm
    print("modulation round-trip (noiseless): OK")


def test_mcs_and_tbs():
    info = mcs_tables.get_mcs(16, table=2)
    assert info.modulation_order == 6
    # a plausible TBS for 51 PRB, 13 symbols, rank 1
    n_re = tbs_mod.re_per_rb(13, 24, 0)
    tb = tbs_mod.compute_tbs(n_re, 51, info.modulation_order,
                             info.target_code_rate, 1)
    assert tb > 0 and tb % 8 == 0
    print(f"MCS/TBS: OK (TBS={tb} bits)")


def test_awgn_mimo_channel():
    """AWGN MIMO: unit gain on every link (same normalisation as the fading
    models), orthogonal rows -> every eigenmode carries the full array gain."""
    from nrdlsim.channel_models import awgn_frequency_response
    assert np.allclose(awgn_frequency_response(5, 1, 1), 1.0)       # SISO
    for nt, nr in ((2, 2), (4, 4), (32, 4), (128, 4), (2, 4)):
        H = awgn_frequency_response(3, nt, nr)
        assert H.shape == (3, nr, nt) and np.allclose(np.abs(H), 1.0)
        g = np.linalg.svd(H[0], compute_uv=False) ** 2
        assert np.allclose(g, max(nt, nr)), (nt, nr, g)
    print("AWGN MIMO channel: unit link gain, equal eigen-gains: OK")


def test_code_block_tb_model():
    """TS 38.212 segmentation, TB BLER from code blocks and its inverse."""
    from nrdlsim.link_abstraction import (cb_segmentation, tb_bler_from_cb_sinrs,
                                          required_eff_sinr_db_tb, cb_re_weights)
    # B = A + 24; BG1 (K_cb 8448): C = ceil(B / 8424); BG2 (3840): ceil(B / 3816)
    assert cb_segmentation(200, 0.5) == (1, 216)                  # small TB: 16-bit CRC
    assert cb_segmentation(100000, 0.9)[0] == 12                  # BG1
    assert cb_segmentation(100000, 0.2)[0] == 27                  # BG2 (R <= 1/4)
    c, k = cb_segmentation(100000, 0.9)
    assert 0 <= k * c - (100024 + 24 * c) < c                     # CB CRCs added, K' rounded up
    # TB BLER combines code blocks; the inverse hits the target exactly
    for tbs in (5000, 60000, 500000):
        s = required_eff_sinr_db_tb(6, 0.6, tbs, 0.1)
        assert abs(tb_bler_from_cb_sinrs(s, 6, 0.6, tbs) - 0.1) < 1e-9
    # more code blocks need more SINR for the same TB BLER
    assert required_eff_sinr_db_tb(6, 0.6, 500000, 0.1) > required_eff_sinr_db_tb(6, 0.6, 60000, 0.1)
    # one bad code block fails the TB
    c, _ = cb_segmentation(500000, 0.6)
    e = np.full(c, 30.0); e[3] = -10.0
    assert tb_bler_from_cb_sinrs(e, 6, 0.6, 500000) > 0.99
    # CB -> RB footprint: rows are distributions; each CB covers ~1/C of the REs
    w = cb_re_weights(264, 62, 132)
    assert np.allclose(w.sum(axis=1), 1.0) and w.shape == (62, 264)
    print("code-block TB model (segmentation, TB BLER, inverse, RE mapping): OK")


def test_tdl_power_normalised():
    ch = TDLChannel("TDL-C", 100e0, 50.0, n_tx=4, n_rx=2,
                    rng=np.random.default_rng(1))
    t = np.linspace(0, 1e-3, 50)
    h = ch.taps(t)                       # [tap, rx, tx, time]
    # average power across taps/time for a single antenna pair ~ 1
    p = np.mean(np.abs(h[:, 0, 0, :]) ** 2) * ch.n_taps
    assert 0.3 < p < 3.0, p
    print(f"TDL fading power sanity: OK (mean tap power sum ~ {p:.2f})")


def test_cdl_channel():
    """CDL models: correct shape, ~unit average gain, and freq selectivity."""
    from nrdlsim.channel_models import CDLChannel
    freqs = (np.arange(51) - 25) * 12 * 30e3
    for m in ("CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"):
        ch = CDLChannel(m, 100.0, 100.0, n_tx=4, n_rx=2,
                        rng=np.random.default_rng(0))
        p = np.mean([np.mean(np.abs(ch.frequency_response(freqs, k * 5e-4)) ** 2)
                     for k in range(120)])
        assert ch.frequency_response(freqs, 0.0).shape == (51, 2, 4), m
        assert 0.5 < p < 1.7, (m, p)
        assert (m in ("CDL-D", "CDL-E")) == ch.has_los, m
    # frequency selectivity for a dispersive profile
    ch = CDLChannel("CDL-C", 300.0, 50.0, 4, 2, rng=np.random.default_rng(1))
    g = np.abs(ch.frequency_response(freqs, 0.0)[:, 0, 0])
    assert g.max() / g.min() > 1.3
    print("CDL channel shape / power / LOS / selectivity: OK")


def test_cdl_dualpol_upa():
    """Dual-polarized UPA panels: correct port count and unit-power ensemble."""
    from nrdlsim.channel_models import CDLChannel, build_panel
    # panel: 2x2 grid of dual-pol positions -> 8 ports, 4 positions, +/-45 deg
    pos, slant = build_panel(8, pol=2, layout=(2, 2), spacing_v=0.5, spacing_h=0.5)
    assert len(slant) == 8 and len(np.unique(pos, axis=0)) == 4
    assert set(np.round(np.rad2deg(np.unique(slant))).astype(int)) == {-45, 45}

    freqs = (np.arange(51) - 25) * 12 * 30e3

    def ensemble(**kw):
        vals = []
        for seed in range(30):
            ch = CDLChannel("CDL-C", 100.0, 100.0, rng=np.random.default_rng(seed),
                            **kw)
            vals.append(np.mean([np.mean(np.abs(
                ch.frequency_response(freqs, k * 3e-3)) ** 2) for k in range(8)]))
        return float(np.mean(vals))

    # dual-pol and UPA must be power-normalised to the same ~unit level
    p_single = ensemble(n_tx=4, n_rx=2, tx_pol=1, rx_pol=1)
    p_dual = ensemble(n_tx=4, n_rx=2, tx_pol=2, rx_pol=2)
    p_upa = ensemble(n_tx=8, n_rx=2, tx_pol=2, rx_pol=2, tx_layout=(2, 2))
    for name, p in [("single", p_single), ("dual", p_dual), ("upa", p_upa)]:
        assert 0.8 < p < 1.2, (name, p)
    # dual-pol UPA channel must have the right shape
    ch = CDLChannel("CDL-C", 100.0, 100.0, n_tx=8, n_rx=4, tx_pol=2, rx_pol=2,
                    tx_layout=(2, 2), rx_layout=(1, 2),
                    rng=np.random.default_rng(0))
    assert ch.frequency_response(freqs, 0.0).shape == (51, 4, 8)
    print(f"CDL dual-pol / UPA: OK (power single={p_single:.2f} "
          f"dual={p_dual:.2f} upa={p_upa:.2f})")


def test_antenna_gain_shaping():
    """TR 38.901 element pattern: boresight/3dB/backlobe gains and angular
    filtering, with unit-power normalisation preserved."""
    from nrdlsim.channel_models import element_power_gain, CDLChannel

    def dbi(az, zen):
        return 10 * np.log10(element_power_gain(az, zen))

    assert abs(dbi(0, 90) - 8.0) < 1e-6                 # boresight = G_max
    assert abs(dbi(32.5, 90) - 5.0) < 1e-2              # -3 dB at hpbw/2
    assert abs(dbi(0, 122.5) - 5.0) < 1e-2              # -3 dB in elevation
    assert abs(dbi(180, 90) - (-22.0)) < 1e-6           # backlobe floor

    freqs = (np.arange(51) - 25) * 12 * 30e3

    def ens(**kw):
        v = []
        for s in range(30):
            ch = CDLChannel("CDL-C", 100.0, 100.0, rng=np.random.default_rng(s),
                            **kw)
            v.append(np.mean([np.mean(np.abs(
                ch.frequency_response(freqs, k * 3e-3)) ** 2) for k in range(8)]))
        return float(np.mean(v))

    base = dict(n_tx=8, n_rx=2, tx_pol=2, rx_pol=2, tx_layout=(2, 2))
    p_omni = ens(**base)
    p_dir = ens(tx_pattern="38.901", rx_pattern="38.901", **base)
    assert 0.85 < p_omni < 1.15 and 0.85 < p_dir < 1.15  # normalisation holds

    # directional element angularly filters the rays -> non-unit ray gains
    ch = CDLChannel("CDL-C", 100.0, 100.0, tx_pattern="38.901", downtilt_deg=8.0,
                    rng=np.random.default_rng(1), **base)
    assert ch.ray_gain.min() < 0.5 < ch.ray_gain.max()
    print(f"antenna gain shaping: OK (|H|^2 omni={p_omni:.2f} dir={p_dir:.2f}, "
          f"ray_gain {ch.ray_gain.min():.2f}..{ch.ray_gain.max():.2f})")


def test_ofdm_waveform():
    """Time-domain OFDM: loopback identity, CP orthogonality, CFO ICI, timing."""
    from nrdlsim.ofdm import (OFDMModulator, waveform_evm, cfo_ici_sinr_db,
                              apply_multipath)
    rng = np.random.default_rng(0)
    mod = OFDMModulator(51, 30e3, mu=1)
    cp = int(mod.cp[1])
    g = ((rng.integers(0, 2, (mod.n_sc, mod.n_sym)) * 2 - 1)
         + 1j * (rng.integers(0, 2, (mod.n_sc, mod.n_sym)) * 2 - 1)) / np.sqrt(2)

    # 1) perfect reconstruction with no channel/impairment/noise
    err = np.max(np.abs(mod.demodulate(mod.modulate(g)) - g))
    assert err < 1e-9, err

    # 2) CP preserves orthogonality when delay < CP: time-domain multipath
    #    equals per-subcarrier multiply by the channel frequency response
    h = np.zeros(20, complex); h[0] = 1; h[5] = 0.5; h[12] = 0.3j
    grh = mod.demodulate(apply_multipath(mod.modulate(g), h))
    H = mod.channel_frequency(h)[:, None]
    assert np.max(np.abs(grh - H * g)) < 1e-9

    # 3) CFO-induced SINR matches the analytic ICI ceiling within ~1 dB
    for eps in (0.01, 0.02, 0.05):
        _, sinr = waveform_evm(mod, g, 50, cfo_hz=eps * 30e3, h_time=h,
                               rng=np.random.default_rng(1))
        assert abs(sinr - cfo_ici_sinr_db(eps * 30e3, 30e3)) < 1.5, eps

    # 4) timing error inside the CP is recoverable; beyond it causes ISI
    _, s_in = waveform_evm(mod, g, 50, timing_offset=-cp // 2, h_time=h,
                           rng=np.random.default_rng(2))
    _, s_out = waveform_evm(mod, g, 50, timing_offset=-cp - 40, h_time=h,
                            rng=np.random.default_rng(2))
    assert s_in > 45 and s_out < 20
    print("OFDM waveform: loopback / CP orthogonality / CFO ICI / timing: OK")


def test_bicm_capacity_monotonic():
    snr = np.linspace(-20, 40, 1201)
    for qm in (2, 4, 6, 8, 10):
        cap = bicm_capacity(snr, qm)
        assert np.all(np.diff(cap) >= 0), qm
        assert cap[-1] > qm - 1e-3
    print("BICM capacity monotonic + saturating: OK")


def test_bicm_capacity_reference():
    """The exact BICM capacity against references computed independently."""
    from scipy.integrate import quad

    def bpsk(snr_db):
        # binary-input AWGN: 1 - E[log2(1 + exp(-2y / s2))], y ~ N(1, s2)
        s2 = 10 ** (-snr_db / 10)
        f = lambda n: (np.exp(-n * n / (2 * s2)) / np.sqrt(2 * np.pi * s2)
                       * np.logaddexp(0, -2 * (1 + n) / s2) / np.log(2))
        sd = np.sqrt(s2)
        return 1 - quad(f, -40 * sd, 40 * sd, points=[-1.0], limit=200)[0]

    # QPSK is one BPSK per axis, each at the symbol SNR
    assert abs(bpsk(0) - 0.486) < 1e-3
    for snr_db in (-10, -5, 0, 5, 10):
        assert abs(bicm_capacity(snr_db, 2) - 2 * bpsk(snr_db)) < 1e-5, snr_db
    print("BICM capacity vs references: OK")


def test_ldpc_corrects_errors():
    """The QC-LDPC must decode successfully at high SNR and fail at very low."""
    rng = np.random.default_rng(3)
    code = ldpc_mod.build_code(bg=2, z=32)
    from nrdlsim.modulation import Modulator
    mod = Modulator(2)

    def run(snr_db, trials=20):
        ok = 0
        snr = 10 ** (snr_db / 10)
        noise = np.sqrt(1 / (2 * snr))
        for _ in range(trials):
            info = rng.integers(0, 2, size=code.k_bits).astype(np.int8)
            cw = ldpc_mod.encode(code, info)
            pad = (-cw.size) % 2
            tx = mod.modulate(np.concatenate([cw, np.zeros(pad, np.int8)]))
            rx = tx + noise * (rng.standard_normal(tx.size)
                               + 1j * rng.standard_normal(tx.size))
            llr = mod.demodulate_llr(rx, 2 * noise ** 2)[:cw.size]
            hard, success = ldpc_mod.decode(code, llr, max_iter=25)
            if success and np.array_equal(hard[:code.k_bits], info):
                ok += 1
        return ok / trials

    hi = run(6.0)
    lo = run(-6.0)
    print(f"LDPC decode success: SNR=+6dB -> {hi:.2f}, SNR=-6dB -> {lo:.2f}")
    assert hi > lo
    assert hi >= 0.7


if __name__ == "__main__":
    test_constellation_unit_energy()
    test_modulation_roundtrip_no_noise()
    test_mcs_and_tbs()
    test_awgn_mimo_channel()
    test_code_block_tb_model()
    test_tdl_power_normalised()
    test_cdl_channel()
    test_cdl_dualpol_upa()
    test_antenna_gain_shaping()
    test_ofdm_waveform()
    test_bicm_capacity_monotonic()
    test_bicm_capacity_reference()
    test_ldpc_corrects_errors()
    print("\nAll module self-tests passed.")

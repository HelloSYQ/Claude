"""Sanity/self tests for the NR simulator building blocks."""

import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim.modulation import Modulator, _build_constellation
from nrdlsim import mcs_tables, tbs as tbs_mod
from nrdlsim import ldpc as ldpc_mod
from nrdlsim.channel_models import TDLChannel
from nrdlsim.link_abstraction import bicm_capacity, required_snr_db


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
        # average |H|^2 per antenna pair over many snapshots -> ~1 (E[.]=1)
        p = np.mean([np.mean(np.abs(ch.frequency_response(freqs, k * 5e-4)) ** 2)
                     for k in range(120)])
        assert ch.frequency_response(freqs, 0.0).shape == (51, 2, 4), m
        assert 0.6 < p < 1.6, (m, p)
        assert (m in ("CDL-D", "CDL-E")) == ch.has_los, m
    # frequency selectivity for a dispersive profile
    ch = CDLChannel("CDL-C", 300.0, 50.0, 4, 2, rng=np.random.default_rng(1))
    g = np.abs(ch.frequency_response(freqs, 0.0)[:, 0, 0])
    assert g.max() / g.min() > 1.3
    print("CDL channel shape / power / LOS / selectivity: OK")


def test_bicm_capacity_monotonic():
    snr = np.linspace(-10, 30, 20)
    for qm in (2, 4, 6, 8):
        cap = bicm_capacity(snr, qm)
        assert np.all(np.diff(cap) >= -1e-3), qm
        assert cap[-1] > qm * 0.8
    print("BICM capacity monotonic + saturating: OK")


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
    test_tdl_power_normalised()
    test_cdl_channel()
    test_bicm_capacity_monotonic()
    test_ldpc_corrects_errors()
    print("\nAll module self-tests passed.")

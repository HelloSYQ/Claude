"""Regression tests for CSI, link adaptation, HARQ, TBS and SE accounting.

Each test pins down a bug that previously went unnoticed:
  * CSI evaluated the precoder without the 1/sqrt(rank) power split
  * CQI thresholds sat at ~38% BLER instead of the target
  * OLLA could only back off (offset clamped at >= 0)
  * the simulator mutated cfg.pdsch.num_layers (serial != parallel sweeps)
  * SE was normalised by the carrier grid, not the allocated bandwidth
  * small-TBS quantisation used round() instead of the spec's floor()
  * MCS table 4 (1024QAM) crashed under link adaptation
  * HARQ retransmissions were re-scheduled as a different TB
"""

import copy
import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim.config import SimConfig
from nrdlsim.link_simulator import NRDownlinkSimulator
from nrdlsim.channel_models import CDLChannel
from nrdlsim.csi import compute_csi
from nrdlsim.scheduler import Scheduler
from nrdlsim.receiver import batch_mmse_sinr
from nrdlsim.link_abstraction import (effective_sinr_miesm,
                                      bler_from_effective_sinr,
                                      required_eff_sinr_db)
from nrdlsim import mcs_tables, tbs as tbs_mod


def test_csi_sinr_matches_transmitted_power():
    freqs = (np.arange(24) - 12) * 12 * 30e3
    H = CDLChannel("CDL-C", 30, 10, n_tx=8, n_rx=4,
                   rng=np.random.default_rng(0)).frequency_response(freqs, 0)
    rep = compute_csi(H, 10 ** (-20 / 10), 4, mcs_table=2)
    qm = mcs_tables.get_cqi(rep.cqi, 2)[0]
    W = rep.precoder / np.sqrt(rep.rank)          # what the gNB transmits
    eff_tx = effective_sinr_miesm(batch_mmse_sinr(H @ W, 0.01).reshape(-1), qm)
    assert abs(rep.eff_sinr_db - eff_tx) < 1e-9, (rep.eff_sinr_db, eff_tx)
    assert set(rep.precoders) == {1, 2, 3, 4}
    print(f"CSI SINR == transmitted SINR (rank {rep.rank}): OK")


def test_required_sinr_inverts_waterfall():
    for qm, r in [(2, 0.12), (4, 0.48), (6, 0.75), (8, 0.93), (10, 0.93)]:
        for tb in (100, 5000, 80000):
            for p in (0.01, 0.1, 0.3):
                s = required_eff_sinr_db(qm, r, tb, p)
                assert abs(bler_from_effective_sinr(s, qm, r, tb) - p) < 1e-9
    print("required_eff_sinr_db inverts the BLER waterfall: OK")


def test_olla_is_bidirectional_and_balanced():
    s = Scheduler(24, 2, target_bler=0.1)
    for _ in range(50):
        s.update_olla(True)
    assert s.olla_offset < 0, "OLLA must be able to become aggressive"
    # zero drift at exactly the target NACK rate
    s.olla_offset = 0.0
    for _ in range(90):
        s.update_olla(True)
    for _ in range(10):
        s.update_olla(False)
    assert abs(s.olla_offset) < 1e-9, s.olla_offset
    print("OLLA bidirectional, zero drift at target BLER: OK")


def test_first_tx_bler_tracks_target():
    for target in (0.1, 0.3):
        cfg = SimConfig(num_slots=600)
        cfg.pdsch.target_bler = target
        cfg.channel.max_doppler_hz = 10
        r = NRDownlinkSimulator(cfg).run_point(15.0)
        assert abs(r.bler - target) < 0.35 * target + 0.02, (target, r.bler)
        print(f"first-tx BLER {r.bler:.3f} vs target {target}: OK")


def test_config_not_mutated_and_serial_equals_parallel():
    cfg = SimConfig(snr_db_range=(5, 25, 10), num_slots=60)
    cfg.pdsch.num_layers = 1
    before = copy.deepcopy(cfg)
    ser = NRDownlinkSimulator(cfg).run(n_jobs=1)
    assert cfg == before, "run() must not modify the caller's config"
    par = NRDownlinkSimulator(copy.deepcopy(before)).run(n_jobs=2)
    for a, b in zip(ser, par):
        assert a == b, (a, b)
    print("config untouched, serial == parallel: OK")


def test_se_uses_allocated_bandwidth():
    c = SimConfig(num_slots=20, link_adaptation=False)
    c.pdsch.num_rb = 24                       # 24 RB on a 51 RB carrier
    se_a = NRDownlinkSimulator(c).run_point(30.0).spectral_efficiency
    c.carrier.n_size_grid = 24
    se_b = NRDownlinkSimulator(c).run_point(30.0).spectral_efficiency
    assert se_a == se_b, (se_a, se_b)
    c.pdsch.num_rb = 60
    try:
        NRDownlinkSimulator(c)
        raise AssertionError("num_rb > n_size_grid must be rejected")
    except ValueError:
        pass
    print("SE over allocated bandwidth; oversize allocation rejected: OK")


def test_tbs_small_quantisation_uses_floor():
    # N_info = 12 * 616/1024 * 4 = 28.875 -> n = 3, N'_info = 8*floor(3.6) = 24
    assert tbs_mod.compute_tbs(12, 1, 4, 616 / 1024, 1) == 24
    # (the old round() gave N'_info = 32 -> TBS 32)
    # N_info = 1000 exactly -> N'_info = 1000 -> smallest table TBS >= 1000
    assert tbs_mod.compute_tbs(125, 1, 2, 1.0, 4) == 1032
    # large-N branch, exact half: N_info = 4952 -> n = 7,
    # (4952-24)/128 = 38.5 -> rounds half up to 39 -> N'_info = 4992 -> TBS 4992
    # (Python's round-half-to-even would give 38 -> 4864)
    assert tbs_mod.compute_tbs(619, 1, 8, 1.0, 1) == 4992
    print("TBS quantisation (floor for small N, round-half-up for large): OK")


def test_mcs_table4_with_link_adaptation():
    c = SimConfig(num_slots=40, ideal_channel_estimation=True)
    c.pdsch.mcs_table = 4
    r = NRDownlinkSimulator(c).run_point(45.0)
    assert r.avg_mcs > 27, r.avg_mcs          # reaches the 1024QAM rows
    print(f"MCS table 4 under link adaptation (avg MCS {r.avg_mcs:.1f}): OK")


def test_harq_accounting():
    c = SimConfig(num_slots=300)
    c.harq.enabled = False
    r = NRDownlinkSimulator(c).run_point(10.0)
    # without HARQ every TB gets exactly one transmission
    assert r.residual_bler == r.bler, (r.residual_bler, r.bler)
    c.harq.enabled = True
    r = NRDownlinkSimulator(c).run_point(10.0)
    assert r.residual_bler < r.bler, (r.residual_bler, r.bler)
    print(f"HARQ: residual {r.residual_bler:.3f} < first-tx {r.bler:.3f}: OK")


if __name__ == "__main__":
    test_csi_sinr_matches_transmitted_power()
    test_required_sinr_inverts_waterfall()
    test_olla_is_bidirectional_and_balanced()
    test_first_tx_bler_tracks_target()
    test_config_not_mutated_and_serial_equals_parallel()
    test_se_uses_allocated_bandwidth()
    test_tbs_small_quantisation_uses_floor()
    test_mcs_table4_with_link_adaptation()
    test_harq_accounting()
    print("\nAll link-adaptation regression tests passed.")

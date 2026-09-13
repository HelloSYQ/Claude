#!/usr/bin/env python3
"""Codeword mapping (1-CW vs 4-CW, 4 layers) with a handheld UT UE model.

Redo of the 1-CW-vs-4-CW comparison contrasting two UE antenna models:

  * 'array' : omni dual-pol UPA (fixed orientation) -- the earlier model.
  * 'UT'    : handheld -- directional dual-pol elements (broad UE pattern,
              ~5 dBi / 90 deg HPBW) at a RANDOM device orientation per drop
              (uniform azimuth, tilted zenith). The directional element +
              orientation change the angular filtering, hence the per-layer
              (eigenmode) SINR spread that drives the codeword-mapping trade-off.

Scenario: CDL-C DS=30 ns, 24 RB @ 30 kHz, 32T4R, SVD, upper 6 GHz (6.7 GHz),
~3 km/h. Each scheme adapts MCS naturally (max expected throughput).

Note: the CDL channel normalises average per-port power to unity, so the UT
model here shapes the spatial STRUCTURE (which clusters the UE favours) rather
than adding an absolute orientation-dependent power offset.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim import mcs_tables, tbs as tbs_mod
from nrdlsim.channel_models import CDLChannel
from nrdlsim.layer_mapping import svd_precoder
from nrdlsim.receiver import batch_mmse_sinr
from nrdlsim.link_abstraction import effective_sinr_miesm, bler_from_effective_sinr

N_RB, SCS = 24, 30e3
MU_BW = N_RB * 12 * SCS
SLOT_TIME = 0.5e-3
N_RE_PRB = tbs_mod.re_per_rb(13, 24, 0)
RANK, N_TX, N_RX = 4, 32, 4
FC = 6.7e9                                   # upper 6 GHz
FD = (3.0 / 3.6) * FC / 3e8
QMS = (2, 4, 6, 8)


def choose_mcs(sinr_lin, n_layers):
    eff = {qm: effective_sinr_miesm(sinr_lin, qm) for qm in QMS}
    best_tp, best = -1.0, 0.0
    for idx in range(mcs_tables.num_mcs(2)):
        info = mcs_tables.get_mcs(idx, 2)
        tb = tbs_mod.compute_tbs(N_RE_PRB, N_RB, info.modulation_order,
                                 info.target_code_rate, n_layers)
        bler = bler_from_effective_sinr(eff[info.modulation_order],
                                        info.modulation_order,
                                        info.target_code_rate, tb)
        tp = tb * (1 - bler)
        if tp > best_tp:
            best_tp, best = tp, tp
    return best


def make_channel(seed, ue_model, orient_rng):
    kw = dict(model="CDL-C", delay_spread_ns=30, max_doppler_hz=FD,
              n_tx=N_TX, n_rx=N_RX, carrier_freq_hz=FC,
              tx_pol=2, rx_pol=2, tx_layout=(2, 8), rx_layout=(1, 2),
              rng=np.random.default_rng(seed))
    if ue_model == "ut":
        kw.update(rx_pattern="38.901",
                  rx_boresight_az_deg=orient_rng.uniform(0, 360),
                  rx_downtilt_deg=orient_rng.uniform(70, 110),
                  element_max_gain_dbi=5.0, element_hpbw_deg=90.0)
    return CDLChannel(**kw)


def run_point(snr_db, ue_model, n_slots=150, base=0):
    nv = 10 ** (-snr_db / 10.0)
    freqs = (np.arange(N_RB) - N_RB / 2) * 12 * SCS
    orient_rng = np.random.default_rng(777 + int(snr_db))
    bits1 = bits4 = 0.0
    for s in range(n_slots):
        ch = make_channel(base + s, ue_model, orient_rng)
        H = ch.frequency_response(freqs, 0.0)
        W = svd_precoder(H, RANK) / np.sqrt(RANK)
        sinr = batch_mmse_sinr(H @ W, nv)                  # [n_rb, 4]
        bits1 += choose_mcs(sinr.reshape(-1), RANK)
        bits4 += sum(choose_mcs(sinr[:, k], 1) for k in range(RANK))
    se1 = bits1 / n_slots / SLOT_TIME / MU_BW
    se4 = bits4 / n_slots / SLOT_TIME / MU_BW
    return se1, se4


def main():
    snrs = np.arange(-5, 26, 2.5)
    data = {"array": ([], []), "ut": ([], [])}
    for model in ("array", "ut"):
        print(f"\n=== UE model: {model} (32T4R CDL-C, upper 6 GHz) ===")
        print(f"{'SNR':>6} {'SE 1CW':>8} {'SE 4CW':>8} {'gain%':>7}")
        for snr in snrs:
            s1, s4 = run_point(float(snr), model)
            data[model][0].append(s1); data[model][1].append(s4)
            g = (s4 / s1 - 1) * 100 if s1 > 0 else 0
            print(f"{snr:6.1f} {s1:8.3f} {s4:8.3f} {g:6.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for a, model, title in [(ax[0], "array", "Array UE (omni dual-pol UPA)"),
                            (ax[1], "ut", "UT UE (directional, random orient.)")]:
        a.plot(snrs, data[model][0], "o-", color="#2563eb", label="1 codeword")
        a.plot(snrs, data[model][1], "s-", color="#dc2626", label="4 codewords")
        a.set_xlabel("SNR (dB)"); a.set_title(title); a.grid(alpha=0.3); a.legend()
    ax[0].set_ylabel("spectral efficiency (b/s/Hz)")
    fig.suptitle("1-CW vs 4-CW under array vs UT UE model (32T4R CDL-C, 6.7 GHz)",
                 fontweight="bold")
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/codeword_mapping_ut.png", dpi=130)
    print("saved results/codeword_mapping_ut.png")


if __name__ == "__main__":
    main()

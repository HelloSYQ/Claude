#!/usr/bin/env python3
"""Codeword-to-layer mapping study: 4-layer transmission, 1 codeword vs 4.

Setup (as requested):
  * CDL-C channel, RMS delay spread 30 ns
  * 24 RB, 30 kHz SCS (8.64 MHz), FR1 @ 3.5 GHz
  * 32T4R, dual-polarized UPA panels; rank fixed at 4
  * SVD closed-loop precoding (ideal CSI), equal power split across layers
  * UE ~3 km/h (f_d ~ 10 Hz)
  * each scheme adapts its MCS naturally (max expected throughput):
        - 1 codeword : one MCS for the joint 4-layer block (aggregate MIESM SINR)
        - 4 codewords: each layer is its own codeword with its own MCS/BLER

Outputs SE vs SNR for both schemes plus, for 4-CW, the per-layer selected MCS
(showing why per-codeword adaptation helps when the eigenmode SINRs differ).
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

# ---- fixed scenario parameters -------------------------------------------
N_RB      = 24
SCS       = 30e3
MU_BW     = N_RB * 12 * SCS                 # occupied bandwidth (Hz)
SLOT_TIME = 1e-3 / (2 ** 1)                 # mu=1 -> 0.5 ms
N_SYM     = 13                              # PDSCH data symbols in slot
N_DMRS_RE = 24                              # DM-RS RE per PRB (type1, pos1)
N_RE_PRB  = tbs_mod.re_per_rb(N_SYM, N_DMRS_RE, 0)
RANK      = 4
N_TX, N_RX = 32, 4
FC        = 3.5e9
FD        = (3.0 / 3.6) * FC / 3e8          # 3 km/h -> Hz
MCS_TABLE = 2                               # up to 256QAM
QMS       = (2, 4, 6, 8)


def choose_mcs(sinr_lin, n_layers):
    """Pick the MCS maximising expected throughput for a block.

    Returns (mcs_index, expected_bits, chosen_bler). ``sinr_lin`` are the
    per-RE post-equaliser SINRs feeding this codeword (all its layers).
    """
    eff = {qm: effective_sinr_miesm(sinr_lin, qm) for qm in QMS}
    best = (0, 0.0, 1.0)
    best_tp = -1.0
    for idx in range(mcs_tables.num_mcs(MCS_TABLE)):
        info = mcs_tables.get_mcs(idx, MCS_TABLE)
        tb = tbs_mod.compute_tbs(N_RE_PRB, N_RB, info.modulation_order,
                                 info.target_code_rate, n_layers)
        bler = bler_from_effective_sinr(eff[info.modulation_order],
                                        info.modulation_order,
                                        info.target_code_rate, tb)
        tp = tb * (1.0 - bler)
        if tp > best_tp:
            best_tp = tp
            best = (idx, tp, bler)
    return best


def make_channel(seed):
    return CDLChannel("CDL-C", delay_spread_ns=30, max_doppler_hz=FD,
                      n_tx=N_TX, n_rx=N_RX, carrier_freq_hz=FC,
                      tx_pol=2, rx_pol=2, tx_layout=(2, 8), rx_layout=(1, 2),
                      rng=np.random.default_rng(seed))


def run_point(snr_db, n_slots=150, base_seed=0):
    noise_var = 10 ** (-snr_db / 10.0)
    freqs = (np.arange(N_RB) - N_RB / 2) * 12 * SCS
    bits_1cw = 0.0
    bits_4cw = 0.0
    layer_mcs = np.zeros(RANK)
    for s in range(n_slots):
        ch = make_channel(base_seed + s)
        H = ch.frequency_response(freqs, 0.0)              # [n_rb, 4, 32]
        W = svd_precoder(H, RANK) / np.sqrt(RANK)          # equal power split
        sinr = batch_mmse_sinr(H @ W, noise_var)           # [n_rb, 4] linear

        # 1 codeword: one MCS over all 4 layers' REs
        _, tp1, _ = choose_mcs(sinr.reshape(-1), RANK)
        bits_1cw += tp1

        # 4 codewords: each layer its own MCS
        for k in range(RANK):
            idx, tpk, _ = choose_mcs(sinr[:, k], 1)
            bits_4cw += tpk
            layer_mcs[k] += idx
    se_1 = bits_1cw / n_slots / SLOT_TIME / MU_BW
    se_4 = bits_4cw / n_slots / SLOT_TIME / MU_BW
    return se_1, se_4, layer_mcs / n_slots


def main():
    snrs = np.arange(-5, 26, 2.5)
    se1, se4, lmcs = [], [], []
    print(f"32T4R CDL-C DS=30ns, {N_RB} RB, rank {RANK}, SVD, f_d={FD:.1f} Hz")
    print(f"{'SNR':>6} {'SE 1-CW':>9} {'SE 4-CW':>9} {'gain%':>7}  per-layer MCS (4-CW)")
    for snr in snrs:
        s1, s4, lm = run_point(float(snr))
        se1.append(s1); se4.append(s4); lmcs.append(lm)
        gain = (s4 / s1 - 1) * 100 if s1 > 0 else 0
        print(f"{snr:6.1f} {s1:9.3f} {s4:9.3f} {gain:6.1f}%  "
              f"[{lm[0]:4.1f} {lm[1]:4.1f} {lm[2]:4.1f} {lm[3]:4.1f}]")

    lmcs = np.array(lmcs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.plot(snrs, se1, "o-", color="#2563eb", label="1 codeword (joint)")
    a1.plot(snrs, se4, "s-", color="#dc2626", label="4 codewords (per-layer)")
    a1.set_xlabel("SNR (dB)"); a1.set_ylabel("spectral efficiency (b/s/Hz)")
    a1.set_title("4-layer SE: 1-CW vs 4-CW  (32T4R, CDL-C, SVD)")
    a1.grid(True, alpha=0.3); a1.legend()

    for k in range(RANK):
        a2.plot(snrs, lmcs[:, k], marker=".", label=f"layer {k} (σ{k+1})")
    a2.set_xlabel("SNR (dB)"); a2.set_ylabel("mean selected MCS index (4-CW)")
    a2.set_title("Per-layer MCS: strong eigenmodes get higher MCS")
    a2.grid(True, alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/codeword_mapping_study.png", dpi=130)
    print("saved results/codeword_mapping_study.png")


if __name__ == "__main__":
    main()

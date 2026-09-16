#!/usr/bin/env python3
"""Effect of adding 1024QAM on the 1-CW vs 4-CW codeword-mapping comparison.

Re-runs the 4-layer 1-CW vs 4-CW study (32T4R CDL-C DS=30 ns, SVD, genie MCS)
with two MCS tables:
  * table 2 -- up to 256QAM (Qm 8), the earlier study.
  * table 4 -- 256QAM extended with 1024QAM (Qm 10) at the top.

Question: does lifting the modulation ceiling change the high-SNR behaviour,
where the 256QAM saturation made 4-CW tie / slightly lose to 1-CW?
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
FC = 3.5e9
FD = (3.0 / 3.6) * FC / 3e8
QMS = (2, 4, 6, 8, 10)                       # include 1024QAM


def choose_mcs(sinr_lin, n_layers, table):
    eff = {qm: effective_sinr_miesm(sinr_lin, qm) for qm in QMS}
    best = 0.0
    for idx in range(mcs_tables.num_mcs(table)):
        info = mcs_tables.get_mcs(idx, table)
        tb = tbs_mod.compute_tbs(N_RE_PRB, N_RB, info.modulation_order,
                                 info.target_code_rate, n_layers)
        bler = bler_from_effective_sinr(eff[info.modulation_order],
                                        info.modulation_order,
                                        info.target_code_rate, tb)
        best = max(best, tb * (1 - bler))
    return best


def run_point(snr_db, table, n_slots=120, base=0):
    nv = 10 ** (-snr_db / 10.0)
    freqs = (np.arange(N_RB) - N_RB / 2) * 12 * SCS
    b1 = b4 = 0.0
    for s in range(n_slots):
        ch = CDLChannel("CDL-C", 30, FD, n_tx=N_TX, n_rx=N_RX, carrier_freq_hz=FC,
                        tx_pol=2, rx_pol=2, tx_layout=(2, 8), rx_layout=(1, 2),
                        rng=np.random.default_rng(base + s))
        H = ch.frequency_response(freqs, 0.0)
        W = svd_precoder(H, RANK) / np.sqrt(RANK)
        sinr = batch_mmse_sinr(H @ W, nv)
        b1 += choose_mcs(sinr.reshape(-1), RANK, table)
        b4 += sum(choose_mcs(sinr[:, k], 1, table) for k in range(RANK))
    return (b1 / n_slots / SLOT_TIME / MU_BW,
            b4 / n_slots / SLOT_TIME / MU_BW)


def main():
    snrs = np.arange(-5, 41, 2.5)
    res = {2: ([], []), 4: ([], [])}
    for table, name in [(2, "256QAM"), (4, "+1024QAM")]:
        print(f"\n=== MCS table {table} ({name}) ===")
        print(f"{'SNR':>6} {'SE 1CW':>8} {'SE 4CW':>8} {'gain%':>7}")
        for snr in snrs:
            s1, s4 = run_point(float(snr), table)
            res[table][0].append(s1); res[table][1].append(s4)
            g = (s4 / s1 - 1) * 100 if s1 > 0 else 0
            print(f"{snr:6.1f} {s1:8.3f} {s4:8.3f} {g:6.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
    ax[0].plot(snrs, res[2][0], "o-", color="#93c5fd", label="1-CW, 256QAM")
    ax[0].plot(snrs, res[2][1], "s-", color="#fca5a5", label="4-CW, 256QAM")
    ax[0].plot(snrs, res[4][0], "o-", color="#2563eb", label="1-CW, +1024QAM")
    ax[0].plot(snrs, res[4][1], "s-", color="#dc2626", label="4-CW, +1024QAM")
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("spectral efficiency (b/s/Hz)")
    ax[0].set_title("4-layer SE: 256QAM vs +1024QAM (32T4R CDL-C)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    for table, c, lab in [(2, "#6b7280", "256QAM"), (4, "#16a34a", "+1024QAM")]:
        g = [(a / b - 1) * 100 for a, b in zip(res[table][1], res[table][0])]
        ax[1].plot(snrs, g, "^-", color=c, label=lab)
    ax[1].axhline(0, ls="--", color="gray")
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("4-CW gain over 1-CW (%)")
    ax[1].set_title("4-CW advantage: does 1024QAM lift the high-SNR ceiling?")
    ax[1].grid(alpha=0.3); ax[1].legend()
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/codeword_mapping_1024qam.png", dpi=130)
    print("saved results/codeword_mapping_1024qam.png")


if __name__ == "__main__":
    main()

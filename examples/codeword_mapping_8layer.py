#!/usr/bin/env python3
"""Codeword mapping at high rank: 2 CW vs 4 CW for up to 8 layers.

32T8R (rank up to 8), UT handheld UE model, CDL-C DS=30 ns. Compares two
codeword-to-layer mappings under rank adaptation:

  * 2 CW  -- NR's rank 5-8 scheme: up to 2 codewords, layers grouped by
             eigenmode strength (e.g. rank 8 -> CW0={sigma1..4}, CW1={sigma5..8}).
  * 4 CW  -- up to 4 codewords (rank 8 -> 4 CWs of 2 eigenmodes each).

Each codeword carries one MCS over its layers, chosen for max expected
throughput (genie inner-loop). The rank (RI) is chosen per slot to maximise the
scheme's throughput. Layers are SVD eigenmode-ordered, so contiguous grouping
minimises the within-codeword SINR spread.

Because 32T8R CDL-C has a large eigenmode spread (~25 dB, sigma1..sigma8), 2 CWs
still leave a big within-codeword spread at rank 8 that 4 CWs can tighten -- the
regime where extra codewords help, unlike rank<=4.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim import mcs_tables, tbs as tbs_mod
from nrdlsim.channel_models import CDLChannel
from nrdlsim.receiver import batch_mmse_sinr
from nrdlsim.link_abstraction import effective_sinr_miesm, bler_from_effective_sinr

N_RB, SCS = 24, 30e3
MU_BW = N_RB * 12 * SCS
SLOT_TIME = 0.5e-3
N_RE_PRB = tbs_mod.re_per_rb(13, 24, 0)
N_TX, N_RX, MAX_RANK = 32, 8, 8
FC = 6.7e9
FD = (3.0 / 3.6) * FC / 3e8
QMS = (2, 4, 6, 8)


def choose_mcs(sinr_lin, n_layers):
    """Max expected throughput (bits) over the MCS table for a codeword."""
    eff = {qm: effective_sinr_miesm(sinr_lin, qm) for qm in QMS}
    best = 0.0
    for idx in range(mcs_tables.num_mcs(2)):
        info = mcs_tables.get_mcs(idx, 2)
        tb = tbs_mod.compute_tbs(N_RE_PRB, N_RB, info.modulation_order,
                                 info.target_code_rate, n_layers)
        bler = bler_from_effective_sinr(eff[info.modulation_order],
                                        info.modulation_order,
                                        info.target_code_rate, tb)
        best = max(best, tb * (1 - bler))
    return best


def scheme_throughput(sinr, ncw_max):
    """Throughput of a rank-`r` transmission split into up to ncw_max codewords.

    sinr: [n_rb, r] per-layer SINRs (eigenmode-ordered). Layers are grouped
    contiguously (by strength) into min(ncw_max, r) codewords.
    """
    r = sinr.shape[1]
    groups = np.array_split(np.arange(r), min(ncw_max, r))
    return sum(choose_mcs(sinr[:, g].reshape(-1), len(g)) for g in groups)


def make_ut_channel(seed, orient_rng):
    return CDLChannel("CDL-C", 30, FD, n_tx=N_TX, n_rx=N_RX, carrier_freq_hz=FC,
                      tx_pol=2, rx_pol=2, tx_layout=(2, 8), rx_layout=(2, 2),
                      rx_pattern="38.901",
                      rx_boresight_az_deg=orient_rng.uniform(0, 360),
                      rx_downtilt_deg=orient_rng.uniform(70, 110),
                      element_max_gain_dbi=5.0, element_hpbw_deg=90.0,
                      rng=np.random.default_rng(seed))


def run_point(snr_db, n_slots=120, base=0):
    nv = 10 ** (-snr_db / 10.0)
    freqs = (np.arange(N_RB) - N_RB / 2) * 12 * SCS
    orient_rng = np.random.default_rng(555 + int(snr_db))
    bits = {2: 0.0, 4: 0.0}
    rank_sum = {2: 0, 4: 0}
    for s in range(n_slots):
        H = make_ut_channel(base + s, orient_rng).frequency_response(freqs, 0.0)
        _, _, vh = np.linalg.svd(H, full_matrices=False)
        V = np.conj(np.swapaxes(vh, -1, -2))                 # [n_rb, 32, 8]
        per_rank = {}
        for r in range(1, MAX_RANK + 1):
            Wr = V[:, :, :r] / np.sqrt(r)
            per_rank[r] = batch_mmse_sinr(H @ Wr, nv)        # [n_rb, r]
        for ncw in (2, 4):
            best_r, best_tp = 1, -1.0
            for r, sinr in per_rank.items():
                tp = scheme_throughput(sinr, ncw)
                if tp > best_tp:
                    best_tp, best_r = tp, r
            bits[ncw] += best_tp
            rank_sum[ncw] += best_r
    se = {c: bits[c] / n_slots / SLOT_TIME / MU_BW for c in (2, 4)}
    rk = {c: rank_sum[c] / n_slots for c in (2, 4)}
    return se, rk


def main():
    snrs = np.arange(-5, 31, 2.5)
    se2, se4, rk2, rk4 = [], [], [], []
    print("32T8R UT CDL-C DS=30ns, upper 6 GHz -- 2 CW vs 4 CW (up to 8 layers)")
    print(f"{'SNR':>6} {'SE 2CW':>8} {'SE 4CW':>8} {'gain%':>7} "
          f"{'rank2':>6} {'rank4':>6}")
    for snr in snrs:
        se, rk = run_point(float(snr))
        se2.append(se[2]); se4.append(se[4]); rk2.append(rk[2]); rk4.append(rk[4])
        g = (se[4] / se[2] - 1) * 100 if se[2] > 0 else 0
        print(f"{snr:6.1f} {se[2]:8.3f} {se[4]:8.3f} {g:6.1f} "
              f"{rk[2]:6.1f} {rk[4]:6.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(snrs, se2, "o-", color="#2563eb", label="2 codewords (NR rank5-8)")
    ax[0].plot(snrs, se4, "s-", color="#dc2626", label="4 codewords")
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("spectral efficiency (b/s/Hz)")
    ax[0].set_title("2 CW vs 4 CW up to 8 layers (32T8R UT CDL-C)")
    ax[0].grid(alpha=0.3); ax[0].legend()
    gain = [(a / b - 1) * 100 for a, b in zip(se4, se2)]
    ax[1].plot(snrs, gain, "^-", color="#16a34a")
    ax[1].axhline(0, ls="--", color="gray")
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("4-CW gain over 2-CW (%)")
    ax[1].set_title("4-CW advantage vs SNR"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/codeword_mapping_8layer.png", dpi=130)
    print("saved results/codeword_mapping_8layer.png")


if __name__ == "__main__":
    main()

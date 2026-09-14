#!/usr/bin/env python3
"""Rank adaptation vs fixed 8-layer transmission (2-CW and 4-CW).

Overlays the two rank strategies from codeword_mapping_8layer.py:
  * adaptive : the UE reports the rank (1..8) that maximises throughput.
  * fixed 8  : all 8 layers are transmitted every slot.

for both codeword mappings (2-CW, 4-CW), to show that adaptation gives higher
SE at low/mid SNR (do not over-provision rank) and that the two strategies
converge at high SNR where rank 8 is genuinely optimal.

32T8R UT handheld UE, CDL-C DS=30 ns, upper 6 GHz.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from examples.codeword_mapping_8layer import run_point


def main():
    snrs = np.arange(-5, 31, 2.5)
    A = {2: [], 4: []}     # adaptive
    F = {2: [], 4: []}     # fixed rank 8
    rank_adapt = []
    print(f"{'SNR':>6} | {'adaptSE 2CW':>11} {'4CW':>7} rank | "
          f"{'fix8SE 2CW':>10} {'4CW':>7}")
    for snr in snrs:
        se_a, rk_a = run_point(float(snr), fixed_rank=None)
        se_f, _ = run_point(float(snr), fixed_rank=8)
        for c in (2, 4):
            A[c].append(se_a[c]); F[c].append(se_f[c])
        rank_adapt.append(rk_a[4])
        print(f"{snr:6.1f} | {se_a[2]:11.2f} {se_a[4]:7.2f} {rk_a[4]:4.1f} | "
              f"{se_f[2]:10.2f} {se_f[4]:7.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
    ax[0].plot(snrs, A[2], "o-", color="#2563eb", label="2-CW, adaptive rank")
    ax[0].plot(snrs, A[4], "s-", color="#dc2626", label="4-CW, adaptive rank")
    ax[0].plot(snrs, F[2], "o--", color="#93c5fd", label="2-CW, fixed rank 8")
    ax[0].plot(snrs, F[4], "s--", color="#fca5a5", label="4-CW, fixed rank 8")
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("spectral efficiency (b/s/Hz)")
    ax[0].set_title("Adaptive rank vs fixed rank 8 (32T8R UT CDL-C)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)

    ax[1].plot(snrs, rank_adapt, "o-", color="#16a34a", label="adaptive rank (RI)")
    ax[1].axhline(8, ls="--", color="gray", label="fixed rank 8")
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("layers transmitted")
    ax[1].set_title("Rank used: adaptation ramps 3 -> 8 with SNR")
    ax[1].set_ylim(0, 8.6); ax[1].grid(alpha=0.3); ax[1].legend()

    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/codeword_mapping_rank_compare.png", dpi=130)
    print("saved results/codeword_mapping_rank_compare.png")


if __name__ == "__main__":
    main()

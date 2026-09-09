#!/usr/bin/env python3
"""Singular-value distribution of a MIMO CDL channel.

Evaluates the per-subcarrier channel singular values over many realizations to
show the eigenmode-power profile (effective spatial rank). Default: 32T8R
CDL-C, 100 samples.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim.channel_models import CDLChannel

N_TX, N_RX = 32, 8
N_SAMPLES = 100
SCS, N_RB = 30e3, 24
FC = 3.5e9


def main():
    freqs = (np.arange(N_RB) - N_RB / 2) * 12 * SCS
    svals = np.zeros((N_SAMPLES, N_RX))
    for i in range(N_SAMPLES):
        ch = CDLChannel("CDL-C", delay_spread_ns=300, max_doppler_hz=10,
                        n_tx=N_TX, n_rx=N_RX, carrier_freq_hz=FC,
                        tx_pol=2, rx_pol=2, tx_layout=(2, 8), rx_layout=(2, 2),
                        rng=np.random.default_rng(1000 + i))
        H = ch.frequency_response(freqs, 0.0)              # [n_rb, 8, 32]
        svals[i] = np.linalg.svd(H[N_RB // 2], compute_uv=False)  # centre RB

    pow_db = 20 * np.log10(svals)                          # 10log10(sigma^2)
    rel_db = pow_db - pow_db[:, [0]]                       # relative to sigma_1

    print(f"{N_TX}T{N_RX}R CDL-C, {N_SAMPLES} samples, {N_RX} singular values")
    print(f"{'mode':>4} {'mean sigma':>11} {'sigma^2 (dB)':>13} "
          f"{'rel s1 (dB)':>12} {'P10..P90':>20}")
    for k in range(N_RX):
        p10, p90 = np.percentile(svals[:, k], [10, 90])
        print(f"{k+1:>4} {svals[:,k].mean():>11.3f} {pow_db[:,k].mean():>13.2f} "
              f"{rel_db[:,k].mean():>12.2f} {p10:>9.3f}..{p90:<9.3f}")
    cond = svals[:, 0] / svals[:, -1]
    print(f"condition number sigma_1/sigma_{N_RX}: mean={cond.mean():.1f}, "
          f"median={np.median(cond):.1f}, P90={np.percentile(cond,90):.1f}")
    print(f"sum(sigma^2) mean = {np.sum(svals**2,axis=1).mean():.1f} "
          f"(~ n_rx*n_tx = {N_RX*N_TX})")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].violinplot([svals[:, k] for k in range(N_RX)],
                     positions=range(1, N_RX + 1), showmedians=True)
    ax[0].set_xlabel("singular value index (mode)")
    ax[0].set_ylabel(r"singular value $\sigma_i$")
    ax[0].set_title(f"{N_TX}T{N_RX}R CDL-C: distribution of all {N_RX} singular values")
    ax[0].grid(alpha=0.3)
    mean_rel = rel_db.mean(axis=0)
    p10 = np.percentile(rel_db, 10, axis=0); p90 = np.percentile(rel_db, 90, axis=0)
    ax[1].plot(range(1, N_RX + 1), mean_rel, "o-", color="#2563eb")
    ax[1].fill_between(range(1, N_RX + 1), p10, p90, alpha=0.2, color="#2563eb",
                       label="P10-P90")
    ax[1].set_xlabel("eigenmode index")
    ax[1].set_ylabel(r"$10\log_{10}(\sigma_i^2/\sigma_1^2)$  (dB)")
    ax[1].set_title("Eigenmode power relative to strongest mode")
    ax[1].grid(alpha=0.3); ax[1].legend()
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sv_distribution_32t8r.png", dpi=130)
    print("saved results/sv_distribution_32t8r.png")


if __name__ == "__main__":
    main()

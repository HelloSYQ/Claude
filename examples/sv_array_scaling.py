#!/usr/bin/env python3
"""Singular-value distribution vs array size at upper 6 GHz (CDL-C, 8 Rx).

Two points:
  1. At a fixed array geometry (spacing in wavelengths), the singular-value
     distribution is independent of carrier frequency: 3.5 GHz and 6.7 GHz 32T8R
     are identical. The CDL spatial structure lives in wavelengths, not Hz.
  2. The meaningful upper-6-GHz change is the LARGE array that fits the same
     aperture. Growing the tx UPA (32 -> 256, dual-pol) raises the absolute
     singular values (~sqrt(Ntx) beamforming gain) and modestly improves the
     conditioning, but the effective spatial rank stays channel-limited (CDL-C's
     angular richness), not array-limited.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim.channel_models import CDLChannel

N_RX, N_SAMPLES, SCS, N_RB = 8, 100, 30e3, 24
freqs = (np.arange(N_RB) - N_RB / 2) * 12 * SCS

CONFIGS = [
    ("3.5GHz 32T8R", 32, (2, 8), 3.5e9, "#94a3b8", "--"),
    ("6.7GHz 32T8R", 32, (2, 8), 6.7e9, "#2563eb", "-"),
    ("6.7GHz 64T8R", 64, (4, 8), 6.7e9, "#16a34a", "-"),
    ("6.7GHz 128T8R", 128, (8, 8), 6.7e9, "#f59e0b", "-"),
    ("6.7GHz 256T8R", 256, (8, 16), 6.7e9, "#dc2626", "-"),
]


def eval_cfg(n_tx, layout, fc):
    sv = np.zeros((N_SAMPLES, N_RX))
    for i in range(N_SAMPLES):
        ch = CDLChannel("CDL-C", 300, 10, n_tx=n_tx, n_rx=N_RX, carrier_freq_hz=fc,
                        tx_pol=2, rx_pol=2, tx_layout=layout, rx_layout=(2, 2),
                        rng=np.random.default_rng(1000 + i))
        sv[i] = np.linalg.svd(ch.frequency_response(freqs, 0.0)[N_RB // 2],
                              compute_uv=False)
    return sv


def main():
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
    print(f"{'config':>15} | " + " ".join(f"m{k+1}" for k in range(8))
          + " (dB rel s1) | cond  s1")
    res = {}
    for name, nt, lay, fc, col, ls in CONFIGS:
        sv = eval_cfg(nt, lay, fc); res[name] = (sv, col, ls)
        rel = 20 * np.log10(sv.mean(0) / sv[:, 0].mean())
        cond = (sv[:, 0] / sv[:, -1]).mean()
        print(f"{name:>15} | " + " ".join(f"{v:3.0f}" for v in rel)
              + f" | {cond:4.1f} {sv[:,0].mean():5.1f}")
        ax[0].plot(range(1, 9), rel, marker="o", ls=ls, color=col, label=name)
        if name.startswith("6.7"):
            ax[1].plot(range(1, 9), sv.mean(0), marker="s", color=col, label=name)
    ax[0].set_xlabel("eigenmode index")
    ax[0].set_ylabel(r"$10\log_{10}(\sigma_i^2/\sigma_1^2)$ (dB)")
    ax[0].set_title("Eigenmode profile vs array size @ upper 6 GHz (CDL-C, 8 Rx)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("singular value index"); ax[1].set_ylabel(r"mean $\sigma_i$")
    ax[1].set_title("Absolute singular values: larger array -> beamforming gain")
    ax[1].grid(alpha=0.3); ax[1].legend(fontsize=8)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/sv_upper6ghz.png", dpi=130)
    print("saved results/sv_upper6ghz.png")


if __name__ == "__main__":
    main()

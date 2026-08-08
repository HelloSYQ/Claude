#!/usr/bin/env python3
"""Demonstrate the time-domain OFDM waveform path and RF impairments.

Produces results/waveform_impairments.png with two panels:
  * effective SINR (from EVM) vs carrier frequency offset, against the
    analytic inter-carrier-interference ceiling;
  * effective SINR vs symbol-timing offset, showing the cyclic prefix protects
    the signal until the offset leaves the CP window.
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim.ofdm import OFDMModulator, waveform_evm, cfo_ici_sinr_db


def main():
    scs = 30e3
    mod = OFDMModulator(n_rb=51, scs_hz=scs, mu=1)
    cp = int(mod.cp[1])
    rng = np.random.default_rng(0)
    g = ((rng.integers(0, 2, (mod.n_sc, mod.n_sym)) * 2 - 1)
         + 1j * (rng.integers(0, 2, (mod.n_sc, mod.n_sym)) * 2 - 1)) / np.sqrt(2)
    h = np.zeros(20, complex); h[0] = 1; h[5] = 0.5; h[12] = 0.3j
    snr = 45.0

    print(f"FFT={mod.n_fft}  fs={mod.fs/1e6:.2f} MHz  CP={cp} samples  "
          f"slot={mod.slot_length} samples")

    # --- CFO sweep ---
    eps = np.linspace(0, 0.06, 13)
    sinr_cfo, ceiling = [], []
    for e in eps:
        _, s = waveform_evm(mod, g, snr, cfo_hz=e * scs, h_time=h,
                            rng=np.random.default_rng(1))
        sinr_cfo.append(s)
        ceiling.append(cfo_ici_sinr_db(e * scs, scs) if e > 0 else np.nan)

    # --- timing sweep ---
    offs = np.arange(-cp - 60, 61, 6)
    sinr_to = []
    for to in offs:
        _, s = waveform_evm(mod, g, snr, timing_offset=int(to), h_time=h,
                            rng=np.random.default_rng(2))
        sinr_to.append(s)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    ax1.plot(eps, sinr_cfo, "o-", color="#2563eb", label="waveform (measured)")
    ax1.plot(eps, ceiling, "--", color="#dc2626", label="analytic ICI ceiling")
    ax1.axhline(snr, ls=":", color="gray", label=f"AWGN floor ({snr:.0f} dB)")
    ax1.set_xlabel("normalised CFO  ε = f_CFO / SCS")
    ax1.set_ylabel("effective SINR (dB)")
    ax1.set_title("CFO → inter-carrier interference")
    ax1.grid(True, alpha=0.3); ax1.legend()

    ax2.plot(offs, sinr_to, "s-", color="#16a34a")
    ax2.axvspan(-cp, 0, color="#16a34a", alpha=0.1, label="within CP")
    ax2.set_xlabel("symbol-timing offset (samples)")
    ax2.set_ylabel("effective SINR (dB)")
    ax2.set_title(f"Timing offset vs cyclic prefix (CP = {cp} samples)")
    ax2.grid(True, alpha=0.3); ax2.legend()

    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/waveform_impairments.png", dpi=120)
    print("Saved results/waveform_impairments.png")


if __name__ == "__main__":
    main()

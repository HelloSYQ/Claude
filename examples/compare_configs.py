#!/usr/bin/env python3
"""Compare spectral efficiency across NR channel models and MIMO orders.

Produces results/compare_se.png with two panels:
  * SE vs SNR for TDL-A/C/D and AWGN (2x2)
  * SE vs SNR for 1x1, 2x2 and 4x4 MIMO (TDL-C)
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim.config import (SimConfig, CarrierConfig, PDSCHConfig,
                            AntennaConfig, ChannelConfig)
from nrdlsim.link_simulator import NRDownlinkSimulator


def sweep(model, ntx, nrx, snr=(-5, 30, 2.5), slots=80):
    cfg = SimConfig(
        carrier=CarrierConfig(mu=1, n_size_grid=51),
        pdsch=PDSCHConfig(num_rb=51, num_layers=min(ntx, nrx), mcs_table=2),
        antenna=AntennaConfig(n_tx=ntx, n_rx=nrx, correlation="low"),
        channel=ChannelConfig(model=model, delay_spread_ns=100, max_doppler_hz=100),
        snr_db_range=snr, num_slots=slots, seed=7)
    sim = NRDownlinkSimulator(cfg)
    res = sim.run()
    return np.array([r.snr_db for r in res]), np.array([r.spectral_efficiency
                                                        for r in res])


def main():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    print("Channel-model comparison (2x2)...")
    for model, col in [("AWGN", "#111827"), ("TDL-C", "#2563eb"),
                       ("CDL-C", "#16a34a"), ("CDL-D", "#dc2626")]:
        snr, se = sweep(model, 2, 2)
        ax1.plot(snr, se, "o-", label=model, color=col, lw=1.8, ms=4)
        print(f"  {model:6s} peak SE = {se.max():.2f} b/s/Hz")
    ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("Spectral efficiency (b/s/Hz)")
    ax1.set_title("SE vs channel model (2x2, 256QAM)")
    ax1.grid(True, alpha=0.3); ax1.legend()

    print("MIMO-order comparison (TDL-C)...")
    for (ntx, nrx), col in [((1, 1), "#6b7280"), ((2, 2), "#2563eb"),
                            ((4, 4), "#dc2626")]:
        snr, se = sweep("TDL-C", ntx, nrx)
        ax2.plot(snr, se, "s-", label=f"{ntx}x{nrx}", color=col, lw=1.8, ms=4)
        print(f"  {ntx}x{nrx} peak SE = {se.max():.2f} b/s/Hz")
    ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel("Spectral efficiency (b/s/Hz)")
    ax2.set_title("SE vs MIMO order (TDL-C, 256QAM)")
    ax2.grid(True, alpha=0.3); ax2.legend()

    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/compare_se.png", dpi=120)
    print("Saved results/compare_se.png")


if __name__ == "__main__":
    main()

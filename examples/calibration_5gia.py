#!/usr/bin/env python3
"""Calibrate nrdlsim against the 5G-IA link-level calibration results.

Reference: "5G-IA Link Level Calibration Results v2", Table 3 — SNR (dB) at 70%
of maximum throughput for 9 FR1/FDD evaluation cases (3GPP RAN4 methodology,
CP-OFDM, MMSE, ideal channel estimation).  These PUSCH cases are reproduced on
the (link-symmetric) PDSCH chain with matching MCS / channel / antenna configs.

For each case we sweep SNR at a fixed MCS, measure throughput, and locate the
SNR where throughput reaches 70% of its high-SNR ceiling, then compare to the
5G-PPP WG value and the 8-company average from the document.
"""

import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim.config import (SimConfig, CarrierConfig, PDSCHConfig,
                            AntennaConfig, ChannelConfig, HARQConfig, DMRSConfig)
from nrdlsim.link_simulator import NRDownlinkSimulator

# case: (BW MHz, SCS kHz, antenna, layers, model, DS ns, Doppler Hz, MCS)
CASES = {
    1: (10, 15, (1, 2), 1, "TDL-B", 100, 400, 2),
    2: (10, 15, (1, 2), 1, "TDL-C", 300, 100, 16),
    3: (10, 15, (1, 2), 1, "TDL-A", 30, 10, 20),
    4: (10, 15, (2, 2), 2, "TDL-B", 100, 400, 2),
    5: (10, 15, (2, 2), 2, "TDL-C", 300, 100, 16),
    6: (40, 30, (1, 2), 1, "TDL-B", 100, 400, 2),
    7: (40, 30, (1, 2), 1, "TDL-C", 300, 100, 16),
    8: (40, 30, (1, 2), 1, "TDL-A", 30, 10, 20),
    9: (40, 30, (2, 2), 2, "TDL-B", 100, 400, 2),
}

# Reference SNR (dB) @ 70% throughput from Table 3.
REF_5GPPP = {1: -4.15, 2: 8.6, 3: 10.2, 4: -0.45, 5: 15.45,
             6: -4.15, 7: 8.4, 8: 9.7, 9: -0.1}
REF_AVG = {1: -4.68, 2: 8.24, 3: 10.48, 4: -0.59, 5: 16.31,
           6: -4.68, 7: 7.97, 8: 10.12, 9: -0.96}

# 10 MHz/15 kHz -> 52 PRB; 40 MHz/30 kHz -> 106 PRB (TS 38.101-1 Table 5.3.2-1)
NRB = {(10, 15): 52, (40, 30): 106}
MU = {15: 0, 30: 1}


def snr_at_70pct(snrs, tput):
    ceiling = np.max(tput)
    target = 0.7 * ceiling
    for i in range(1, len(snrs)):
        if tput[i - 1] < target <= tput[i]:
            # linear interpolation
            x0, x1 = snrs[i - 1], snrs[i]
            y0, y1 = tput[i - 1], tput[i]
            return x0 + (target - y0) * (x1 - x0) / (y1 - y0)
    return float("nan")


def run_case(cid, slots=60):
    bw, scs, (ntx, nrx), layers, model, ds, dop, mcs = CASES[cid]
    n_rb = NRB[(bw, scs)]
    ref = REF_5GPPP[cid]
    lo, hi = ref - 7, ref + 7
    cfg = SimConfig(
        carrier=CarrierConfig(mu=MU[scs], n_size_grid=n_rb),
        pdsch=PDSCHConfig(num_rb=n_rb, num_layers=layers, mcs_index=mcs,
                          mcs_table=1, start_symbol=0, num_symbols=14,
                          dmrs=DMRSConfig(additional_positions=1)),
        antenna=AntennaConfig(n_tx=ntx, n_rx=nrx, correlation="low"),
        channel=ChannelConfig(model=model, delay_spread_ns=ds, max_doppler_hz=dop),
        harq=HARQConfig(enabled=True, max_transmissions=4),
        snr_db_range=(lo, hi, 1.0), num_slots=slots,
        link_adaptation=False, ideal_channel_estimation=True,
        precoding="none", seed=100 + cid)
    sim = NRDownlinkSimulator(cfg)
    snrs, tput = [], []
    for i, snr in enumerate(np.arange(lo, hi + 1e-9, 1.0)):
        r = sim.run_point(float(snr), snr_seed=i)
        snrs.append(snr)
        tput.append(r.throughput_bps)
    return snr_at_70pct(np.array(snrs), np.array(tput))


def main():
    print("=" * 78)
    print(" 5G-IA Link-Level Calibration — SNR (dB) @ 70% throughput")
    print("=" * 78)
    print(f"{'Case':>4} {'config':>26} {'sim':>7} {'5G-PPP':>8} {'avg':>7} "
          f"{'Δ(WG)':>7} {'Δ(avg)':>7}")
    print("-" * 78)
    d_wg, d_avg = [], []
    for cid in CASES:
        bw, scs, (ntx, nrx), layers, model, ds, dop, mcs = CASES[cid]
        sim_snr = run_case(cid)
        dw = sim_snr - REF_5GPPP[cid]
        da = sim_snr - REF_AVG[cid]
        d_wg.append(dw); d_avg.append(da)
        cfg = f"{ntx}x{nrx} {model} MCS{mcs}"
        print(f"{cid:>4} {cfg:>26} {sim_snr:7.2f} {REF_5GPPP[cid]:8.2f} "
              f"{REF_AVG[cid]:7.2f} {dw:+7.2f} {da:+7.2f}")
    print("-" * 78)
    d_wg = np.array(d_wg); d_avg = np.array(d_avg)
    print(f" mean Δ vs 5G-PPP WG = {d_wg.mean():+.2f} dB   "
          f"RMS = {np.sqrt(np.mean(d_wg**2)):.2f} dB")
    print(f" mean Δ vs company avg = {d_avg.mean():+.2f} dB   "
          f"RMS = {np.sqrt(np.mean(d_avg**2)):.2f} dB")
    print(f" std of Δ (bias-removed spread) = {d_wg.std():.2f} dB")


if __name__ == "__main__":
    main()

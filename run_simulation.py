#!/usr/bin/env python3
"""Command-line driver for the NR downlink link-level simulator.

Examples
--------
  python run_simulation.py                      # default 4x2, TDL-C, 256QAM
  python run_simulation.py --model AWGN --nrx 1 --ntx 1
  python run_simulation.py --snr -5 30 2.5 --slots 300 --plot
  python run_simulation.py --fec ldpc --slots 40 --snr 0 12 3
"""

from __future__ import annotations

import argparse
import json
import numpy as np

from nrdlsim.config import (SimConfig, CarrierConfig, PDSCHConfig,
                            AntennaConfig, ChannelConfig, HARQConfig)
from nrdlsim.link_simulator import NRDownlinkSimulator


def _parse_layout(s):
    """Parse a 'VxH' array layout string (e.g. '2x2') into a (rows, cols) tuple."""
    if not s:
        return None
    v, h = s.lower().split("x")
    return (int(v), int(h))


def build_config(args) -> SimConfig:
    carrier = CarrierConfig(mu=args.mu, n_size_grid=args.rb)
    pdsch = PDSCHConfig(num_rb=args.rb, num_layers=args.layers,
                        mcs_index=args.mcs, mcs_table=args.mcs_table,
                        num_symbols=args.symbols)
    antenna = AntennaConfig(n_tx=args.ntx, n_rx=args.nrx,
                            correlation=args.correlation,
                            tx_pol=args.tx_pol, rx_pol=args.rx_pol,
                            tx_layout=_parse_layout(args.tx_layout),
                            rx_layout=_parse_layout(args.rx_layout),
                            spacing_v=args.spacing_v, spacing_h=args.spacing_h,
                            tx_pattern=args.tx_pattern, rx_pattern=args.rx_pattern,
                            boresight_az_deg=args.boresight_az,
                            downtilt_deg=args.downtilt,
                            element_max_gain_dbi=args.element_gain)
    channel = ChannelConfig(model=args.model, delay_spread_ns=args.ds,
                            max_doppler_hz=args.doppler,
                            carrier_freq_hz=args.fc)
    harq = HARQConfig(enabled=not args.no_harq)
    return SimConfig(
        carrier=carrier, pdsch=pdsch, antenna=antenna, channel=channel,
        harq=harq,
        snr_db_range=tuple(args.snr),
        num_slots=args.slots,
        csi_feedback_delay_slots=args.csi_delay,
        link_adaptation=not args.fixed_mcs,
        fec_mode=args.fec,
        precoding=args.precoding,
        ideal_channel_estimation=args.ideal_csi,
        seed=args.seed,
    )


def main():
    p = argparse.ArgumentParser(description="NR downlink link-level simulator")
    p.add_argument("--mu", type=int, default=1, help="numerology (SCS=15*2^mu kHz)")
    p.add_argument("--rb", type=int, default=51, help="number of resource blocks")
    p.add_argument("--symbols", type=int, default=13, help="PDSCH OFDM symbols")
    p.add_argument("--layers", type=int, default=2, help="max transmission layers")
    p.add_argument("--mcs", type=int, default=16, help="fixed/initial MCS index")
    p.add_argument("--mcs-table", type=int, default=2, choices=[1, 2, 3])
    p.add_argument("--ntx", type=int, default=4, help="gNB tx antennas")
    p.add_argument("--nrx", type=int, default=2, help="UE rx antennas")
    p.add_argument("--correlation", default="low",
                   choices=["low", "medium", "high"],
                   help="TDL antenna correlation")
    p.add_argument("--tx-pol", type=int, default=1, choices=[1, 2],
                   help="CDL: tx polarizations per position (2 = cross-polar)")
    p.add_argument("--rx-pol", type=int, default=1, choices=[1, 2],
                   help="CDL: rx polarizations per position")
    p.add_argument("--tx-layout", default=None,
                   help="CDL: tx panel 'VxH' element-position grid, e.g. 2x2")
    p.add_argument("--rx-layout", default=None,
                   help="CDL: rx panel 'VxH' element-position grid")
    p.add_argument("--spacing-v", type=float, default=0.5,
                   help="CDL: vertical element spacing (wavelengths)")
    p.add_argument("--spacing-h", type=float, default=0.5,
                   help="CDL: horizontal element spacing (wavelengths)")
    p.add_argument("--tx-pattern", default="omni", choices=["omni", "38.901"],
                   help="CDL tx element pattern: omni or 38.901 directional")
    p.add_argument("--rx-pattern", default="omni", choices=["omni", "38.901"],
                   help="CDL rx element pattern")
    p.add_argument("--downtilt", type=float, default=0.0,
                   help="CDL: tx mechanical downtilt (deg, 38.901 pattern)")
    p.add_argument("--boresight-az", type=float, default=0.0,
                   help="CDL: tx panel boresight azimuth (deg)")
    p.add_argument("--element-gain", type=float, default=8.0,
                   help="CDL: max element gain G_E,max (dBi)")
    p.add_argument("--model", default="TDL-C",
                   help="channel model: TDL-A..E, CDL-A..E or AWGN")
    p.add_argument("--ds", type=float, default=100.0, help="delay spread (ns)")
    p.add_argument("--doppler", type=float, default=100.0, help="max Doppler (Hz)")
    p.add_argument("--fc", type=float, default=3.5e9, help="carrier freq (Hz)")
    p.add_argument("--snr", type=float, nargs=3, default=[-5.0, 30.0, 2.5],
                   metavar=("START", "STOP", "STEP"))
    p.add_argument("--slots", type=int, default=200)
    p.add_argument("--csi-delay", type=int, default=4)
    p.add_argument("--fixed-mcs", action="store_true",
                   help="disable link adaptation (use fixed --mcs)")
    p.add_argument("--no-harq", action="store_true")
    p.add_argument("--fec", default="miesm", choices=["miesm", "ldpc"])
    p.add_argument("--precoding", default="svd", choices=["svd", "none"],
                   help="svd = closed-loop; none = open-loop (no tx precoding)")
    p.add_argument("--ideal-csi", action="store_true",
                   help="perfect channel estimation at the receiver")
    p.add_argument("--seed", type=int, default=2025)
    p.add_argument("--jobs", type=int, default=1,
                   help="parallel worker processes for the SNR sweep (-1 = all cores)")
    p.add_argument("--plot", action="store_true")
    p.add_argument("--out", default="results/se_results.json")
    args = p.parse_args()

    cfg = build_config(args)
    sim = NRDownlinkSimulator(cfg)

    print("=" * 78)
    print(" NR Downlink Link-Level Simulation")
    print("=" * 78)
    print(f" Numerology mu={cfg.carrier.mu}  SCS={cfg.carrier.scs_khz} kHz  "
          f"RB={cfg.carrier.n_size_grid}  BW={cfg.carrier.occupied_bandwidth_hz/1e6:.2f} MHz")
    print(f" MIMO {cfg.antenna.n_tx}x{cfg.antenna.n_rx} ({cfg.antenna.correlation} corr)  "
          f"max rank={min(cfg.antenna.n_tx, cfg.antenna.n_rx)}")
    print(f" Channel: {cfg.channel.model}  DS={cfg.channel.delay_spread_ns} ns  "
          f"Doppler={cfg.channel.max_doppler_hz} Hz")
    print(f" MCS table {cfg.pdsch.mcs_table}  link-adaptation="
          f"{cfg.link_adaptation}  HARQ={cfg.harq.enabled}  FEC={cfg.fec_mode}")
    print(f" Slots/point={cfg.num_slots}  CSI delay={cfg.csi_feedback_delay_slots} slots")
    print("-" * 78)
    header = f"{'SNR[dB]':>8} {'SE[b/s/Hz]':>12} {'Tput[Mbps]':>12} " \
             f"{'BLER':>8} {'avgMCS':>8} {'avgRank':>8} {'avgCQI':>8}"
    print(header)
    print("-" * 78)

    if args.jobs == 1:
        results = []
        for i, snr in enumerate(np.arange(*_range_args(args.snr))):
            r = sim.run_point(float(snr), snr_seed=i)
            results.append(r)
            print(f"{r.snr_db:8.2f} {r.spectral_efficiency:12.4f} "
                  f"{r.throughput_bps/1e6:12.3f} {r.bler:8.3f} "
                  f"{r.avg_mcs:8.2f} {r.avg_rank:8.2f} {r.avg_cqi:8.2f}")
    else:
        results = sim.run(n_jobs=args.jobs)
        for r in results:
            print(f"{r.snr_db:8.2f} {r.spectral_efficiency:12.4f} "
                  f"{r.throughput_bps/1e6:12.3f} {r.bler:8.3f} "
                  f"{r.avg_mcs:8.2f} {r.avg_rank:8.2f} {r.avg_cqi:8.2f}")

    print("-" * 78)
    peak = max(results, key=lambda x: x.spectral_efficiency)
    print(f" Peak spectral efficiency: {peak.spectral_efficiency:.4f} b/s/Hz "
          f"at SNR={peak.snr_db:.1f} dB")

    _save(results, cfg, args.out)
    print(f" Results written to {args.out}")

    if args.plot:
        _plot(results, cfg)


def _range_args(snr):
    start, stop, step = snr
    return (start, stop + 1e-9, step)


def _save(results, cfg, path):
    import os
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    data = {
        "config": {
            "mu": cfg.carrier.mu, "rb": cfg.carrier.n_size_grid,
            "bandwidth_hz": cfg.carrier.occupied_bandwidth_hz,
            "ntx": cfg.antenna.n_tx, "nrx": cfg.antenna.n_rx,
            "channel": cfg.channel.model, "mcs_table": cfg.pdsch.mcs_table,
            "fec_mode": cfg.fec_mode,
        },
        "points": [
            {"snr_db": r.snr_db, "se": r.spectral_efficiency,
             "throughput_bps": r.throughput_bps, "bler": r.bler,
             "avg_mcs": r.avg_mcs, "avg_rank": r.avg_rank,
             "avg_cqi": r.avg_cqi}
            for r in results
        ],
    }
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def _plot(results, cfg):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    snr = [r.snr_db for r in results]
    se = [r.spectral_efficiency for r in results]
    bler = [r.bler for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(snr, se, "o-", color="#2563eb", lw=2)
    ax1.set_xlabel("SNR (dB)"); ax1.set_ylabel("Spectral efficiency (b/s/Hz)")
    ax1.set_title(f"NR DL SE — {cfg.channel.model}, "
                  f"{cfg.antenna.n_tx}x{cfg.antenna.n_rx}")
    ax1.grid(True, alpha=0.3)

    ax2.semilogy(snr, np.clip(bler, 1e-3, 1), "s-", color="#dc2626", lw=2)
    ax2.axhline(cfg.pdsch.target_bler, ls="--", color="gray",
                label=f"target {cfg.pdsch.target_bler}")
    ax2.set_xlabel("SNR (dB)"); ax2.set_ylabel("BLER")
    ax2.set_title("Residual BLER"); ax2.grid(True, which="both", alpha=0.3)
    ax2.legend()

    fig.tight_layout()
    out = "results/se_vs_snr.png"
    fig.savefig(out, dpi=120)
    print(f" Plot written to {out}")


if __name__ == "__main__":
    main()

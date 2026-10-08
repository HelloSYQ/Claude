#!/usr/bin/env python3
"""DL throughput vs SNR for 4T4R, 32T4R and 128T4R (full link-level chain).

Runs NRDownlinkSimulator -- CSI feedback with rank adaptation (RI/PMI/CQI,
4-slot delay), CQI + OLLA link adaptation, HARQ -- for three gNB array sizes:

  * 4T   : dual-pol 1x2 panel   (4 ports)
  * 32T  : dual-pol 2x8 panel   (32 ports)
  * 128T : dual-pol 4x16 panel  (128 ports)

UE 4R (dual-pol 1x2), up to 4 layers (one codeword). CDL-C, DS 30 ns, 3 km/h
at 3.5 GHz, 100 MHz carrier (273 PRB @ 30 kHz), omni elements, SVD precoding
from noisy estimated CSI. Every slot is a downlink slot (no TDD split).

The simulator draws one channel per run, so each SNR point averages N_DROPS
independent CDL drops. Drop d uses the same channel seed at every SNR and for
every array size, so the curves compare the arrays on identical propagation.

Writes results/array_size_throughput.png.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")          # one BLAS thread per worker
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim.config import SimConfig
from nrdlsim.link_simulator import NRDownlinkSimulator

ARRAYS = {"4T4R": (4, (1, 2)), "32T4R": (32, (2, 8)), "128T4R": (128, (4, 16))}
SNRS = [0, 4, 8, 12, 16, 20, 24]
N_DROPS, N_SLOTS, N_RB = 8, 150, 273


def make_cfg(name, drop):
    n_tx, layout = ARRAYS[name]
    c = SimConfig(num_slots=N_SLOTS, seed=1000 + drop)
    c.carrier.n_size_grid = c.pdsch.num_rb = N_RB
    c.pdsch.num_layers = 4
    c.channel.model, c.channel.delay_spread_ns = "CDL-C", 30.0
    c.channel.ue_speed_kmh, c.channel.carrier_freq_hz = 3.0, 3.5e9
    a = c.antenna
    a.n_tx, a.n_rx, a.tx_pol, a.rx_pol = n_tx, 4, 2, 2
    a.tx_layout, a.rx_layout = layout, (1, 2)
    return c


def work(task):
    name, snr, drop = task
    r = NRDownlinkSimulator(make_cfg(name, drop)).run_point(float(snr), 0)
    return name, snr, drop, r


def main():
    tasks = [(n, s, d) for n in ARRAYS for s in SNRS for d in range(N_DROPS)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        out = list(ex.map(work, tasks, chunksize=2))
    agg = {}
    for name, snr, _, r in out:
        agg.setdefault((name, snr), []).append(r)

    bw_mhz = N_RB * 12 * 30e3 / 1e6
    print(f"CDL-C DS 30 ns, 3 km/h @ 3.5 GHz, {N_RB} PRB ({bw_mhz:.1f} MHz), 4R UE, "
          f"{N_DROPS} drops x {N_SLOTS} slots per point\n")
    print(f"{'array':7} {'SNR':>4} | {'Tput Mbps':>9} {'SE b/s/Hz':>9} | {'rank':>4} "
          f"{'MCS':>5} {'BLER1st':>7} {'resBLER':>7}")
    res = {}
    for name in ARRAYS:
        for snr in SNRS:
            rs = agg[(name, snr)]
            tp = np.mean([r.throughput_bps for r in rs]) / 1e6
            se = np.mean([r.spectral_efficiency for r in rs])
            rk = np.mean([r.avg_rank for r in rs])
            res[(name, snr)] = (tp, se, rk)
            print(f"{name:7} {snr:4d} | {tp:9.1f} {se:9.2f} | {rk:4.2f} "
                  f"{np.mean([r.avg_mcs for r in rs]):5.1f} "
                  f"{np.mean([r.bler for r in rs]):7.3f} "
                  f"{np.mean([r.residual_bler for r in rs]):7.3f}")
        print()
    plot(res, bw_mhz)


def plot(res, bw_mhz):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    # reference palette, first three slots (validated all-pairs); distinct
    # markers + legend + direct labels as secondary encoding
    style = {"4T4R": ("#2a78d6", "o"), "32T4R": ("#eb6834", "D"),
             "128T4R": ("#1baf7a", "s")}
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5))
    for ax in (a1, a2):
        ax.grid(True, color=grid, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=ink2, labelsize=9)
        ax.set_xticks(SNRS); ax.set_xlabel("SNR (dB)", color=ink2)
    for name, (col, mk) in style.items():
        tp = [res[(name, s)][0] for s in SNRS]
        rk = [res[(name, s)][2] for s in SNRS]
        a1.plot(SNRS, tp, marker=mk, color=col, lw=2, ms=8, label=name)
        a2.plot(SNRS, rk, marker=mk, color=col, lw=2, ms=8, label=name)
    # ceiling: 4 layers at the top 256QAM MCS (table 2, MCS 27)
    from nrdlsim import tbs as tbs_mod, mcs_tables
    top = mcs_tables.get_mcs(mcs_tables.num_mcs(2) - 1, 2)
    cap = tbs_mod.compute_tbs(tbs_mod.re_per_rb(13, 24, 0), N_RB, top.modulation_order,
                              top.target_code_rate, 4) / 0.5e-3 / 1e6
    a1.axhline(cap, ls="--", lw=1, color=ink2)
    a1.text(SNRS[0], cap, f"  4 layers x 256QAM (MCS 27) ceiling: {cap:.0f} Mbps",
            va="bottom", fontsize=9, color=ink2)
    a1.set_ylim(0, cap * 1.08)
    a1.set_ylabel(f"throughput (Mbps, {bw_mhz:.0f} MHz, all slots DL)", color=ink2)
    a1.set_title("Throughput vs SNR", color=ink)
    a1.legend(frameon=False, loc="lower right")
    a2.set_ylim(0.5, 4.3)
    a2.set_ylabel("average transmission rank", color=ink2)
    a2.set_title("Rank chosen by rank adaptation", color=ink)
    a2.legend(frameon=False, loc="lower right")
    fig.suptitle("4T4R vs 32T4R vs 128T4R — CDL-C 30 ns, 3 km/h, 4R UE, "
                 "SVD precoding, CQI/OLLA link adaptation", color=ink)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/array_size_throughput.png", dpi=120, facecolor="#fcfcfb")
    print("saved results/array_size_throughput.png")


if __name__ == "__main__":
    main()

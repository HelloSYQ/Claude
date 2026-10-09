#!/usr/bin/env python3
"""DL spectral efficiency vs SNR for 4T4R, 32T4R and 128T4R, CDL-C and AWGN.

Runs NRDownlinkSimulator -- CSI feedback with rank adaptation (RI/PMI/CQI,
4-slot delay), CQI + OLLA link adaptation, HARQ -- for three gNB array sizes:

  * 4T   : dual-pol 1x2 panel   (4 ports)
  * 32T  : dual-pol 2x8 panel   (32 ports)
  * 128T : dual-pol 4x16 panel  (128 ports)

UE 4R (dual-pol 1x2), up to 4 layers (one codeword); 100 MHz carrier (273 PRB
@ 30 kHz), SVD precoding from noisy estimated CSI, every slot downlink.

Channels:
  * CDL-C : DS 30 ns, 3 km/h at 3.5 GHz, omni elements.
  * AWGN  : static flat channel, unit gain on every tx-rx link with orthogonal
            rows (see awgn_frequency_response): no fading, no angular
            structure, but each of the 4 eigenmodes keeps the full array
            gain (4, 32 or 128).

Spectral efficiency = delivered bits / (time x allocated bandwidth), b/s/Hz.
Each CDL-C point averages N_DROPS independent drops; drop d uses the same
channel seed at every SNR and array size, so the curves are a paired
comparison. Writes results/array_size_se.png.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")          # one BLAS thread per worker
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim.config import SimConfig
from nrdlsim.link_simulator import NRDownlinkSimulator
from nrdlsim import tbs as tbs_mod, mcs_tables

ARRAYS = {"4T4R": (4, (1, 2)), "32T4R": (32, (2, 8)), "128T4R": (128, (4, 16))}
CHANNELS = ("CDL-C", "AWGN")
SNRS = [0, 4, 8, 12, 16, 20, 24]
N_DROPS, N_SLOTS, N_RB = 8, 150, 273


def make_cfg(channel, name, drop):
    n_tx, layout = ARRAYS[name]
    c = SimConfig(num_slots=N_SLOTS, seed=1000 + drop)
    c.carrier.n_size_grid = c.pdsch.num_rb = N_RB
    c.pdsch.num_layers = 4
    c.channel.model, c.channel.delay_spread_ns = channel, 30.0
    c.channel.ue_speed_kmh, c.channel.carrier_freq_hz = 3.0, 3.5e9
    a = c.antenna
    a.n_tx, a.n_rx, a.tx_pol, a.rx_pol = n_tx, 4, 2, 2
    a.tx_layout, a.rx_layout = layout, (1, 2)
    return c


def work(task):
    channel, name, snr, drop = task
    r = NRDownlinkSimulator(make_cfg(channel, name, drop)).run_point(float(snr), 0)
    return channel, name, snr, r


def se_ceiling():
    """SE of 4 layers at the top MCS of table 2 (256QAM, MCS 27)."""
    top = mcs_tables.get_mcs(mcs_tables.num_mcs(2) - 1, 2)
    tb = tbs_mod.compute_tbs(tbs_mod.re_per_rb(13, 24, 0), N_RB, top.modulation_order,
                             top.target_code_rate, 4)
    return tb / 0.5e-3 / (N_RB * 12 * 30e3)


def main():
    tasks = [(ch, n, s, d) for ch in CHANNELS for n in ARRAYS for s in SNRS
             for d in range(N_DROPS if ch != "AWGN" else 2)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        out = list(ex.map(work, tasks, chunksize=2))
    agg = {}
    for channel, name, snr, r in out:
        agg.setdefault((channel, name, snr), []).append(r)

    print(f"{N_RB} PRB ({N_RB * 12 * 30e-3:.1f} MHz), 4R UE, {N_SLOTS} slots per run; "
          f"CDL-C: {N_DROPS} drops per point\n")
    print(f"{'channel':7} {'array':7} {'SNR':>4} | {'SE b/s/Hz':>9} | {'rank':>4} "
          f"{'MCS':>5} {'BLER1st':>7} {'resBLER':>7}")
    res = {}
    for channel in CHANNELS:
        for name in ARRAYS:
            for snr in SNRS:
                rs = agg[(channel, name, snr)]
                se = np.mean([r.spectral_efficiency for r in rs])
                rk = np.mean([r.avg_rank for r in rs])
                res[(channel, name, snr)] = (se, rk)
                print(f"{channel:7} {name:7} {snr:4d} | {se:9.2f} | {rk:4.2f} "
                      f"{np.mean([r.avg_mcs for r in rs]):5.1f} "
                      f"{np.mean([r.bler for r in rs]):7.3f} "
                      f"{np.mean([r.residual_bler for r in rs]):7.3f}")
            print()
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # colour = array (reference palette, first three slots, validated
    # all-pairs; distinct markers as secondary encoding); line style = channel
    style = {"4T4R": ("#2a78d6", "o"), "32T4R": ("#eb6834", "D"),
             "128T4R": ("#1baf7a", "s")}
    dash = {"CDL-C": "-", "AWGN": "--"}
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2))
    for ax in (a1, a2):
        ax.grid(True, color=grid, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=ink2, labelsize=9)
        ax.set_xticks(SNRS); ax.set_xlabel("SNR (dB)", color=ink2)
    for channel in CHANNELS:
        for name, (col, mk) in style.items():
            se = [res[(channel, name, s)][0] for s in SNRS]
            rk = [res[(channel, name, s)][1] for s in SNRS]
            kw = dict(color=col, marker=mk, ls=dash[channel], lw=2, ms=7,
                      mfc=col if channel == "CDL-C" else "white", mew=1.8)
            a1.plot(SNRS, se, **kw)
            a2.plot(SNRS, rk, **kw)
    cap = se_ceiling()
    a1.axhline(cap, ls=":", lw=1, color=ink2)
    a1.text(SNRS[0], cap, f"  4 layers x 256QAM (MCS 27) ceiling: {cap:.2f} b/s/Hz",
            va="bottom", fontsize=9, color=ink2)
    a1.set_ylim(0, cap * 1.08)
    a1.set_ylabel("spectral efficiency (b/s/Hz)", color=ink2)
    a1.set_title("Spectral efficiency vs SNR", color=ink)
    a2.set_ylim(0.5, 4.3)
    a2.set_ylabel("average transmission rank", color=ink2)
    a2.set_title("Rank chosen by rank adaptation", color=ink)
    handles = [Line2D([], [], color=c, marker=m, lw=2, ms=7, label=n)
               for n, (c, m) in style.items()]
    handles += [Line2D([], [], color=ink2, ls=dash[ch], lw=2, label=ch)
                for ch in CHANNELS]
    a1.legend(handles=handles, frameon=False, loc="lower right", fontsize=9, ncol=2)
    fig.suptitle("4T4R vs 32T4R vs 128T4R, CDL-C (solid) and AWGN (dashed) — "
                 "4R UE, 100 MHz, SVD precoding, CQI/OLLA link adaptation", color=ink)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/array_size_se.png", dpi=120, facecolor="#fcfcfb")
    print("saved results/array_size_se.png")


if __name__ == "__main__":
    main()

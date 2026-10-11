#!/usr/bin/env python3
"""When is rank 4 chosen over rank 3?  32T4R, CDL-A, DS 30 ns.

For many independent CDL-A drops and SNRs, rank 3 and rank 4 (SVD precoding,
total power split over the layers, MMSE receiver, one codeword) are compared
under four decision rules:

  shannon : sum of per-layer log2(1+SINR) -- no MCS table, no codeword
  genie   : best MCS for the single codeword (MIESM + BLER model), i.e. the
            throughput-optimal choice with ideal CSI
  ue_cqi  : the simulator's UE rule (csi.compute_csi): rank x SE of the highest
            CQI meeting 10% BLER -- what the UE reports and the gNB follows
  stale   : genie on the channel 4 slots (2 ms) later at 30 km/h, with the
            precoder from the reported channel -- what is *actually* better
            under mobility

and the full link chain (NRDownlinkSimulator: estimation error, 4-slot CSI
delay, OLLA, HARQ) gives the average rank with ideal vs realistic CSI.

Writes results/rank3_vs_rank4_cdla.png.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim import mcs_tables, tbs as tbs_mod
from nrdlsim.channel_models import CDLChannel
from nrdlsim.receiver import batch_mmse_sinr
from nrdlsim.link_abstraction import effective_sinr_miesm, bler_from_effective_sinr
from nrdlsim.csi import compute_csi, cqi_required_sinr_db
from nrdlsim.config import SimConfig
from nrdlsim.link_simulator import NRDownlinkSimulator

N_RB, SCS, SLOT = 273, 30e3, 0.5e-3
N_RE = tbs_mod.re_per_rb(13, 24, 0)
FC = 3.5e9
FD_30KMH = (30 / 3.6) * FC / 3e8               # ~97 Hz
DELAY = 4 * SLOT                               # CSI report -> transmission
SNRS = np.arange(-12, 37, 2)
N_DROPS = 400
LOW_SNR = -4            # below this, ranks 1-2 win and the 3-vs-4 genie is at MCS 0
MCS_TABLE, CQI_TABLE, TARGET = 2, 2, 0.1
QMS = (2, 4, 6, 8)
FREQS = (np.arange(N_RB) - N_RB / 2) * 12 * SCS
BW = N_RB * 12 * SCS

_TBS = {r: [tbs_mod.compute_tbs(N_RE, N_RB, mcs_tables.get_mcs(i, MCS_TABLE).modulation_order,
                                mcs_tables.get_mcs(i, MCS_TABLE).target_code_rate, r)
            for i in range(mcs_tables.num_mcs(MCS_TABLE))] for r in (1, 2, 3, 4)}


def channel(seed):
    return CDLChannel("CDL-A", 30, FD_30KMH, n_tx=32, n_rx=4, carrier_freq_hz=FC,
                      tx_pol=2, rx_pol=2, tx_layout=(2, 8), rx_layout=(1, 2),
                      rng=np.random.default_rng(seed))


def shannon(sinr):
    return float(np.mean(np.sum(np.log2(1 + sinr), axis=1)))     # bits/RE


def genie(sinr, r):
    """Best expected SE (b/s/Hz) over the MCS table for one codeword."""
    eff = {q: effective_sinr_miesm(sinr.reshape(-1), q) for q in QMS}
    best = 0.0
    for i, tb in enumerate(_TBS[r]):
        m = mcs_tables.get_mcs(i, MCS_TABLE)
        bl = bler_from_effective_sinr(eff[m.modulation_order], m.modulation_order,
                                      m.target_code_rate, tb)
        best = max(best, tb * (1 - bl) / SLOT / BW)
    return best


def ue_cqi(sinr, r):
    """csi.compute_csi's per-rank metric: rank x SE of the reported CQI."""
    qms = sorted({mcs_tables.get_cqi(i, CQI_TABLE)[0] for i in range(1, 16)})
    eff = {q: effective_sinr_miesm(sinr.reshape(-1), q) for q in qms}
    reqs = cqi_required_sinr_db(CQI_TABLE, r, TARGET, N_RE, N_RB)
    cqi = next((i for i in range(15, 0, -1)
                if eff[mcs_tables.get_cqi(i, CQI_TABLE)[0]] >= reqs[i - 1]), 0)
    return r * mcs_tables.get_cqi(cqi, CQI_TABLE)[2], cqi


def drop(seed):
    ch = channel(seed)
    H0 = ch.frequency_response(FREQS, 0.0)               # channel the UE reports on
    Hd = ch.frequency_response(FREQS, DELAY)             # channel at transmission
    _, s, vh = np.linalg.svd(H0, full_matrices=False)
    V = np.conj(np.swapaxes(vh, -1, -2))
    lam = np.mean(s ** 2, axis=0)                        # mean eigenvalues (4,)
    out = {k: np.zeros((len(SNRS), 2)) for k in ("shannon", "genie", "ue", "stale")}
    out["sinr4"] = np.zeros(len(SNRS))                   # layer-4 SINR at rank 4
    out["sinr13"] = np.zeros(len(SNRS))                  # mean SINR, layers 1-3 at rank 3
    for i, snr in enumerate(SNRS):
        nv = 10 ** (-snr / 10)
        for j, r in enumerate((3, 4)):
            W = V[:, :, :r] / np.sqrt(r)
            s0 = batch_mmse_sinr(H0 @ W, nv)
            sd = batch_mmse_sinr(Hd @ W, nv)
            out["shannon"][i, j] = shannon(s0)
            out["genie"][i, j] = genie(s0, r)
            out["ue"][i, j] = ue_cqi(s0, r)[0]
            out["stale"][i, j] = genie(sd, r)
            if r == 4:
                out["sinr4"][i] = 10 * np.log10(s0[:, 3].mean())
            else:
                out["sinr13"][i] = 10 * np.log10(s0.mean())
    out["lam"] = lam
    return out


def check_ue_rule(n=12):
    """ue_cqi must reproduce csi.compute_csi's rank choice exactly."""
    for seed in range(n):
        H = channel(10_000 + seed).frequency_response(FREQS, 0.0)
        _, _, vh = np.linalg.svd(H, full_matrices=False)
        V = np.conj(np.swapaxes(vh, -1, -2))
        for snr in (0.0, 10.0, 20.0, 30.0):
            nv = 10 ** (-snr / 10)
            m = [ue_cqi(batch_mmse_sinr(H @ (V[:, :, :r] / np.sqrt(r)), nv), r)[0]
                 for r in (1, 2, 3, 4)]
            rep = compute_csi(H, nv, 4, MCS_TABLE, TARGET, N_RE, N_RB)
            assert rep.rank == int(np.argmax(m)) + 1, (seed, snr, rep.rank, m)


def full_chain(task):
    snr, seed, ideal = task
    c = SimConfig(num_slots=120, seed=seed, ideal_channel_estimation=ideal,
                  csi_feedback_delay_slots=0 if ideal else 4)
    c.carrier.n_size_grid = c.pdsch.num_rb = N_RB
    c.pdsch.num_layers = 4
    c.channel.model, c.channel.delay_spread_ns = "CDL-A", 30.0
    c.channel.ue_speed_kmh, c.channel.carrier_freq_hz = 3.0, FC
    a = c.antenna
    a.n_tx, a.n_rx, a.tx_pol, a.rx_pol, a.tx_layout, a.rx_layout = 32, 4, 2, 2, (2, 8), (1, 2)
    r = NRDownlinkSimulator(c).run_point(float(snr), 0)
    return snr, ideal, r.avg_rank, r.spectral_efficiency


def main():
    check_ue_rule()
    print("ue_cqi rule reproduces csi.compute_csi rank decisions: OK\n")
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        drops = list(ex.map(drop, range(N_DROPS), chunksize=4))
        fc_snrs = list(range(-12, 25, 4))
        fc = list(ex.map(full_chain, [(s, 3000 + d, ideal) for s in fc_snrs
                                      for d in range(6) for ideal in (True, False)]))

    A = {k: np.stack([d[k] for d in drops]) for k in drops[0]}
    lam_db = 10 * np.log10(A["lam"] / A["lam"][:, :1])
    print("CDL-A 32T4R eigenvalues relative to the strongest (dB): median "
          + ", ".join(f"l{k+1} {np.median(lam_db[:, k]):+.1f}" for k in range(4))
          + f";  l4/l3 median {np.median(lam_db[:, 3] - lam_db[:, 2]):+.1f} dB "
          f"(10-90%: {np.percentile(lam_db[:, 3] - lam_db[:, 2], 10):+.1f} .. "
          f"{np.percentile(lam_db[:, 3] - lam_db[:, 2], 90):+.1f})\n")

    print(f"{'SNR':>4} | P(rank 4 better than rank 3)                | SE b/s/Hz (genie)    "
          f"| layer-4 SINR")
    print(f"{'':>4} | {'shannon':>8} {'genie':>6} {'UE CQI':>7} {'stale 30k':>10} | "
          f"{'rank3':>6} {'rank4':>6} {'stale3':>6} {'stale4':>6} | median dB | "
          f"regret: UE CQI rule, ideal-CSI choice under 30 km/h")
    P = {}
    for k in ("shannon", "genie", "ue", "stale"):
        P[k] = np.mean(A[k][:, :, 1] > A[k][:, :, 0], axis=0)
    # regret: SE lost by a rule's rank choice vs the best of {3, 4}
    g, st = A["genie"], A["stale"]
    best_g, best_s = g.max(axis=2), st.max(axis=2)
    ue_pick = (A["ue"][:, :, 1] > A["ue"][:, :, 0]).astype(int)     # 0 -> rank 3, 1 -> rank 4
    id_pick = (g[:, :, 1] > g[:, :, 0]).astype(int)
    pick = lambda arr, idx: np.take_along_axis(arr, idx[..., None], axis=2)[..., 0]
    reg_ue = 100 * np.mean(1 - pick(g, ue_pick) / best_g, axis=0)
    reg_stale = 100 * np.mean(1 - pick(st, id_pick) / best_s, axis=0)
    for i, snr in enumerate(SNRS):
        print(f"{snr:4d} | {P['shannon'][i]:8.2f} {P['genie'][i]:6.2f} {P['ue'][i]:7.2f} "
              f"{P['stale'][i]:10.2f} | {A['genie'][:, i, 0].mean():6.2f} "
              f"{A['genie'][:, i, 1].mean():6.2f} {A['stale'][:, i, 0].mean():6.2f} "
              f"{A['stale'][:, i, 1].mean():6.2f} | {np.median(A['sinr4'][:, i]):6.1f}"
              f" | {reg_ue[i]:5.1f}% {reg_stale[i]:5.1f}%")

    # decision threshold on the layer-4 SINR (genie rule), pooled below the
    # MCS ceiling (rank-3 SE < 95% of its maximum)
    cap3 = _TBS[3][-1] / SLOT / BW
    unsat = (A["genie"][:, :, 0] < 0.95 * cap3) & (SNRS[None, :] >= LOW_SNR)
    x = A["sinr4"][unsat]; y = (A["genie"][:, :, 1] > A["genie"][:, :, 0])[unsat]
    bins = np.arange(-6, 22, 1.0)
    print(f"\nP(genie picks rank 4 | layer-4 SINR), SNR >= {LOW_SNR} dB, rank 3 below its "
          "MCS ceiling:")
    thr = None
    for lo in bins:
        m = (x >= lo) & (x < lo + 1)
        if m.sum() >= 30:
            p = y[m].mean()
            if thr is None and p >= 0.5:
                thr = lo + 0.5
            print(f"  {lo:+5.1f}..{lo+1:+5.1f} dB: {p:4.2f}  (n={m.sum()})")
    print(f"=> 50% point of layer-4 SINR ~ {thr} dB")

    print("\nFull chain (6 drops x 120 slots): average rank, ideal CSI vs realistic CSI"
          " (estimation error + 4-slot delay, 3 km/h)")
    for s in fc_snrs:
        ri = np.mean([x[2] for x in fc if x[0] == s and x[1]])
        rr = np.mean([x[2] for x in fc if x[0] == s and not x[1]])
        si = np.mean([x[3] for x in fc if x[0] == s and x[1]])
        sr = np.mean([x[3] for x in fc if x[0] == s and not x[1]])
        print(f"  SNR {s:2d}: rank {ri:4.2f} (ideal)  {rr:4.2f} (realistic)   "
              f"SE {si:5.2f} / {sr:5.2f} b/s/Hz")
    plot(A, P, cap3)


def plot(A, P, cap3):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    c = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]          # reference palette 1-4
    fig, ax = plt.subplots(1, 3, figsize=(20, 5.6))
    for a in ax:
        a.grid(True, color=grid, lw=0.8); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=ink2, labelsize=9)

    # (a) probability rank 4 beats rank 3
    lab = {"shannon": "Shannon (no MCS table, no codeword)",
           "genie": "best MCS, 1 codeword (ideal CSI)",
           "ue": "UE CQI rule (what the gNB is told)",
           "stale": "best MCS, CSI 2 ms old at 30 km/h"}
    mk = {"shannon": "o", "genie": "D", "ue": "s", "stale": "^"}
    for k, col in zip(("shannon", "genie", "ue", "stale"), c):
        ax[0].plot(SNRS, P[k], marker=mk[k], color=col, lw=2, ms=6, label=lab[k])
    ax[0].axvspan(SNRS[0] - 1, LOW_SNR - 1, color="#e4e3df", zorder=0,
                  label="ranks 1-2 win here (rank 3 vs 4 both at lowest MCS)")
    ax[0].set_xlim(SNRS[0] - 1, SNRS[-1] + 1)
    ax[0].set_ylim(-0.03, 1.03); ax[0].set_xlabel("SNR (dB)", color=ink2)
    ax[0].set_ylabel("fraction of drops where rank 4 > rank 3", color=ink2)
    ax[0].set_title("How often rank 4 beats rank 3", color=ink)
    ax[0].legend(frameon=False, fontsize=8.5, loc="lower right")

    # (b) decision map in (layers 1-3 SINR, layer-4 SINR)
    w = A["genie"][:, :, 1] > A["genie"][:, :, 0]
    sat = A["genie"][:, :, 0] >= 0.95 * cap3
    ok = SNRS[None, :].repeat(len(xs := A["sinr4"]), 0) >= LOW_SNR
    ys = A["sinr13"] - A["sinr4"]               # how far layer 4 lags layers 1-3
    for m, col, l in ((~w & ~sat, c[0], "rank 3 better"), (w & ~sat, c[1], "rank 4 better")):
        m = m & ok & (xs < 16)
        ax[1].scatter(xs[m], ys[m], s=6, color=col, alpha=0.5, label=l, rasterized=True)
    ax[1].set_xlabel("SINR of layer 4 at rank 4 (dB)", color=ink2)
    ax[1].set_ylabel("layers 1-3 mean SINR (rank 3) minus layer-4 SINR (dB)", color=ink2)
    ax[1].set_title(f"Genie decision map (SNR >= {LOW_SNR} dB, each point = drop x SNR)",
                    color=ink)
    lg = ax[1].legend(frameon=False, fontsize=8.5, loc="upper left", markerscale=4)
    for h in lg.legend_handles:
        h.set_alpha(1)

    # (c) mean SE rank 3 vs rank 4, ideal vs stale CSI
    for j, (col, mkr, name) in enumerate(((c[0], "o", "rank 3"), (c[1], "D", "rank 4"))):
        ax[2].plot(SNRS, A["genie"][:, :, j].mean(0), marker=mkr, color=col, lw=2, ms=6,
                   label=f"{name}, ideal CSI")
        ax[2].plot(SNRS, A["stale"][:, :, j].mean(0), marker=mkr, color=col, lw=2, ms=6,
                   ls="--", mfc="white", label=f"{name}, CSI 2 ms old @ 30 km/h")
    ax[2].set_xlabel("SNR (dB)", color=ink2)
    ax[2].set_ylabel("spectral efficiency (b/s/Hz, best MCS)", color=ink2)
    ax[2].set_title("Average SE: rank 3 vs rank 4", color=ink)
    ax[2].legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.suptitle(f"Rank 3 vs rank 4 — 32T4R, CDL-A, DS 30 ns, {N_RB} PRB, SVD + MMSE, "
                 f"one codeword, {N_DROPS} drops", color=ink)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/rank3_vs_rank4_cdla.png", dpi=115, facecolor="#fcfcfb")
    print("\nsaved results/rank3_vs_rank4_cdla.png")


if __name__ == "__main__":
    main()

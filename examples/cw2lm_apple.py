#!/usr/bin/env python3
"""Codeword-to-layer mapping with Apple's link-level assumptions (R1-2608231).

Apple R1-2608231 (RAN1 #126bis, "Codeword layer mapping") finds large gains
for 2 CW (Alt 2) and one CW per layer (Alt 3) over 1 CW (Alt 1) at fixed
rank 4, but only a few percent once rank adaptation is aware of the
codeword split, and none at 30 km/h.  Nokia R1-2607066 reports 14-25 %.
This script runs Apple's Table 2-1 set-up in nrdlsim and switches on the
differences between the two evaluations one at a time:

  csi    'ideal'     CSI measured in the transmission slot, no ageing;
         'realistic' CSI-RS every 20 ms, 4 ms delay (Apple), Doppler on;
  rank   'fixed4'    rank 4 always (Apple Fig. 2-2, Nokia Fig. 28);
         'adaptive'  per report, the rank maximising sum_CW TBS (1 - BLER)
                     for that mapping (Apple's CW-aware RI);
  la     per-codeword CQI (highest MCS with BLER <= 10 %) + OLLA per CW,
         no HARQ (a failed TB is lost).

Nokia's Table 4 is the envelope over fixed-rank *average* curves; it is
also computed from the 'ideal' / 'fixed r' runs (``nokia_envelope``).

    python examples/cw2lm_apple.py [--drops 40] [--slots 400] [--jobs 4]

Writes results/cw2lm_apple.json and results/cw2lm_apple.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nrdlsim import tbs as tbs_mod                                     # noqa: E402
from nrdlsim.channel_models import (CDLChannel, ut_handset_array,       # noqa: E402
                                    UT_HANDSET_ELEMENT)
from nrdlsim.layer_mapping import layer_groups                         # noqa: E402
from nrdlsim.link_abstraction import _amc_table, effective_sinr_miesm  # noqa: E402
from nrdlsim.receiver import (batch_mmse_sinr, batch_mmse_sinr_est,    # noqa: E402
                              cancelled_sinr, rml_sinr_samples)

OUT = os.path.join(ROOT, "results")

# ---- Apple R1-2608231 Table 2-1 ------------------------------------------
FC = 4e9
SCS = 30e3
N_RB = 51                          # 20 MHz at 30 kHz
SLOT = 0.5e-3
BW = N_RB * 12 * SCS
N_RE = tbs_mod.re_per_rb(13, 12, 0)    # 144 RE/PRB: 4 x 6.35 = 25.4 bit/s/Hz peak
DS_NS = 100.0
CSI_PERIOD = 40                    # slots (20 ms)
CSI_DELAY = 8                      # slots (4 ms)
CE_GAIN = 6.0                      # DM-RS estimate, error variance N0/6
TARGET = 0.1
OLLA_UP = 0.5
OLLA_DN = OLLA_UP * TARGET / (1 - TARGET)
QMS = (2, 4, 6, 8)
MAPPINGS = {"alt1": 1, "alt2": 2, "alt3": 4}
SNRS = np.arange(-10.0, 43.0, 4.0)
CONFIGS = [(csi, rank) for csi in ("ideal", "realistic") for rank in ("fixed4", "adaptive")]
# receivers: 'mmse' (LMMSE), 'sic' (codeword-level MMSE-SIC, CW order, cancel
# only after a CRC pass), 'rml' (ML abstraction, nrdlsim.receiver).  The UE's
# CQI assumes its own receiver (SIC: earlier codewords cancelled).
RECEIVERS = ("mmse",)
RX_CSI = ("realistic",)            # the alternative receivers run with this CSI only


def make_channel(model: str, seed: int, speed_kmh: float) -> CDLChannel:
    rng = np.random.default_rng(seed)
    return CDLChannel(model, DS_NS, speed_kmh / 3.6 * FC / 3e8, n_tx=32, n_rx=4,
                      carrier_freq_hz=FC, tx_pol=2, tx_layout=(2, 8),
                      spacing_v=0.8, spacing_h=0.5, tx_pattern="38.901",
                      rx_elements=ut_handset_array(FC, indices=(1, 3, 5, 7)),
                      rx_pattern="38.901",
                      rx_element_max_gain_dbi=UT_HANDSET_ELEMENT["g_max_dbi"],
                      rx_element_hpbw_deg=UT_HANDSET_ELEMENT["hpbw_deg"],
                      rx_element_front_back_db=UT_HANDSET_ELEMENT["front_back_db"],
                      travel_az_deg=float(rng.uniform(-180, 180)), rng=rng)


def wideband_svd(H: np.ndarray, rank: int) -> np.ndarray:
    """Wideband SVD precoder (PRG = wideband): top eigenvectors of sum H^H H."""
    _, vec = np.linalg.eigh(np.einsum("fus,fut->st", np.conj(H), H))
    return vec[:, ::-1][:, :rank] / np.sqrt(rank)


class Table:
    """Per-MCS arrays for a codeword of n layers; required SINR at 10 % BLER."""

    def __init__(self, n_layers: int):
        qm, tbs, thr, slope = _amc_table(2, n_layers, N_RE, N_RB)
        self.qm, self.tbs, self.thr, self.slope = qm, tbs, thr, slope
        self.req = thr + np.log((1 - TARGET) / TARGET) / slope

    def eff(self, sinr, weights=None):
        e = {q: effective_sinr_miesm(sinr, q, weights) for q in QMS}
        return np.array([e[q] for q in self.qm])

    def select(self, eff, offset_db=0.0):
        ok = np.flatnonzero(eff - offset_db >= self.req)
        return int(ok[-1]) if ok.size else 0

    def bler(self, eff, i):
        return 1.0 / (1.0 + np.exp(self.slope[i] * (eff[i] - self.thr[i])))

    def bler_at(self, sinr, i, weights=None):
        """BLER of MCS i on the SINRs ``sinr`` (MIESM at that MCS's Qm only)."""
        e = effective_sinr_miesm(sinr, int(self.qm[i]), weights)
        return 1.0 / (1.0 + np.exp(self.slope[i] * (e - self.thr[i])))


TABLES = {n: Table(n) for n in (1, 2, 3, 4)}


def cw_sinr(store: dict, G, Gest, n0, groups, k, rx, cancelled=None):
    """(SINR samples, weights) of codeword k for receiver ``rx``.

    SIC: the layers in ``cancelled`` (default: all earlier codewords) are
    removed.  ``store`` caches per (rx, cancelled set)."""
    g = groups[k]
    if rx == "rml":
        if "rml" not in store:
            store["rml"] = rml_sinr_samples(G, Gest, n0)
        smp, w = store["rml"]
        return smp[:, g].ravel(), np.broadcast_to(w[g], smp[:, g].shape).ravel()
    if rx == "sic":
        if cancelled is None:
            cancelled = [x for gg in groups[:k] for x in gg]
        key = ("sic", tuple(sorted(cancelled)))
        if key not in store:
            store[key] = cancelled_sinr(G, Gest, n0, cancelled)
        return store[key][:, g].ravel(), None
    if "mmse" not in store:
        store["mmse"] = batch_mmse_sinr_est(G, Gest, n0)
    return store["mmse"][:, g].ravel(), None


def report_view(rep: dict, rank: int, max_cw: int, rx: str = "mmse"):
    """Cached per-codeword effective SINRs of a report and its RI metric,
    sum over codewords of TBS (1 - BLER) at the CQI MCS (no OLLA).  With
    SIC the CQI of codeword k assumes codewords 1..k-1 cancelled."""
    key = (rank, max_cw, rx)
    if key not in rep["cache"]:
        groups = layer_groups(rank, max_cw)
        store = rep["stores"].setdefault(rank, {})
        G = rep["G"][rank - 1]
        effs, tot = [], 0.0
        for k, g in enumerate(groups):
            t = TABLES[len(g)]
            if rx == "mmse":
                e = t.eff(rep["sinr"][rank - 1][:, g].ravel())
            else:
                e = t.eff(*cw_sinr(store, G, G, rep["n0"], groups, k, rx))
            i = t.select(e)
            tot += t.tbs[i] * (1.0 - t.bler(e, i))
            effs.append(e)
        rep["cache"][key] = (groups, effs, tot)
    return rep["cache"][key]


def _worker(args):
    model, speed, seeds, n_slots, warm, receivers = args
    f = (np.arange(N_RB) - N_RB / 2 + 0.5) * 12 * SCS
    n_snr = len(SNRS)
    keys = [(rx, c, r, m) for rx in receivers for c, r in CONFIGS for m in MAPPINGS
            if rx == "mmse" or c in RX_CSI]
    bits = {k: np.zeros(n_snr) for k in keys}
    ntx = {k: np.zeros(n_snr) for k in keys}
    nack = {k: np.zeros(n_snr) for k in keys}
    rank_sum = {k: np.zeros(n_snr) for k in keys}
    genie = np.zeros((4, 3, n_snr))       # Nokia-style: ideal CSI, ideal AMC, fixed r
    from nrdlsim.link_abstraction import ideal_amc
    for seed in seeds:
        ch = make_channel(model, int(seed), speed)
        rng = np.random.default_rng(10_000 + int(seed))
        # alternative receivers draw their ACKs from their own streams, so
        # the LMMSE results do not depend on which receivers run
        rng_rx = {rx: rng if rx == "mmse" else np.random.default_rng([10_000 + int(seed), i])
                  for i, rx in enumerate(receivers)}
        # state per SNR and config: latest usable report, OLLA offsets per CW
        report = {(j, c): None for j in range(n_snr) for c in ("ideal", "realistic")}
        pending = {j: [] for j in range(n_snr)}
        olla = {(j, k): np.zeros(4) for j in range(n_snr) for k in keys}
        for t in range(n_slots):
            H = ch.frequency_response(f, t * SLOT)                    # (F, 4, 32)
            W = [wideband_svd(H, r) for r in (1, 2, 3, 4)]
            for j, snr in enumerate(SNRS):
                n0 = 10.0 ** (-snr / 10.0)
                # CSI measured now (ideal CSI-RS): per-rank layer SINRs
                if True:
                    G = [H @ W[q - 1] for q in (1, 2, 3, 4)]
                    meas = [batch_mmse_sinr(g, n0) for g in G]
                    rep = {"W": W, "G": G, "n0": n0, "sinr": meas, "cache": {}, "stores": {}}
                    report[(j, "ideal")] = rep
                    if t % CSI_PERIOD == 0:
                        pending[j].append((t + CSI_DELAY, rep))
                    if pending[j] and pending[j][0][0] <= t:
                        report[(j, "realistic")] = pending[j].pop(0)[1]
                # actual per-layer SINR this slot, DM-RS estimate, any precoder
                E = np.sqrt(n0 / CE_GAIN / 2) * (rng.standard_normal((N_RB, 4, 4))
                                                 + 1j * rng.standard_normal((N_RB, 4, 4)))
                cache = {}
                stores = {}

                def actual_cw(Wr, groups, k, rx, cancelled):
                    st = stores.setdefault(id(Wr), {})
                    h = H @ Wr
                    return cw_sinr(st, h, h + E[:, :, :Wr.shape[1]], n0, groups, k, rx,
                                   cancelled)

                def actual(Wr):
                    key = id(Wr)
                    if key not in cache:
                        h = H @ Wr
                        cache[key] = batch_mmse_sinr_est(h, h + E[:, :, :Wr.shape[1]], n0)
                    return cache[key]

                if t >= warm and t % 4 == 0:                           # Nokia-style
                    for r in (1, 2, 3, 4):
                        s_ = actual(W[r - 1])
                        for mi, max_cw in enumerate(MAPPINGS.values()):
                            genie[r - 1, mi, j] += sum(
                                ideal_amc(s_[:, g].ravel(), len(g), N_RE, N_RB)[3]
                                for g in layer_groups(r, max_cw))
                for c, rmode in CONFIGS:
                    rep = report[(j, c)]
                    if rep is None:
                        continue
                    for rx in receivers:
                        if rx != "mmse" and c not in RX_CSI:
                            continue
                        for m, max_cw in MAPPINGS.items():
                            k = (rx, c, rmode, m)
                            if rmode == "fixed4":
                                r = 4
                            else:
                                r = 1 + int(np.argmax([report_view(rep, q, max_cw, rx)[2]
                                                       for q in (1, 2, 3, 4)]))
                            groups, effs, _ = report_view(rep, r, max_cw, rx)
                            if rx == "mmse":
                                s_act = actual(rep["W"][r - 1])
                            off = olla[(j, k)]
                            done = []                    # SIC: layers decoded so far
                            for ci, g in enumerate(groups):
                                tab = TABLES[len(g)]
                                i = tab.select(effs[ci], off[ci])
                                if rx == "mmse":
                                    bl = tab.bler_at(s_act[:, g].ravel(), i)
                                else:
                                    s_cw, w_cw = actual_cw(rep["W"][r - 1], groups, ci, rx, done)
                                    bl = tab.bler_at(s_cw, i, w_cw)
                                ack = rng_rx[rx].random() > bl
                                if ack:
                                    done = done + list(g)
                                off[ci] += -OLLA_DN if ack else OLLA_UP
                                if t >= warm:
                                    bits[k][j] += tab.tbs[i] * ack
                                    ntx[k][j] += 1
                                    nack[k][j] += (not ack)
                            if t >= warm:
                                rank_sum[k][j] += r
    return bits, ntx, nack, rank_sum, genie


def run(model: str, speed: float, n_drops: int, n_slots: int, warm: int, jobs: int,
        receivers=RECEIVERS):
    seeds = np.arange(n_drops)
    chunks = [seeds[i::jobs] for i in range(jobs) if len(seeds[i::jobs])]
    with ProcessPoolExecutor(max_workers=jobs) as ex:
        parts = list(ex.map(_worker, [(model, speed, c, n_slots, warm, tuple(receivers))
                                      for c in chunks]))
    n_tx_slots = n_drops * (n_slots - warm)
    out = {}
    for rx in receivers:
        for c, r in CONFIGS:
            if rx != "mmse" and c not in RX_CSI:
                continue
            for m in MAPPINGS:
                k = (rx, c, r, m)
                b = sum(p[0][k] for p in parts)
                n = sum(p[1][k] for p in parts)
                out[("" if rx == "mmse" else f"{rx}:") + f"{c}/{r}/{m}"] = {
                    "se": (b / n_tx_slots / SLOT / BW).tolist(),
                    "bler": (sum(p[2][k] for p in parts) / np.maximum(n, 1)).tolist(),
                    "mean_rank": (sum(p[3][k] for p in parts) / n_tx_slots).tolist()}
    g = sum(p[4] for p in parts) / (n_drops * len(range(warm, n_slots, 4))) / SLOT / BW
    out["nokia_fixed_rank"] = {m: {str(r): g[r - 1, mi].tolist() for r in (1, 2, 3, 4)}
                               for mi, m in enumerate(MAPPINGS)}
    out["nokia_envelope"] = {m: g[:, mi].max(axis=0).tolist() for mi, m in enumerate(MAPPINGS)}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drops", type=int, default=40)
    ap.add_argument("--slots", type=int, default=400)
    ap.add_argument("--warmup", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--cases", default="CDL-A:3,CDL-C:3,CDL-C:30")
    ap.add_argument("--no-plot", action="store_true")
    ap.add_argument("--receivers", default="mmse", help="comma list of mmse, sic, rml")
    ap.add_argument("--out", default=os.path.join(OUT, "cw2lm_apple.json"))
    args = ap.parse_args()
    receivers = tuple(args.receivers.split(","))
    if "mmse" not in receivers:
        receivers = ("mmse",) + receivers
    res = {"snr_db": SNRS.tolist(), "n_drops": args.drops, "n_slots": args.slots,
           "warmup": args.warmup, "receivers": list(receivers), "cases": {}}
    for case in args.cases.split(","):
        model, speed = case.split(":")
        t0 = time.time()
        res["cases"][case] = run(model, float(speed), args.drops, args.slots,
                                 args.warmup, args.jobs, receivers)
        print(f"{case}: {time.time() - t0:.0f} s", flush=True)
        summarize(case, res["cases"][case], receivers)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    if not args.no_plot and receivers == ("mmse",):
        plot(res)
    return 0


def gains(se: dict, a: str, b: str = "alt1") -> np.ndarray:
    return 100 * (np.array(se[a]) / np.maximum(np.array(se[b]), 1e-9) - 1)


def summarize(case: str, r: dict, receivers=("mmse",)) -> None:
    snr = SNRS
    print(f"\n{case}: SE (bit/s/Hz) alt1 / alt2 / alt3, gain of alt3 over alt1")
    rows = [("Nokia-style envelope (ideal CSI, genie MCS)",
             {m: r["nokia_envelope"][m] for m in MAPPINGS})]
    for c, rk in CONFIGS:
        rows.append((f"{c} CSI, {rk}", {m: r[f"{c}/{rk}/{m}"]["se"] for m in MAPPINGS}))
    for label, se in rows:
        g3 = gains(se, "alt3")
        print(f"  {label}")
        for j in range(0, len(snr), 2):
            print(f"    {snr[j]:5.0f} dB: {se['alt1'][j]:6.2f} {se['alt2'][j]:6.2f} "
                  f"{se['alt3'][j]:6.2f}  ({g3[j]:+5.1f} %)")
    for rx in receivers:
        if rx == "mmse":
            continue
        for c, rk in [x for x in CONFIGS if x[0] in RX_CSI]:
            base = np.array(r[f"{c}/{rk}/alt1"]["se"])
            print(f"  {rx}, {c} CSI, {rk}: gain over Alt 1 with LMMSE (%) for Alt 1 / 2 / 3")
            for j in range(0, len(snr), 2):
                g = [100 * (r[f"{rx}:{c}/{rk}/{m}"]["se"][j] / max(base[j], 1e-9) - 1)
                     for m in MAPPINGS]
                print(f"    {snr[j]:5.0f} dB: " + " ".join(f"{x:+6.1f}" for x in g))


def plot(res: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    slot = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
    ink2 = "#52514e"
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.color": "#e4e3df", "font.size": 9.5,
                         "lines.linewidth": 2.0, "legend.frameon": False,
                         "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb"})
    cases = list(res["cases"])
    snr = np.array(res["snr_db"])
    fig, ax = plt.subplots(2, len(cases), figsize=(5.2 * len(cases), 8.4), squeeze=False)
    names = {"alt1": "Alt 1 (1 CW)", "alt2": "Alt 2 (<= 2 CW)", "alt3": "Alt 3 (CW per layer)"}
    for i, case in enumerate(cases):
        r = res["cases"][case]
        a = ax[0, i]
        for k, m in enumerate(MAPPINGS):
            a.plot(snr, r[f"realistic/fixed4/{m}"]["se"], color=slot[k], ls="--", lw=1.5,
                   label=f"{names[m]}, fixed rank 4")
            a.plot(snr, r[f"realistic/adaptive/{m}"]["se"], color=slot[k],
                   label=f"{names[m]}, CW-aware RI")
        a.set_title(f"{case.replace(':', ', ')} km/h, CSI 20 ms / 4 ms", loc="left",
                    fontweight="bold")
        a.set_xlabel("SNR (dB)"); a.set_ylabel("SE (bit/s/Hz)")
        a.legend(fontsize=7.5, loc="upper left")
        b = ax[1, i]
        series = [("Nokia-style: envelope of fixed-rank means, ideal CSI",
                   gains(r["nokia_envelope"], "alt3"), ink2, "-"),
                  ("ideal CSI, fixed rank 4", gains({m: r[f"ideal/fixed4/{m}"]["se"] for m in MAPPINGS}, "alt3"), slot[0], "--"),
                  ("ideal CSI, CW-aware RI", gains({m: r[f"ideal/adaptive/{m}"]["se"] for m in MAPPINGS}, "alt3"), slot[0], "-"),
                  ("CSI 20 ms / 4 ms, fixed rank 4", gains({m: r[f"realistic/fixed4/{m}"]["se"] for m in MAPPINGS}, "alt3"), slot[1], "--"),
                  ("CSI 20 ms / 4 ms, CW-aware RI (Apple)", gains({m: r[f"realistic/adaptive/{m}"]["se"] for m in MAPPINGS}, "alt3"), slot[1], "-")]
        for label, g, col, ls in series:
            b.plot(snr, g, color=col, ls=ls, label=label)
        b.axhline(0, color=ink2, lw=0.8)
        b.set_title("Gain of Alt 3 over Alt 1", loc="left", fontweight="bold")
        b.set_xlabel("SNR (dB)"); b.set_ylabel("gain (%)")
        b.set_ylim(-20, 80)
        b.legend(fontsize=7.5, loc="upper right")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cw2lm_apple.png"), dpi=130)
    print("saved results/cw2lm_apple.png")


if __name__ == "__main__":
    sys.exit(main())

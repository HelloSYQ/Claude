#!/usr/bin/env python3
"""Codeword-to-layer mapping with 2-TRP coherent joint transmission (CJT).

Do more codewords gain in multi-TRP?  Coherent JT from two TRPs adds the
second TRP's signal, often weaker (further away or behind blockage), to
the precoder.  The layers of the joint precoder then come from TRPs of
different strength, which can give exactly the layer imbalance that
per-layer link adaptation exploits.

Set-up: Apple R1-2608231's link (examples/cw2lm_apple.py) per TRP: 4 GHz,
51 PRB at 30 kHz, 32T (2,8,2;2,8), 4-Rx handset (elements 1, 3, 5, 7),
CDL, DS 100 ns, 3 km/h.  TRP 2 is an independent CDL link reaching the UE
from the opposite side, ``trp2_db`` weaker.  CJT precoder: wideband SVD of
the stacked 64-port channel, each TRP's part scaled to that TRP's full
power.  Single-TRP: TRP 1 alone.  The SNR is TRP 1's (per-TRP power).

Per scenario, Alt 1 / 2 / 3 (1 CW, <= 2 CW, one CW per layer) with
  * CSI 'ideal' (measured in the slot) or 'realistic' (20 ms period, 4 ms delay);
  * rank fixed to 4 or chosen per report knowing the codeword split;
  * per-codeword CQI and OLLA; optionally SIC / R-ML receivers.

    python examples/cw2lm_mtrp.py [--drops 16] [--jobs 4] [--receivers mmse,sic]   # ~1 h

Writes results/cw2lm_mtrp.json and results/cw2lm_mtrp.png.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cw2lm_apple as base                                          # noqa: E402

OUT = base.OUT
SCENARIOS = {
    "sTRP": None,
    "CJT, TRP 2 at 0 dB": {"kind": "cjt", "trp2_db": 0.0},
    "CJT, TRP 2 at -6 dB": {"kind": "cjt", "trp2_db": 6.0},
    "CJT, TRP 2 at -10 dB": {"kind": "cjt", "trp2_db": 10.0},
}
MAPPINGS = tuple(base.MAPPINGS)


def gain(se, a, b):
    return 100 * (np.array(se[a]) / np.maximum(np.array(se[b]), 1e-9) - 1)


def summarize(res: dict) -> None:
    snr = np.array(res["snr_db"])
    idx = [int(np.flatnonzero(snr == s)[0]) for s in (-2, 6, 14, 22, 30)]
    f = lambda xs: " / ".join(f"{x:+5.1f}" for x in xs)                # noqa: E731
    for model, scen in res["cases"].items():
        ref = scen["sTRP"]
        print(f"\n== {model}: SNR -2 / 6 / 14 / 22 / 30 dB")
        for name, r in scen.items():
            for cfg in ("realistic/adaptive", "realistic/fixed4", "ideal/adaptive"):
                se = {m: r[f"{cfg}/{m}"]["se"] for m in MAPPINGS}
                cjt = gain({"a": se["alt1"], "b": ref[f"{cfg}/alt1"]["se"]}, "a", "b")
                print(f"  {name:22s} {cfg:19s} Alt1 SE "
                      + " / ".join(f"{se['alt1'][i]:5.2f}" for i in idx)
                      + f" | Alt2 {f(gain(se, 'alt2', 'alt1')[idx])}"
                      + f" | Alt3 {f(gain(se, 'alt3', 'alt1')[idx])}"
                      + (f" | vs sTRP {f(cjt[idx])}" if name != "sTRP" else ""))
            for rx in res.get("receivers", ["mmse"]):
                if rx == "mmse":
                    continue
                cfg = "realistic/adaptive"
                b0 = np.array(r[f"{cfg}/alt1"]["se"])
                g = {m: 100 * (np.array(r[f"{rx}:{cfg}/{m}"]["se"]) / b0 - 1) for m in MAPPINGS}
                print(f"  {name:22s} {rx}, {cfg}: Alt1 {f(g['alt1'][idx])} | Alt2 "
                      f"{f(g['alt2'][idx])} | Alt3 {f(g['alt3'][idx])}  (vs Alt 1 LMMSE)")
            d = r["diag"]
            j = int(np.flatnonzero(snr == 22)[0])
            print(f"  {name:22s} rank-4 report @22 dB: layer SINR (dB) "
                  f"{np.round(d['rank4_layer_sinr_db'][j], 1).tolist()}, TRP-2 power share "
                  f"{np.round(d['rank4_trp2_power_share'][j], 2).tolist()}; mean rank (Alt 1) "
                  + " / ".join(f"{r['realistic/adaptive/alt1']['mean_rank'][i]:.2f}" for i in idx))


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
    snr = np.array(res["snr_db"])
    models = list(res["cases"])
    fig, ax = plt.subplots(len(models), 3, figsize=(16, 4.6 * len(models)), squeeze=False)
    for i, model in enumerate(models):
        scen = res["cases"][model]
        a, b, c = ax[i]
        for k, (name, r) in enumerate(scen.items()):
            a.plot(snr, r["realistic/adaptive/alt1"]["se"], color=slot[k], label=f"{name}, Alt 1")
            a.plot(snr, r["realistic/adaptive/alt3"]["se"], color=slot[k], ls="--", lw=1.5,
                   label=f"{name}, Alt 3")
            se = {m: r[f"realistic/adaptive/{m}"]["se"] for m in MAPPINGS}
            b.plot(snr, gain(se, "alt3", "alt1"), color=slot[k], label=f"{name}, CW-aware RI")
            se4 = {m: r[f"realistic/fixed4/{m}"]["se"] for m in MAPPINGS}
            b.plot(snr, gain(se4, "alt3", "alt1"), color=slot[k], ls=":", lw=1.5,
                   label=f"{name}, fixed rank 4")
        a.set(xlabel="SNR of TRP 1 (dB)", ylabel="SE (bit/s/Hz)")
        a.set_title(f"({'abc'[0]}{i + 1}) {model}, 3 km/h: Alt 1 vs Alt 3", loc="left",
                    fontweight="bold")
        a.legend(fontsize=7.5, loc="upper left")
        b.axhline(0, color=ink2, lw=0.8)
        b.set(xlabel="SNR of TRP 1 (dB)", ylabel="gain of Alt 3 over Alt 1 (%)", ylim=(-15, 60))
        b.set_title(f"(b{i + 1}) Gain of one CW per layer, CSI 20 ms / 4 ms", loc="left",
                    fontweight="bold")
        b.legend(fontsize=7.5, loc="upper right")
        j = int(np.flatnonzero(snr == 22)[0])
        width = 0.8 / len(scen)
        for k, (name, r) in enumerate(scen.items()):
            v = np.array(r["diag"]["rank4_layer_sinr_db"][j])
            sh = np.array(r["diag"]["rank4_trp2_power_share"][j])
            x = np.arange(1, 5) + (k - (len(scen) - 1) / 2) * width
            c.bar(x, v - v[0], width * 0.9, color=slot[k], label=name)
            if name != "sTRP":
                for xx, yy, s_ in zip(x, v - v[0], sh):
                    c.annotate(f"{100 * s_:.0f}%", (xx, yy), xytext=(0, -3),
                               textcoords="offset points", ha="center", va="top",
                               rotation=90, fontsize=7, color=ink2)
        c.set_xticks(range(1, 5), [f"layer {n}" for n in range(1, 5)])
        c.set(ylabel="rank-4 layer SINR relative to layer 1 (dB)")
        c.set_title(f"(c{i + 1}) Rank-4 layer SINR at 22 dB", loc="left", fontweight="bold")
        c.text(0.99, 0.02, "labels: share of the layer's power on TRP 2",
               transform=c.transAxes, ha="right", va="bottom", fontsize=7.5, color=ink2)
        c.legend(fontsize=7.5, loc="lower left")
        lo = min(min(np.array(r["diag"]["rank4_layer_sinr_db"][j])
                     - r["diag"]["rank4_layer_sinr_db"][j][0]) for r in scen.values())
        c.set_ylim(lo - 4, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cw2lm_mtrp.png"), dpi=130)
    print("saved results/cw2lm_mtrp.png")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drops", type=int, default=16)
    ap.add_argument("--slots", type=int, default=320)
    ap.add_argument("--warmup", type=int, default=80)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--models", default="CDL-A,CDL-C")
    ap.add_argument("--speed", type=float, default=3.0)
    ap.add_argument("--receivers", default="mmse")
    ap.add_argument("--trp2-delay-ns", type=float, default=0.0)
    ap.add_argument("--power", default="per_trp", choices=("per_trp", "total"))
    ap.add_argument("--out", default=os.path.join(OUT, "cw2lm_mtrp.json"))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    receivers = tuple(args.receivers.split(","))
    if "mmse" not in receivers:
        receivers = ("mmse",) + receivers
    res = {"snr_db": base.SNRS.tolist(), "n_drops": args.drops, "n_slots": args.slots,
           "warmup": args.warmup, "speed_kmh": args.speed, "receivers": list(receivers),
           "trp2_delay_ns": args.trp2_delay_ns, "power": args.power, "cases": {}}
    for model in args.models.split(","):
        res["cases"][model] = {}
        for name, scen in SCENARIOS.items():
            if scen is not None:
                scen = dict(scen, trp2_delay_ns=args.trp2_delay_ns, power=args.power)
            t0 = time.time()
            res["cases"][model][name] = base.run(model, args.speed, args.drops, args.slots,
                                                 args.warmup, args.jobs, receivers, scen)
            print(f"{model} / {name}: {time.time() - t0:.0f} s", flush=True)
    summarize(res)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(res, fh, indent=1)
    if not args.no_plot and os.path.abspath(args.out) == os.path.join(OUT, "cw2lm_mtrp.json"):
        plot(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

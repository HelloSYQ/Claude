"""Figures for examples/cw2lm_nokia.py (Nokia R1-2607066 comparison).

    python examples/cw2lm_nokia_plot.py      # re-plot results/cw2lm_nokia.json
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from cw2lm_nokia import (OUT, RANKS, TABLE4_SNR, NOKIA_TABLE4,   # noqa: E402
                         envelope, nokia_curves)

# reference categorical palette (light), fixed slot order
SLOT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
REF = "#52514e"                       # Nokia reference curves: neutral ink

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def _rank_panel(ax, snr, cur, title, mappings=("1cw", "cw_per_layer")):
    style = {"1cw": dict(ls="-", marker="+", ms=8, mew=1.6),
             "cw_per_layer": dict(ls="--", marker="o", ms=8, mfc="none", mew=1.4)}
    every = max(1, len(snr) // 14)
    for i, r in enumerate(RANKS):
        for m in mappings:
            if r == 1 and m != "1cw":
                continue
            keep = (np.asarray(snr) >= -15) & (np.asarray(snr) <= 40)
            ax.plot(np.asarray(snr)[keep], np.asarray(cur[m][str(r)])[keep], color=SLOT[i],
                    markevery=every, **style[m])
        y_end = np.interp(40.0, snr, cur["cw_per_layer" if r > 1 else "1cw"][str(r)])
        ax.annotate(f"rank {r}", (40.0, y_end),
                    xytext=(4, 0), textcoords="offset points", va="center",
                    fontsize=8.5, color=INK2)
    ax.set_title(title, loc="left")
    ax.set_xlabel("SNR (dB, Nokia scale)")
    ax.set_ylabel("spectral efficiency (bit/s/Hz)")
    ax.set_ylim(0, 20.5)
    ax.set_xlim(-15, 46)


def _style_legend(ax):
    from matplotlib.lines import Line2D
    h = [Line2D([], [], color=INK2, ls="-", marker="+", ms=8, mew=1.6, label="1 CW (NR)"),
         Line2D([], [], color=INK2, ls="--", marker="o", ms=7, mfc="none", label="1 CW per layer")]
    ax.legend(handles=h, loc="upper left", fontsize=8.5)


def plot(res: dict) -> None:
    cmp = res["comparison"]
    d = cmp["snr_offset_db"]
    snr = np.array(res["snr_db"]) + d
    nsnr, ncur, n28 = nokia_curves()
    V = res["variants"]

    # ---- Figure 1: Fig. 26 equivalents and the envelope gain ----------------
    fig, ax = plt.subplots(2, 2, figsize=(13, 9.6))
    _rank_panel(ax[0, 0], snr, V["nokia"]["se"],
                "(a) This simulator, Nokia set-up (MMSE, no SIC)")
    _rank_panel(ax[0, 1], nsnr, ncur, "(b) Nokia R1-2607066 Fig. 26 (digitized)")
    if "matched" in V:
        _rank_panel(ax[1, 0], snr, V["matched"]["se"],
                    "(c) This simulator, eigenmode spread fitted to Nokia")
    for a in (ax[0, 0], ax[0, 1], ax[1, 0]):
        _style_legend(a)

    g = ax[1, 1]
    s_grid = np.arange(-15, 40.5, 0.5)
    def gain(snr_axis, cur):
        e1 = np.interp(s_grid, snr_axis, envelope(cur, "1cw"))
        e4 = np.interp(s_grid, snr_axis, envelope(cur, "cw_per_layer"))
        return 100 * (e4 / e1 - 1)
    g.plot(s_grid, gain(nsnr, ncur), color=REF, lw=2.4, label="Nokia Fig. 26 (digitized)")
    g.plot(TABLE4_SNR, NOKIA_TABLE4["cw_per_layer"], "D", color=REF, ms=8,
           mfc=SURF, mew=1.6, label="Nokia Table 4")
    lab = {"nokia": "this simulator, Nokia set-up", "iso_ue": "isotropic 8-Rx UE",
           "ideal_ce": "ideal channel estimation", "matched": "eigenmode spread fitted to Nokia"}
    for i, name in enumerate(n for n in ("nokia", "iso_ue", "ideal_ce", "matched") if n in V):
        g.plot(s_grid, gain(snr, V[name]["se"]), color=SLOT[i], label=lab[name],
               ls="-" if name in ("nokia", "matched") else ":")
    g.axhline(0, color=INK2, lw=0.8)
    g.set_title("(d) Envelope SE gain of 1 CW per layer over 1 CW", loc="left")
    g.set_xlabel("SNR (dB, Nokia scale)")
    g.set_ylabel("gain with best fixed rank (%)")
    g.set_xlim(-15, 40)
    g.legend(loc="upper left", fontsize=8.5)
    fig.suptitle("Codeword-to-layer mapping, CDL-A, 64T8R, 96 PRB, 256QAM: "
                 f"1 CW vs 1 CW per layer (our SNR {d:+.1f} dB = Nokia SNR)",
                 fontweight="bold", x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cw2lm_nokia.png"), dpi=130)
    plt.close(fig)

    # ---- Figure 2: rank 4, 1 / 2 / 4 CW, and the eigenmode profile ---------
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.9))
    nok4 = {"1cw": ncur["1cw"]["4"], "2cw": n28["2cw"], "cw_per_layer": ncur["cw_per_layer"]["4"]}
    names = {"1cw": "1 CW (4 layers)", "2cw": "2 CW (2 + 2 layers)",
             "cw_per_layer": "4 CW (1 per layer)"}
    panels = [("nokia", "(a) Rank 4, Nokia set-up"), ("matched", "(b) Rank 4, spread fitted to Nokia")]
    for a, (name, title) in zip(ax[:2], panels):
        if name not in V:
            continue
        for i, m in enumerate(("1cw", "2cw", "cw_per_layer")):
            a.plot(snr, V[name]["se"][m]["4"], color=SLOT[i], label=f"{names[m]}, ours")
            a.plot(nsnr, nok4[m], color=SLOT[i], ls="--", lw=1.6, label=f"{names[m]}, Nokia")
        a.set_title(title, loc="left")
        a.set_xlabel("SNR (dB, Nokia scale)")
        a.set_ylabel("spectral efficiency (bit/s/Hz)")
        a.set_xlim(-15, 40); a.set_ylim(0, 20.5)
        a.legend(loc="upper left", fontsize=8)

    b = ax[2]
    profs = [("Nokia (inferred from Fig. 26)", None, REF)]
    profs += [(lab[n], n, SLOT[i]) for i, n in enumerate(("nokia", "iso_ue")) if n in V]
    k = np.arange(1, 5)
    width = 0.8 / len(profs)
    for j, (label, name, col) in enumerate(profs):
        if name is None:
            if "matched" not in V:
                continue
            e = np.array(V["nokia"]["eig_db_mean"])
            prof = e - np.array(V["matched"]["eig_atten_db"]) - e[0]
        else:
            e = np.array(V[name]["eig_db_mean"])
            prof = e - e[0]
        b.bar(k + (j - (len(profs) - 1) / 2) * width, prof, width * 0.9, color=col,
              label=label)
        for x, y in zip(k + (j - (len(profs) - 1) / 2) * width, prof):
            if y < -0.5:
                b.annotate(f"{y:.0f}", (x, y), xytext=(0, -11), textcoords="offset points",
                           ha="center", fontsize=8, color=INK2)
    b.set_xticks(k, [f"σ{i}²" for i in k])
    b.set_ylabel("mean eigenvalue relative to σ1² (dB)")
    b.set_title("(c) Layer imbalance: eigenmode power", loc="left")
    b.set_ylim(-50, 3)
    b.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "cw2lm_nokia_rank4.png"), dpi=130)
    plt.close(fig)
    print("saved results/cw2lm_nokia.png, results/cw2lm_nokia_rank4.png")


if __name__ == "__main__":
    with open(os.path.join(OUT, "cw2lm_nokia.json")) as f:
        plot(json.load(f))

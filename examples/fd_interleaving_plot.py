"""Figure for examples/fd_interleaving.py (FD interleaving, 32T4R CDL-A 30 ns).

    python examples/fd_interleaving_plot.py [32prb|256prb]   # re-plot the saved JSON
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fd_interleaving import OUT, PRG, SETUPS, Setup, cb_tables      # noqa: E402
from nrdlsim.re_mapping import code_block_prb_fraction              # noqa: E402

# reference categorical palette (light), fixed slot order; the reference
# (no interleaving) and the ideal curve are neutral ink
SLOT = ["#2a78d6", "#eb6834"]
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
STYLE = {
    "none": dict(color=INK2, ls="-", marker="o", mfc=SURF),
    "nr": dict(color=SLOT[0], ls="-", marker="D"),
    "blk": dict(color=SLOT[1], ls="-", marker="s"),
    "ideal": dict(color=INK, ls=":", marker=None),
}
# rank / MCS of panels (c), (d) and the CB maps, per set-up
REF = {"fd_interleaving": (4, 27), "fd_interleaving_256prb": (4, 20)}
MAX_CB_ROWS = 24                   # CBs drawn in the CB-to-PRB maps

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold",
    "legend.frameon": False, "lines.linewidth": 2.0,
})


def _mk(v, every):
    s = dict(STYLE[v])
    if s["marker"]:
        s.update(ms=5, markevery=every)
    return s


def plot(res: dict, path: str | None = None) -> str:
    setup = Setup(**res["setup"])
    REF_RANK, REF_MCS = REF.get(setup.name, (4, 27))
    SHORT = {"none": "no interleaving", "nr": "FD intlv 1: NR (L = 4)",
             "blk": f"FD intlv 2: 1 PRB, {setup.blk_rows} x {setup.blk_cols}",
             "ideal": "ideal interleaving"}
    snr = np.asarray(res["snr_db"])
    variants = list(res["variants"])
    every = 4
    fig = plt.figure(figsize=(18, 10.2))
    gs = GridSpec(2, 12, figure=fig, height_ratios=[1, 0.95], hspace=0.42, wspace=1.4)

    # (a) SE with rank adaptation
    ax = fig.add_subplot(gs[0, 0:4])
    for v in variants:
        ax.plot(snr, res["se"][f"{v}@x1"], label=SHORT[v], **_mk(v, every))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("spectral efficiency (b/s/Hz)")
    ax.set_title("(a) SE, ideal AMC + rank adaptation (curves overlap)")
    ax.legend(loc="upper left", fontsize=8.5)

    # (b) gain over no interleaving
    ax = fig.add_subplot(gs[0, 4:8])
    ax.axhline(0, color=INK2, lw=1)
    for v in variants[1:]:
        lo, hi = (np.asarray(x) for x in res["gain_ci95_pct"][f"{v}@x1"])
        st = _mk(v, every)
        ax.fill_between(snr, lo, hi, color=st["color"], alpha=0.12, lw=0)
        ax.plot(snr, res["gain_pct"][f"{v}@x1"], label=SHORT[v], **st)
        st4 = dict(st, ls="--" if v != "ideal" else (0, (1, 3)), mfc=SURF)
        ax.plot(snr, res["gain_pct"][f"{v}@x4"], **st4)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("SE gain over no interleaving (%)")
    ax.set_title("(b) SE gain (band: 95% CI over drops, waterfall x1)")
    # legends go where the gain is small: the right for a gain that peaks at low SNR
    side = "right" if snr[np.argmax(res["gain_pct"]["ideal@x1"])] < snr.mean() else "left"
    leg = ax.legend(loc=f"upper {side}", fontsize=8.5)
    ax.add_artist(leg)
    ax.legend(handles=[plt.Line2D([], [], color=INK2, ls="-", label="waterfall x1 (simulator)"),
                       plt.Line2D([], [], color=INK2, ls="--", label="waterfall x4 (steep)")],
              loc=f"upper {side}", bbox_to_anchor=(0.0 if side == "left" else 1.0, 0.74),
              fontsize=8.5)
    ax.set_ylim(-0.01, None)

    # (c) spread of per-CB effective SINR
    ax = fig.add_subplot(gs[0, 8:12])
    for v in variants:
        ax.plot(snr, np.asarray(res["cb_spread_db"][v][str(REF_RANK)])[:, REF_MCS],
                label=SHORT[v], **_mk(v, every))
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("max - min CB effective SINR (dB)")
    ax.set_title(f"(c) Spread of the {res['n_cb'][str(REF_RANK)][REF_MCS]} CB SINRs, "
                 f"rank {REF_RANK}, MCS {REF_MCS}")
    top = max(np.max(np.asarray(res["cb_spread_db"][v][str(REF_RANK)])[:, REF_MCS])
              for v in variants)
    ax.set_ylim(0, 1.45 * top)
    ax.legend(loc="upper left", fontsize=8.5)

    # (d) TB BLER at fixed rank / MCS
    ax = fig.add_subplot(gs[1, 0:3])
    for v in variants:
        for s, ls in (("x1", "-"), ("x4", "--")):
            b = np.asarray(res["bler"][f"{v}@{s}"][str(REF_RANK)])[REF_MCS]
            st = _mk(v, every)
            st.update(ls=ls if v != "ideal" else ":", mfc=SURF if s == "x4" else st.get("mfc"))
            ax.semilogy(snr, np.maximum(b, 1e-4),
                        label=f"{SHORT[v]}, {s}" if v in ("none", "ideal") or s == "x1" else None,
                        **st)
    ax.axhline(0.1, color=INK2, lw=0.8, ls="-.")
    s10 = res["snr10_db"]["none@x1"][str(REF_RANK)][REF_MCS]
    ax.set_xlim(np.floor(s10) - 11, np.ceil(s10) + 5)
    ax.set_ylim(1e-3, 1.05)
    ax.set_xlabel("SNR (dB)")
    ax.set_ylabel("TB BLER (mean over drops)")
    ax.set_title(f"(d) TB BLER, rank {REF_RANK}, MCS {REF_MCS}\n(solid: waterfall x1, dashed: x4)")
    ax.legend(loc="center left", fontsize=7.5)

    # (e)-(g) where each code block lands
    t = cb_tables(setup)[REF_RANK]
    a, n = t["first"][REF_MCS], t["n_cb"][REF_MCS]
    rows = min(n, MAX_CB_ROWS)
    spans = t["spans"][a:a + rows]
    maps = setup.mappings()
    n_rb = setup.n_rb
    cmap = LinearSegmentedColormap.from_list("seq", [SURF, "#9ec2ee", SLOT[0], "#163e70"])
    fr_all = {v: code_block_prb_fraction(spans, maps[v]) for v in ("none", "nr", "blk")}
    vmax = 1.6 * np.median(fr_all["none"][fr_all["none"] > 0])
    for k, v in enumerate(("none", "nr", "blk")):
        ax = fig.add_subplot(gs[1, 3 + 3 * k:6 + 3 * k])
        im = ax.imshow(fr_all[v], aspect="auto", cmap=cmap, vmin=0, vmax=vmax,
                       interpolation="nearest", origin="upper",
                       extent=(-0.5, n_rb - 0.5, rows - 0.5, -0.5))
        if n_rb // PRG <= 16:
            for g in range(1, n_rb // PRG):
                ax.axvline(g * PRG - 0.5, color=INK2, lw=0.6, alpha=0.6)
        ax.grid(False)
        ax.set_xticks(range(0, n_rb, n_rb // 8))
        ax.set_xlabel("PRB (vertical lines: PRG boundaries)" if n_rb // PRG <= 16 else "PRB")
        if k == 0:
            ax.set_ylabel("code block" + (f" (first {rows} of {n})" if rows < n else ""))
        ax.set_title(f"({'efg'[k]}) {SHORT[v]}:\nshare of each CB's REs per PRB")
    cb = fig.colorbar(im, ax=fig.axes[-3:], fraction=0.025, pad=0.01)
    cb.set_label("fraction of the CB's REs", color=INK2)
    cb.outline.set_visible(False)

    seg = (f", at least {setup.min_cb_per_symbol} CBs per OFDM symbol"
           if setup.min_cb_per_symbol else "")
    fig.suptitle(f"FD (VRB-to-PRB) interleaving: 32T4R, CDL-A, DS {setup.ds_ns:g} ns, "
                 f"{n_rb} PRB, PRG 4 (SVD per PRG), NR CW-to-layer mapping{seg}, "
                 f"{res['n_drops']} drops", color=INK, fontsize=12, fontweight="bold")
    path = path or os.path.join(OUT, setup.name + ".png")
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")
    return path


if __name__ == "__main__":
    name = SETUPS[sys.argv[1] if len(sys.argv) > 1 else "32prb"].name
    with open(os.path.join(OUT, name + ".json")) as fh:
        plot(json.load(fh))

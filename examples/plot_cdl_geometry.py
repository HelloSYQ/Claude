"""Illustrative figure explaining the CDL model's antenna geometry."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Rectangle
from matplotlib.lines import Line2D

from nrdlsim.channel_models import build_panel, _CDL, _RAY_OFFSETS

plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(15, 9.5))
gs = fig.add_gridspec(2, 3, height_ratios=[1.25, 1.0], hspace=0.28, wspace=0.24)

BLUE, GREEN, RED, ORANGE, GRAY = "#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#6b7280"

# ======================================================================
# TOP: end-to-end ray schematic  gNB UPA -> clusters -> UE
# ======================================================================
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 10); ax.set_ylim(-3.2, 3.2); ax.axis("off")
ax.set_title("CDL propagation geometry: gNB panel radiates rays at (AoD, ZoD) → "
             "clusters → UE panel receives at (AoA, ZoA)", fontweight="bold")

# gNB UPA panel (left) — draw a small vertical stack of dual-pol elements
gnb_x = 0.7
for zz in (-0.45, 0.0, 0.45):
    for pol, col in [(+1, BLUE), (-1, RED)]:
        ax.plot([gnb_x-0.12*pol, gnb_x+0.12*pol], [1.1+zz-0.12, 1.1+zz+0.12],
                color=col, lw=2, solid_capstyle="round")
ax.text(gnb_x, 2.15, "gNB\nUPA panel", ha="center", fontweight="bold", color="k")
ax.text(gnb_x, -0.1, "±45° dual-pol\nelements", ha="center", fontsize=8, color=GRAY)

# UE panel (right)
ue_x = 9.3
for zz in (-0.2, 0.2):
    for pol, col in [(+1, BLUE), (-1, RED)]:
        ax.plot([ue_x-0.1*pol, ue_x+0.1*pol], [zz-0.1, zz+0.1],
                color=col, lw=2, solid_capstyle="round")
ax.text(ue_x, 1.0, "UE\npanel", ha="center", fontweight="bold")
# UE velocity arrow
ax.add_patch(FancyArrowPatch((ue_x-0.2, -1.1), (ue_x-1.2, -1.7),
             arrowstyle="-|>", mutation_scale=16, color=ORANGE, lw=2))
ax.text(ue_x-1.4, -2.05, r"UE velocity $\vec{v}$" "\n"
        r"Doppler $\nu=f_d\,(\hat{r}_{rx}\!\cdot\!\hat{v})$",
        ha="center", fontsize=8.5, color=ORANGE)

# a few CDL-C clusters -> place scatterer blobs, connect gNB->blob->UE
clusters = np.array(_CDL["CDL-C"]["clusters"])
pick = [5, 0, 8, 15]                       # a spread of clusters
cols = [GREEN, BLUE, ORANGE, RED]
blob_x = np.array([4.2, 5.3, 5.0, 6.0])
blob_y = np.array([1.9, 0.4, -1.6, 2.4])
for (ci, bx, by, cc) in zip(pick, blob_x, blob_y, cols):
    aod, aoa = clusters[ci, 2], clusters[ci, 3]
    p = clusters[ci, 1]
    ax.add_patch(Circle((bx, by), 0.28, color=cc, alpha=0.25))
    ax.text(bx, by, f"cluster\n{ci+1}", ha="center", va="center", fontsize=7)
    # gNB -> cluster (departure)
    ax.add_patch(FancyArrowPatch((gnb_x+0.25, 1.1), (bx-0.2, by),
                 arrowstyle="-|>", mutation_scale=12, color=cc, lw=1.6, alpha=0.9))
    # cluster -> UE (arrival)
    ax.add_patch(FancyArrowPatch((bx+0.2, by), (ue_x-0.25, 0.0),
                 arrowstyle="-|>", mutation_scale=12, color=cc, lw=1.6, alpha=0.9,
                 linestyle=(0, (4, 2))))
# annotate AoD / AoA on the first cluster path
ax.text(2.3, 1.75, "AoD, ZoD\n(departure angle)", fontsize=8, color=GREEN, ha="center")
ax.text(7.8, 0.55, "AoA, ZoA\n(arrival angle)", fontsize=8, color=GREEN, ha="center")
ax.text(5.1, -2.7, "each cluster = 20 sub-rays; AoD and AoA are specified per "
        "cluster (independent), delays give frequency selectivity",
        ha="center", fontsize=8.5, style="italic", color=GRAY)

# ======================================================================
# BOTTOM-LEFT: UPA panel element geometry (real build_panel positions)
# ======================================================================
axp = fig.add_subplot(gs[1, 0])
pos, slant = build_panel(8, pol=2, layout=(2, 2), spacing_v=0.5, spacing_h=0.5)
uniq = np.unique(pos, axis=0)
for p in uniq:
    y, z = p[1], p[2]
    for s, col in [(+45, BLUE), (-45, RED)]:
        dy, dz = 0.14*np.cos(np.deg2rad(s)), 0.14*np.sin(np.deg2rad(s))
        axp.plot([y-dy, y+dy], [z-dz, z+dz], color=col, lw=2.4,
                 solid_capstyle="round")
    axp.plot(y, z, "o", color="k", ms=3)
axp.annotate("", xy=(0.5, -0.13), xytext=(0.0, -0.13),
             arrowprops=dict(arrowstyle="<->", color=GRAY))
axp.text(0.25, -0.22, r"$d_H=0.5\lambda$", ha="center", fontsize=8, color=GRAY)
axp.annotate("", xy=(-0.16, 0.5), xytext=(-0.16, 0.0),
             arrowprops=dict(arrowstyle="<->", color=GRAY))
axp.text(-0.34, 0.25, r"$d_V=0.5\lambda$", va="center", rotation=90,
         fontsize=8, color=GRAY)
axp.set_xlim(-0.45, 0.75); axp.set_ylim(-0.35, 0.75); axp.set_aspect("equal")
axp.set_xlabel("y  (wavelengths)"); axp.set_ylabel("z  (wavelengths)")
axp.set_title("(a) UPA panel geometry\n2×2 positions × ±45° pol = 8 ports",
              fontsize=9.5)
axp.legend([Line2D([0],[0],color=BLUE,lw=2), Line2D([0],[0],color=RED,lw=2)],
           ["+45° element", "−45° element"], fontsize=7.5, loc="upper right")

# ======================================================================
# BOTTOM-MID: plane-wave steering / location phase
# ======================================================================
axa = fig.add_subplot(gs[1, 1])
axa.set_xlim(-1.4, 1.6); axa.set_ylim(-1.3, 1.5); axa.set_aspect("equal")
axa.axis("off")
axa.set_title("(b) Array steering (location phase)", fontsize=9.5)
# two elements along y
e0, e1 = np.array([-0.6, -0.5]), np.array([0.6, -0.5])
axa.plot(*e0, "o", color="k", ms=7); axa.plot(*e1, "o", color="k", ms=7)
axa.text(0.0, -0.75, "two antenna elements", ha="center", fontsize=8, color=GRAY)
# incoming plane wave from direction r_hat (azimuth phi from broadside)
ang = np.deg2rad(35)
d = np.array([np.sin(ang), np.cos(ang)])       # propagation direction
for off in (-0.5, 0.1, 0.7):
    base = np.array([off*np.cos(ang)-0.2, off*np.sin(ang)+0.6])
    perp = np.array([np.cos(ang), -np.sin(ang)])
    axa.plot([base[0]-0.55*perp[0], base[0]+0.55*perp[0]],
             [base[1]-0.55*perp[1], base[1]+0.55*perp[1]], color=BLUE, lw=1, alpha=0.6)
axa.add_patch(FancyArrowPatch((0.55, 1.15), (0.15, 0.35), arrowstyle="-|>",
              mutation_scale=14, color=BLUE, lw=2))
axa.text(0.75, 1.2, r"$\hat{r}$", color=BLUE, fontsize=12)
# extra path length between elements
axa.plot([e0[0], e0[0]+ (e1-e0)@d*d[0]], [e0[1], e0[1]+ (e1-e0)@d*d[1]],
         color=RED, lw=2)
axa.text(0.05, -0.15, "Δpath", color=RED, fontsize=8.5)
axa.text(0.1, -1.15,
         r"phase$_u = 2\pi\,(\vec{p}_u\!\cdot\!\hat{r})$", ha="center",
         fontsize=10, color="k")

# ======================================================================
# BOTTOM-RIGHT: intra-cluster ray fan (angle spread)
# ======================================================================
axs = fig.add_subplot(gs[1, 2], projection="polar")
axs.set_title("(c) Intra-cluster rays\nmean ± offsets×spread (Tbl 7.5-3)",
              fontsize=9.5, pad=14)
c_asd = _CDL["CDL-C"]["spread"][0]             # C_ASD for CDL-C
mean = 20.0                                    # illustrative cluster mean AoD
rays = mean + c_asd * _RAY_OFFSETS
for r in rays:
    axs.plot([np.deg2rad(r), np.deg2rad(r)], [0, 1], color=GREEN, lw=1.2, alpha=0.6)
axs.plot([np.deg2rad(mean), np.deg2rad(mean)], [0, 1.12], color=RED, lw=2.6)
axs.set_thetamin(mean-3*c_asd-4); axs.set_thetamax(mean+3*c_asd+4)
axs.set_rticks([]); axs.set_rmax(1.15)
axs.set_thetagrids([mean-2*c_asd, mean, mean+2*c_asd])
axs.tick_params(axis="x", labelsize=8, colors=GRAY)
axs.text(np.deg2rad(mean+3.4*c_asd), 0.72, f"cluster mean\n(C_ASD={c_asd}°)",
         ha="left", va="center", fontsize=8, color=RED)
axs.text(np.deg2rad(mean-3.2*c_asd), 0.72, "20 rays",
         ha="right", va="center", fontsize=8, color=GREEN)

fig.suptitle(
    r"CDL channel — antenna geometry (TR 38.901 §7.3/7.5/7.7.1)   |   "
    r"$h_{u,s}=\sum_{\rm clusters}\sum_{\rm rays}\sqrt{P/M}\;"
    r"[\,\mathbf{F}_{rx}\mathbf{M}_{pol}\mathbf{F}_{tx}\,]\;"
    r"e^{j2\pi \vec{p}_{rx}\cdot\hat{r}_{rx}}\,"
    r"e^{j2\pi \vec{p}_{tx}\cdot\hat{r}_{tx}}\,e^{j2\pi\nu t}$",
    fontsize=11.5, fontweight="bold", y=0.995)

import os
os.makedirs("results", exist_ok=True)
fig.savefig("results/cdl_antenna_geometry.png", dpi=130, bbox_inches="tight")
print("saved results/cdl_antenna_geometry.png")

"""Illustrative figure explaining the CDL model's antenna geometry."""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Circle, Arc
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim.channel_models import build_panel, _CDL, _RAY_OFFSETS

plt.rcParams.update({"font.size": 10})
fig = plt.figure(figsize=(15, 9.8))
gs = fig.add_gridspec(2, 3, height_ratios=[1.28, 1.0], hspace=0.30, wspace=0.24)

BLUE, GREEN, RED, ORANGE, GRAY = "#2563eb", "#16a34a", "#dc2626", "#f59e0b", "#6b7280"


def mini_panel(ax, cx, cy, s=0.16, dp=0.42, label_bore=None):
    """Draw a compact 2x2 dual-pol (±45°) panel centred at (cx, cy)."""
    for gx in (-0.5, 0.5):
        for gy in (-0.5, 0.5):
            px, py = cx + gx*dp, cy + gy*dp
            for ang, col in [(45, BLUE), (-45, RED)]:
                dx, dy = s*np.cos(np.deg2rad(ang)), s*np.sin(np.deg2rad(ang))
                ax.plot([px-dx, px+dx], [py-dy, py+dy], color=col, lw=1.8,
                        solid_capstyle="round")


# ======================================================================
# TOP: end-to-end ray schematic  gNB UPA -> clusters -> UE
# ======================================================================
ax = fig.add_subplot(gs[0, :])
ax.set_xlim(0, 10); ax.set_ylim(-3.2, 3.2); ax.axis("off")
ax.set_title("CDL propagation geometry (azimuth-plane view; zenith ZoD/ZoA is the "
             "out-of-plane elevation angle)", fontweight="bold", fontsize=11)

gnb = np.array([0.85, 1.0]); ue = np.array([9.25, 0.0])
mini_panel(ax, *gnb, dp=0.5)
ax.text(gnb[0], 2.2, "gNB\nUPA panel", ha="center", fontweight="bold")
ax.text(gnb[0], -0.35, "±45° dual-pol", ha="center", fontsize=8, color=GRAY)
mini_panel(ax, *ue, dp=0.32, s=0.13)
ax.text(ue[0], 1.0, "UE\npanel", ha="center", fontweight="bold")

# boresight reference lines (dashed)
ax.plot([gnb[0]+0.3, gnb[0]+2.3], [gnb[1], gnb[1]], ls=(0,(5,4)), color=GRAY, lw=1)
ax.text(gnb[0]+2.35, gnb[1], "boresight", fontsize=7.5, color=GRAY, va="center")
ax.plot([ue[0]-0.3, ue[0]-2.3], [ue[1], ue[1]], ls=(0,(5,4)), color=GRAY, lw=1)

# clusters -> connect gNB->blob->UE (solid = departure, dashed = arrival)
clusters = np.array(_CDL["CDL-C"]["clusters"])
pick = [5, 0, 8, 15]
cols = [GREEN, BLUE, ORANGE, RED]
blob = np.array([[4.2, 1.9], [5.3, 0.4], [5.0, -1.6], [6.0, 2.4]])
for (ci, b, cc) in zip(pick, blob, cols):
    ax.add_patch(Circle(b, 0.30, color=cc, alpha=0.22))
    ax.text(b[0], b[1], f"cluster\n{ci+1}", ha="center", va="center", fontsize=7)
    ax.add_patch(FancyArrowPatch(gnb+(0.35, 0.15), b-(0.22, 0.0),
                 arrowstyle="-|>", mutation_scale=12, color=cc, lw=1.6, alpha=0.9))
    ax.add_patch(FancyArrowPatch(b+(0.22, 0.0), ue-(0.30, 0.0),
                 arrowstyle="-|>", mutation_scale=12, color=cc, lw=1.5, alpha=0.9,
                 linestyle=(0, (4, 2))))

# angle arc for AoD at gNB (to green cluster 6) and AoA at UE
v_dep = blob[0] - gnb; a_dep = np.degrees(np.arctan2(v_dep[1], v_dep[0]))
ax.add_patch(Arc(gnb, 2.4, 2.4, angle=0, theta1=0, theta2=a_dep, color=GREEN, lw=1.6))
ax.text(gnb[0]+1.35, gnb[1]+0.42, "AoD  φ_tx", color=GREEN, fontsize=9, fontweight="bold")
v_arr = blob[0] - ue; a_arr = np.degrees(np.arctan2(v_arr[1], v_arr[0]))
# Arc draws CCW from theta1 to theta2: go from the arrival ray up to boresight(180)
ax.add_patch(Arc(ue, 1.7, 1.7, angle=0, theta1=a_arr, theta2=180, color=GREEN, lw=1.6))
ax.text(ue[0]-1.55, ue[1]+0.95, "AoA  φ_rx", color=GREEN, fontsize=9, fontweight="bold")

# UE velocity arrow + Doppler
ax.add_patch(FancyArrowPatch(ue+(-0.2, -0.9), ue+(-1.4, -1.55),
             arrowstyle="-|>", mutation_scale=16, color=ORANGE, lw=2))
ax.text(ue[0]-1.6, ue[1]-2.05, r"UE velocity $\vec{v}$" "\n"
        r"Doppler $\nu=f_d\,(\hat{r}_{rx}\!\cdot\!\hat{v})$",
        ha="center", fontsize=8.5, color=ORANGE)
ax.text(5.0, -2.75, "solid = departure ray (AoD/ZoD),  dashed = arrival ray (AoA/ZoA);  "
        "each cluster = 20 sub-rays;  AoD and AoA are independent per cluster",
        ha="center", fontsize=8.5, style="italic", color=GRAY)

# ======================================================================
# (a) UPA panel element geometry (real build_panel positions)
# ======================================================================
axp = fig.add_subplot(gs[1, 0])
pos, slant = build_panel(8, pol=2, layout=(2, 2), spacing_v=0.5, spacing_h=0.5)
for p in np.unique(pos, axis=0):
    y, z = p[1], p[2]
    for s, col in [(45, BLUE), (-45, RED)]:
        dy, dz = 0.14*np.cos(np.deg2rad(s)), 0.14*np.sin(np.deg2rad(s))
        axp.plot([y-dy, y+dy], [z-dz, z+dz], color=col, lw=2.4, solid_capstyle="round")
    axp.plot(y, z, "o", color="k", ms=3)
axp.annotate("", xy=(0.5, -0.13), xytext=(0.0, -0.13),
             arrowprops=dict(arrowstyle="<->", color=GRAY))
axp.text(0.25, -0.23, r"$d_H=0.5\lambda$", ha="center", fontsize=8, color=GRAY)
axp.annotate("", xy=(-0.16, 0.5), xytext=(-0.16, 0.0),
             arrowprops=dict(arrowstyle="<->", color=GRAY))
axp.text(-0.35, 0.25, r"$d_V=0.5\lambda$", va="center", rotation=90, fontsize=8, color=GRAY)
axp.set_xlim(-0.5, 0.75); axp.set_ylim(-0.35, 0.75); axp.set_aspect("equal")
axp.set_xlabel("y  (wavelengths)"); axp.set_ylabel("z  (wavelengths)")
axp.set_title("(a) UPA panel geometry\n2×2 positions × ±45° pol = 8 ports", fontsize=9.5)
axp.legend([Line2D([0],[0],color=BLUE,lw=2), Line2D([0],[0],color=RED,lw=2)],
           ["+45° element", "−45° element"], fontsize=7.5, loc="upper right")

# ======================================================================
# (b) plane-wave steering / location phase  (clean ULA phased-array diagram)
# ======================================================================
axa = fig.add_subplot(gs[1, 1])
axa.set_xlim(-1.7, 1.7); axa.set_ylim(-1.5, 1.7); axa.set_aspect("equal"); axa.axis("off")
axa.set_title("(b) Array steering: a plane wave reaches\nelements with a path difference",
              fontsize=9.5)
th = np.deg2rad(33)                       # angle from broadside
k = np.array([np.sin(th), -np.cos(th)])   # propagation direction (into array)
perp = np.array([np.cos(th), np.sin(th)]) # along the wavefront
e0, e1 = np.array([-0.7, -0.7]), np.array([0.7, -0.7])   # two elements, spacing d
axa.plot([e0[0], e1[0]], [e0[1], e1[1]], color="k", lw=1.2, alpha=0.4)
axa.plot(*e0, "o", color="k", ms=8); axa.plot(*e1, "o", color="k", ms=8)
axa.text(0.0, -0.95, r"spacing $d$", ha="center", fontsize=8.5, color=GRAY)
# incoming parallel wavefronts (perpendicular to k), passing through e0
for off in (-0.3, 0.35, 1.0, 1.65):
    c = e0 + off*(-k)
    axa.plot([c[0]-0.7*perp[0], c[0]+0.7*perp[0]],
             [c[1]-0.7*perp[1], c[1]+0.7*perp[1]], color=BLUE, lw=1, alpha=0.5)
axa.add_patch(FancyArrowPatch((0.75, 1.45), (0.75-1.0*k[0], 1.45-1.0*k[1]),
              arrowstyle="-|>", mutation_scale=15, color=BLUE, lw=2))
axa.text(0.95, 1.4, r"$\hat{r}$", color=BLUE, fontsize=13)
# extra path Δ = d·sinθ : from e1 back to the wavefront through e0
foot = e1 - (np.dot(e1 - e0, -k))*(-k)
axa.plot([e1[0], foot[0]], [e1[1], foot[1]], color=RED, lw=2.6)
axa.text(0.35, -0.35, r"$\Delta=d\sin\theta$", color=RED, fontsize=9)
# broadside (normal) + angle arc
axa.plot([e0[0], e0[0]], [e0[1], e0[1]+0.9], ls=(0,(4,3)), color=GRAY, lw=1)
axa.add_patch(Arc(e0, 1.0, 1.0, angle=0, theta1=90-33, theta2=90, color=GRAY, lw=1.4))
axa.text(e0[0]+0.34, e0[1]+0.62, r"$\theta$", color=GRAY, fontsize=11)
axa.text(0.0, -1.32, r"phase$_u = 2\pi\,(\vec{p}_u\!\cdot\!\hat{r})$  →  "
         r"$\Delta\phi = 2\pi\frac{d}{\lambda}\sin\theta$", ha="center", fontsize=9.5)

# ======================================================================
# (c) intra-cluster ray fan (angle spread)
# ======================================================================
axs = fig.add_subplot(gs[1, 2], projection="polar")
axs.set_title("(c) Intra-cluster rays: mean ±\noffsets×spread (Table 7.5-3)", fontsize=9.5, pad=16)
c_asd = _CDL["CDL-C"]["spread"][0]
mean = 20.0
for r in mean + c_asd*_RAY_OFFSETS:
    axs.plot([np.deg2rad(r)]*2, [0, 1], color=GREEN, lw=1.2, alpha=0.6)
axs.plot([np.deg2rad(mean)]*2, [0, 1.12], color=RED, lw=2.6)
axs.set_thetamin(mean-3*c_asd-4); axs.set_thetamax(mean+3*c_asd+4)
axs.set_rticks([]); axs.set_rmax(1.15)
axs.set_thetagrids([mean-2*c_asd, mean, mean+2*c_asd])
axs.tick_params(axis="x", labelsize=8, colors=GRAY)
axs.text(np.deg2rad(mean+3.4*c_asd), 0.72, f"cluster mean\n(C_ASD={c_asd}°)",
         ha="left", va="center", fontsize=8, color=RED)
axs.text(np.deg2rad(mean-3.2*c_asd), 0.72, "20 rays", ha="right", va="center",
         fontsize=8, color=GREEN)

fig.suptitle(
    r"CDL channel — antenna geometry (TR 38.901 §7.3/7.5/7.7.1)   |   "
    r"$h_{u,s}=\sum_{\rm clusters}\sum_{\rm rays}\sqrt{P/M}\;"
    r"[\,\mathbf{F}_{rx}\mathbf{M}_{pol}\mathbf{F}_{tx}\,]\;"
    r"e^{j2\pi \vec{p}_{rx}\cdot\hat{r}_{rx}}\,"
    r"e^{j2\pi \vec{p}_{tx}\cdot\hat{r}_{tx}}\,e^{j2\pi\nu t}$",
    fontsize=11.5, fontweight="bold", y=0.998)

os.makedirs("results", exist_ok=True)
fig.savefig("results/cdl_antenna_geometry.png", dpi=130, bbox_inches="tight")
print("saved results/cdl_antenna_geometry.png")

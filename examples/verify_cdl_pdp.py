#!/usr/bin/env python3
"""Power-delay-profile (PDP) verification of the CDL channel generator.

Checks the channel produced by ``CDLChannel.frequency_response`` against the
TR 38.901 Tables 7.7.1-1..5 definition for CDL-A..E (see
``nrdlsim/channel_validation.py`` for the method):

  1. per-cluster powers recovered from H(f) vs the table (+ LS residual)
  2. estimation error shrinks as 1/sqrt(drops) -> no systematic bias
  3. frequency correlation E[H(f)H*(f+df)] vs the PDP's Fourier transform
  4. realised RMS delay spread vs the configured DS (30 / 100 / 300 ns)
  5. Ricean K-factor of CDL-D/E recovered from the generated channel
  6. with the 38.901 directional element, the PDP seen through the antenna
     is the table PDP weighted by each cluster's mean element gain

Writes results/cdl_pdp_verification.png.
"""
import os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim.channel_models import CDLChannel, _CDL
from nrdlsim.channel_validation import (table_pdp, merge_equal_delays,
                                        rms_delay_spread, los_k_factor_db,
                                        estimate_cluster_pdp, estimate_k_factor_db,
                                        frequency_correlation)

MODELS = ["CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E"]
DS = 100.0
N_DROPS = 1000
# reference palette (dataviz skill): blue = TR 38.901 reference, orange = generated
REF, GEN, THIRD = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def db(x):
    return 10 * np.log10(x)


def pattern_expected_pdp(model, **kw):
    """Table PDP weighted by each cluster's mean ray power gain (deterministic
    for a directional tx element and an omni rx)."""
    ch = CDLChannel(model, DS, 100.0, n_tx=2, n_rx=2,
                    rng=np.random.default_rng(0), **kw)
    tau = np.array([c[0] for c in _CDL[model]["clusters"]])
    w = ch.powers * np.mean(ch.ray_gain ** 2, axis=1)
    if ch.has_los:
        tau, w = np.r_[0.0, tau], np.r_[ch.p_los * ch.los_gain ** 2, w]
    t, p = merge_equal_delays(tau, w)
    return t * DS, p / p.sum()


def style(ax):
    ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def main():
    fig = plt.figure(figsize=(21, 12.5))
    gs = fig.add_gridspec(3, 5, hspace=0.55, wspace=0.28, top=0.89, bottom=0.05)
    print(f"CDL PDP verification  (DS = {DS:.0f} ns, {N_DROPS} drops x 4 times x 4 ports)\n")
    print(f"{'model':6} {'bins':>4} {'mean dB err':>11} {'rms dB err':>10} {'max dB err':>10} "
          f"{'LS resid':>9} {'P/port':>7} {'DS table':>8} {'DS est':>7} {'R(df) err':>9}")

    # --- checks 1 & 3 per model --------------------------------------------
    for j, m in enumerate(MODELS):
        r = estimate_cluster_pdp(m, DS, n_drops=N_DROPS)
        err = db(r["p_est"] / r["p_ref"])
        lag, Re, Rt = frequency_correlation(m, DS, n_drops=300)
        print(f"{m:6} {err.size:4d} {err.mean():+11.3f} {np.sqrt(np.mean(err**2)):10.3f} "
              f"{np.abs(err).max():10.3f} {r['worst_residual']:9.1e} {r['total_power']:7.3f} "
              f"{rms_delay_spread(r['tau_ns'], r['p_ref']):8.1f} "
              f"{rms_delay_spread(r['tau_ns'], r['p_est']):7.1f} {np.abs(Re - Rt).max():9.3f}")

        ax = fig.add_subplot(gs[0, j]); style(ax)
        floor = db(r["p_ref"]).min() - 6
        ax.vlines(r["tau_ns"], floor, db(r["p_ref"]), color=REF, lw=2)
        ax.plot(r["tau_ns"], db(r["p_ref"]), "o", ms=4, color=REF)
        ax.plot(r["tau_ns"], db(r["p_est"]), "D", ms=8, mfc="none", mew=2, color=GEN)
        ax.set_ylim(floor, 3)
        ax.set_title(f"{m} PDP (DS {DS:.0f} ns)\nerror: max {np.abs(err).max():.2f} dB, "
                     f"rms {np.sqrt(np.mean(err**2)):.2f} dB", color=INK, fontsize=10.5)
        ax.set_xlabel("delay (ns)", color=INK2, fontsize=9)
        if j == 0:
            ax.set_ylabel("cluster power (dB, normalised)", color=INK2, fontsize=9)

        ax = fig.add_subplot(gs[1, j]); style(ax)
        ax.plot(lag / 1e6, np.abs(Rt), color=REF, lw=2)
        ax.plot(lag[::3] / 1e6, np.abs(Re[::3]), "D", ms=7, mfc="none", mew=1.8,
                color=GEN)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"{m} frequency correlation |R(df)|\nmax |error| "
                     f"{np.abs(Re - Rt).max():.3f}", color=INK, fontsize=10.5)
        ax.set_xlabel("frequency separation df (MHz)", color=INK2, fontsize=9)
        if j == 0:
            ax.set_ylabel("|R(df)| / R(0)", color=INK2, fontsize=9)

    # --- check 2: convergence (unbiasedness) --------------------------------
    ax = fig.add_subplot(gs[2, 0]); style(ax)
    ns = np.array([50, 100, 200, 400, 800, 1600])
    rms = []
    for n in ns:
        r = estimate_cluster_pdp("CDL-A", DS, n_drops=n, seed=10_000)
        rms.append(np.sqrt(np.mean(db(r["p_est"] / r["p_ref"]) ** 2)))
    rms = np.array(rms)
    ax.loglog(ns, rms[0] * np.sqrt(ns[0] / ns), "--", color=INK2, lw=1.2,
              label="1/sqrt(N) reference")
    ax.loglog(ns, rms, "D-", ms=8, color=GEN, lw=2, label="rms cluster-power error")
    ax.set_yticks([0.07, 0.1, 0.2, 0.3, 0.5], ["0.07", "0.1", "0.2", "0.3", "0.5"])
    ax.set_xticks(ns, [str(n) for n in ns]); ax.minorticks_off()
    ax.set_xlabel("channel drops N", color=INK2, fontsize=9)
    ax.set_ylabel("rms error (dB)", color=INK2, fontsize=9)
    ax.set_title("CDL-A: error falls as 1/sqrt(N)\n(no bias floor)", color=INK, fontsize=10.5)
    ax.legend(fontsize=8, frameon=False)
    print("\nconvergence (CDL-A): " + "  ".join(f"N={n}: {e:.3f} dB" for n, e in zip(ns, rms)))

    # --- check 4: realised DS vs configured DS ------------------------------
    ax = fig.add_subplot(gs[2, 1]); style(ax)
    print("\nrealised RMS delay spread (ns):")
    y = np.arange(len(MODELS))
    for k, (ds, col, mk) in enumerate([(30, REF, "o"), (100, GEN, "D"), (300, THIRD, "s")]):
        ratio = []
        for m in MODELS:
            r = estimate_cluster_pdp(m, ds, n_drops=400, seed=20_000)
            ratio.append(rms_delay_spread(r["tau_ns"], r["p_est"]) / ds)
        print(f"  DS {ds:3d}: " + "  ".join(f"{m} {q*ds:6.1f}" for m, q in zip(MODELS, ratio)))
        ax.plot(ratio, y + (k - 1) * 0.2, mk, ms=8, color=col, label=f"DS = {ds} ns")
    ax.axvline(1.0, color=INK2, lw=1)
    ax.set_yticks(y, MODELS); ax.invert_yaxis()
    ax.set_xlim(0.94, 1.06)
    ax.set_xlabel("realised DS / configured DS", color=INK2, fontsize=9)
    ax.set_title("RMS delay spread matches\nthe configured value", color=INK, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="lower right")

    # --- check 5: K-factor ---------------------------------------------------
    ax = fig.add_subplot(gs[2, 2]); style(ax)
    print("\nRicean K-factor (LOS ray vs co-delayed cluster):")
    xs = np.arange(2)
    k_tab = [los_k_factor_db(m) for m in ("CDL-D", "CDL-E")]
    k_est = [estimate_k_factor_db(m, n_drops=2000) for m in ("CDL-D", "CDL-E")]
    for m, a, b in zip(("CDL-D", "CDL-E"), k_tab, k_est):
        print(f"  {m}: table {a:.2f} dB, recovered {b:.2f} dB")
    w = 0.36
    b1 = ax.bar(xs - w / 2 - 0.01, k_tab, w, color=REF, label="TR 38.901 table")
    b2 = ax.bar(xs + w / 2 + 0.01, k_est, w, color=GEN, label="recovered")
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.3,
                    f"{b.get_height():.1f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(xs, ["CDL-D", "CDL-E"]); ax.set_ylim(0, 26)
    ax.set_ylabel("K-factor (dB)", color=INK2, fontsize=9)
    ax.set_title("LOS K-factor recovered from\nthe generated channel", color=INK, fontsize=11)
    ax.legend(fontsize=8, frameon=False, loc="upper left")

    # --- check 6: PDP through a directional element -------------------------
    ax = fig.add_subplot(gs[2, 3:]); style(ax)
    kw = dict(tx_pattern="38.901")
    t_exp, p_exp = pattern_expected_pdp("CDL-C", **kw)
    r = estimate_cluster_pdp("CDL-C", DS, n_drops=N_DROPS, **kw)
    t0, p0 = merge_equal_delays(*table_pdp("CDL-C"))
    err = db(r["p_est"] / p_exp)
    floor = min(db(p_exp).min(), db(p0).min()) - 6
    ax.plot(t0 * DS, db(p0), "_", ms=14, mew=2, color=INK2,
            label="omni table PDP (for comparison)")
    ax.vlines(t_exp, floor, db(p_exp), color=REF, lw=2,
              label="table PDP x mean element gain per cluster")
    ax.plot(r["tau_ns"], db(r["p_est"]), "D", ms=8, mfc="none", mew=2, color=GEN,
            label="recovered from generated H(f)")
    ax.set_ylim(floor, 3)
    ax.set_xlabel("delay (ns)", color=INK2, fontsize=9)
    ax.set_ylabel("cluster power (dB, normalised)", color=INK2, fontsize=9)
    ax.set_title("CDL-C through the 38.901 directional tx element (clusters behind "
                 f"the panel suppressed)\nerror vs gain-weighted PDP: max "
                 f"{np.abs(err).max():.2f} dB", color=INK, fontsize=10.5)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    print(f"\ndirectional element (CDL-C): max err vs gain-weighted PDP "
          f"{np.abs(err).max():.3f} dB, rms {np.sqrt(np.mean(err**2)):.3f} dB")

    fig.suptitle("CDL channel generator — power-delay-profile verification "
                 "against TR 38.901 Tables 7.7.1-1..5", fontsize=14, color=INK, y=0.975)
    from matplotlib.lines import Line2D
    fig.legend(handles=[
        Line2D([], [], color=REF, lw=2, marker="o", ms=4,
               label="TR 38.901 reference (table PDP, its Fourier transform)"),
        Line2D([], [], color=GEN, lw=0, marker="D", ms=8, mfc="none", mew=2,
               label="measured from the generated channel H(f)")],
        loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=False,
        fontsize=10)
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/cdl_pdp_verification.png", dpi=110, bbox_inches="tight",
                facecolor="#fcfcfb")
    print("\nsaved results/cdl_pdp_verification.png")


if __name__ == "__main__":
    main()

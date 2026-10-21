#!/usr/bin/env python3
"""Codeword-to-layer mapping studies on a 32-Tx CDL-C channel (DS 30 ns).

How many codewords should carry the layers?  Each study is a subcommand:

  study             4 layers, 1 vs 4 codewords, 32T4R at 3.5 GHz, with the
                    MCS each layer gets as its own codeword
  ut                the same at 6.7 GHz: dual-pol array UE vs handheld UT UE
                    (5 dBi / 90 deg elements, random orientation per drop)
  1024qam           4 layers, 1 vs 4 codewords, MCS table 2 (up to 256QAM)
                    vs table 4 (up to 1024QAM)
  8layer            32T8R UT at 6.7 GHz, 2 vs 4 codewords with per-slot rank
                    adaptation (--fixed-rank N forces rank N)
  rank-compare      8layer with adaptive rank and fixed rank 8 overlaid
  olla              32T4R at 3.5 GHz with rank adaptation, CSI delayed by 4
                    slots and OLLA at 10 % BLER: realized SE, BLER and rank
  olla-convergence  OLLA offset and running BLER over the slots at 10 dB

All studies use 24 RB at 30 kHz, ideal-CSI SVD precoding with equal power per
layer, an LMMSE receiver, MIESM and the LDPC waterfall, and a UE at 3 km/h.
Except in olla and olla-convergence, every slot is an independent drop and
each codeword gets the MCS with the most expected bits (genie link
adaptation).

    python examples/codeword_mapping.py study

writes results/codeword_mapping_<study>.png (codeword_mapping_8layer_fixed<N>.png
with --fixed-rank, olla_convergence.png for olla-convergence).
"""

import argparse
import os, sys
import zlib
from collections import deque
from functools import lru_cache
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim import mcs_tables, tbs as tbs_mod
from nrdlsim.channel_models import CDLChannel
from nrdlsim.layer_mapping import svd_precoder
from nrdlsim.receiver import batch_mmse_sinr
from nrdlsim.link_abstraction import (effective_sinr_miesm, bler_from_effective_sinr,
                                      ideal_amc)

N_RB, SCS = 24, 30e3
MU_BW = N_RB * 12 * SCS
SLOT_TIME = 0.5e-3
N_RE_PRB = tbs_mod.re_per_rb(13, 24, 0)
FREQS = (np.arange(N_RB) - N_RB / 2) * 12 * SCS
FC_LOW, FC_UPPER6 = 3.5e9, 6.7e9
TARGET, DELAY, WARMUP = 0.10, 4, 120         # OLLA studies
STEP_UP = 0.1
STEP_DN = STEP_UP * TARGET / (1 - TARGET)    # -> steady-state BLER = TARGET


def make_channel(seed, fc, n_rx=4, rx_layout=(1, 2), orient_rng=None):
    """32T CDL-C, DS 30 ns, 3 km/h.  With ``orient_rng`` the UE is a handheld
    UT: boresight at a uniform bearing and within +/-20 deg of horizontal; the
    TR 38.901 rotation turns the pattern, the polarization and the panel
    together."""
    kw = dict(n_tx=32, n_rx=n_rx, carrier_freq_hz=fc, tx_pol=2, rx_pol=2,
              tx_layout=(2, 8), rx_layout=rx_layout, rng=np.random.default_rng(seed))
    if orient_rng is not None:
        kw.update(rx_pattern="38.901",
                  rx_boresight_az_deg=orient_rng.uniform(0, 360),
                  rx_downtilt_deg=orient_rng.uniform(-20, 20),
                  rx_element_max_gain_dbi=5.0, rx_element_hpbw_deg=90.0)
    return CDLChannel("CDL-C", 30, (3.0 / 3.6) * fc / 3e8, **kw)


def genie_mcs(sinr_lin, n_layers, table=2):
    """(MCS, expected bits) of the MCS with the most expected TB bits."""
    idx, _, _, bits = ideal_amc(sinr_lin, n_layers, N_RE_PRB, N_RB, table)
    return idx, bits


@lru_cache(maxsize=None)
def _mcs_rows(n_layers):
    """(MCS, Qm, code rate, TBS) of every MCS of table 2 for n_layers."""
    rows = []
    for idx in range(mcs_tables.num_mcs(2)):
        info = mcs_tables.get_mcs(idx, 2)
        rows.append((idx, info.modulation_order, info.target_code_rate,
                     tbs_mod.compute_tbs(N_RE_PRB, N_RB, info.modulation_order,
                                         info.target_code_rate, n_layers)))
    return rows


def olla_mcs(sinr_lin, n_layers, olla_db):
    """(MCS, TBS) of the largest TB whose predicted BLER after the OLLA
    back-off meets TARGET; (0, 0) if none does."""
    eff = {}
    chosen, chosen_tb = 0, 0
    for idx, qm, rate, tb in _mcs_rows(n_layers):
        if qm not in eff:
            eff[qm] = effective_sinr_miesm(sinr_lin, qm)
        bler = bler_from_effective_sinr(eff[qm] - olla_db, qm, rate, tb)
        if bler <= TARGET and tb > chosen_tb:
            chosen, chosen_tb = idx, tb
    return chosen, chosen_tb


def se(bits, n_slots):
    return bits / n_slots / SLOT_TIME / MU_BW


# ---------------------------------------------------------------------------
# 4 layers: 1 vs 4 codewords (study, ut, 1024qam)
# ---------------------------------------------------------------------------
def run_4layer(snr_db, fc, n_slots, table=2, ut_orient_seed=None):
    """SE of 1 and 4 codewords over 4 SVD layers, and the mean MCS of each
    layer as its own codeword.  ``ut_orient_seed`` selects the UT UE."""
    nv = 10 ** (-snr_db / 10.0)
    orient_rng = (None if ut_orient_seed is None
                  else np.random.default_rng(ut_orient_seed + int(snr_db)))
    bits1 = bits4 = 0.0
    layer_mcs = np.zeros(4)
    for s in range(n_slots):
        H = make_channel(s, fc, orient_rng=orient_rng).frequency_response(FREQS, 0.0)
        sinr = batch_mmse_sinr(H @ (svd_precoder(H, 4) / np.sqrt(4)), nv)  # [n_rb, 4]
        bits1 += genie_mcs(sinr.reshape(-1), 4, table)[1]
        per_layer = [genie_mcs(sinr[:, k], 1, table) for k in range(4)]
        bits4 += sum(b for _, b in per_layer)
        layer_mcs += [i for i, _ in per_layer]
    return se(bits1, n_slots), se(bits4, n_slots), layer_mcs / n_slots


def fig_study(args):
    snrs = np.arange(-5, 26, 2.5)
    se1, se4, lmcs = [], [], []
    print(f"32T4R CDL-C DS=30ns, {N_RB} RB, rank 4, SVD, 3.5 GHz")
    print(f"{'SNR':>6} {'SE 1-CW':>9} {'SE 4-CW':>9} {'gain%':>7}  per-layer MCS (4-CW)")
    for snr in snrs:
        s1, s4, lm = run_4layer(float(snr), FC_LOW, 150)
        se1.append(s1); se4.append(s4); lmcs.append(lm)
        gain = (s4 / s1 - 1) * 100 if s1 > 0 else 0
        print(f"{snr:6.1f} {s1:9.3f} {s4:9.3f} {gain:6.1f}%  "
              f"[{lm[0]:4.1f} {lm[1]:4.1f} {lm[2]:4.1f} {lm[3]:4.1f}]")

    lmcs = np.array(lmcs)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4.6))
    a1.plot(snrs, se1, "o-", color="#2563eb", label="1 codeword (joint)")
    a1.plot(snrs, se4, "s-", color="#dc2626", label="4 codewords (per-layer)")
    a1.set_xlabel("SNR (dB)"); a1.set_ylabel("spectral efficiency (b/s/Hz)")
    a1.set_title("4-layer SE: 1-CW vs 4-CW  (32T4R, CDL-C, SVD)")
    a1.grid(True, alpha=0.3); a1.legend()
    for k in range(4):
        a2.plot(snrs, lmcs[:, k], marker=".", label=f"layer {k} (σ{k+1})")
    a2.set_xlabel("SNR (dB)"); a2.set_ylabel("mean selected MCS index (4-CW)")
    a2.set_title("Per-layer MCS: strong eigenmodes get higher MCS")
    a2.grid(True, alpha=0.3); a2.legend(fontsize=8)
    fig.tight_layout()
    return fig, "results/codeword_mapping_study.png"


def fig_ut(args):
    snrs = np.arange(-5, 26, 2.5)
    data = {"array": ([], []), "ut": ([], [])}
    for model in ("array", "ut"):
        print(f"\n=== UE model: {model} (32T4R CDL-C, upper 6 GHz) ===")
        print(f"{'SNR':>6} {'SE 1CW':>8} {'SE 4CW':>8} {'gain%':>7}")
        for snr in snrs:
            s1, s4, _ = run_4layer(float(snr), FC_UPPER6, 150,
                                   ut_orient_seed=777 if model == "ut" else None)
            data[model][0].append(s1); data[model][1].append(s4)
            g = (s4 / s1 - 1) * 100 if s1 > 0 else 0
            print(f"{snr:6.1f} {s1:8.3f} {s4:8.3f} {g:6.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6), sharey=True)
    for a, model, title in [(ax[0], "array", "Array UE (omni dual-pol UPA)"),
                            (ax[1], "ut", "UT UE (directional, random orient.)")]:
        a.plot(snrs, data[model][0], "o-", color="#2563eb", label="1 codeword")
        a.plot(snrs, data[model][1], "s-", color="#dc2626", label="4 codewords")
        a.set_xlabel("SNR (dB)"); a.set_title(title); a.grid(alpha=0.3); a.legend()
    ax[0].set_ylabel("spectral efficiency (b/s/Hz)")
    fig.suptitle("1-CW vs 4-CW under array vs UT UE model (32T4R CDL-C, 6.7 GHz)",
                 fontweight="bold")
    fig.tight_layout()
    return fig, "results/codeword_mapping_ut.png"


def fig_1024qam(args):
    snrs = np.arange(-5, 41, 2.5)
    res = {2: ([], []), 4: ([], [])}
    for table, name in [(2, "256QAM"), (4, "+1024QAM")]:
        print(f"\n=== MCS table {table} ({name}) ===")
        print(f"{'SNR':>6} {'SE 1CW':>8} {'SE 4CW':>8} {'gain%':>7}")
        for snr in snrs:
            s1, s4, _ = run_4layer(float(snr), FC_LOW, 120, table=table)
            res[table][0].append(s1); res[table][1].append(s4)
            g = (s4 / s1 - 1) * 100 if s1 > 0 else 0
            print(f"{snr:6.1f} {s1:8.3f} {s4:8.3f} {g:6.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
    ax[0].plot(snrs, res[2][0], "o-", color="#93c5fd", label="1-CW, 256QAM")
    ax[0].plot(snrs, res[2][1], "s-", color="#fca5a5", label="4-CW, 256QAM")
    ax[0].plot(snrs, res[4][0], "o-", color="#2563eb", label="1-CW, +1024QAM")
    ax[0].plot(snrs, res[4][1], "s-", color="#dc2626", label="4-CW, +1024QAM")
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("spectral efficiency (b/s/Hz)")
    ax[0].set_title("4-layer SE: 256QAM vs +1024QAM (32T4R CDL-C)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    for table, c, lab in [(2, "#6b7280", "256QAM"), (4, "#16a34a", "+1024QAM")]:
        g = [(a / b - 1) * 100 for a, b in zip(res[table][1], res[table][0])]
        ax[1].plot(snrs, g, "^-", color=c, label=lab)
    ax[1].axhline(0, ls="--", color="gray")
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("4-CW gain over 1-CW (%)")
    ax[1].set_title("4-CW advantage: does 1024QAM lift the high-SNR ceiling?")
    ax[1].grid(alpha=0.3); ax[1].legend()
    fig.tight_layout()
    return fig, "results/codeword_mapping_1024qam.png"


# ---------------------------------------------------------------------------
# up to 8 layers: 2 vs 4 codewords (8layer, rank-compare)
# ---------------------------------------------------------------------------
def scheme_bits(sinr, n_cw):
    """Expected bits of a rank-r transmission in up to n_cw codewords.  The
    layers are eigenmode-ordered and grouped contiguously, which minimises
    the SINR spread inside each codeword (NR's rank 5-8 split for 2)."""
    r = sinr.shape[1]
    groups = np.array_split(np.arange(r), min(n_cw, r))
    return sum(genie_mcs(sinr[:, g].reshape(-1), len(g))[1] for g in groups)


def run_8layer(snr_db, n_slots=120, fixed_rank=None):
    """32T8R UT: SE and mean rank of 2 and 4 codewords.  The rank is chosen
    per slot for the most expected bits of each scheme, or fixed."""
    nv = 10 ** (-snr_db / 10.0)
    orient_rng = np.random.default_rng(555 + int(snr_db))
    ranks = [fixed_rank] if fixed_rank else list(range(1, 9))
    bits = {2: 0.0, 4: 0.0}
    rank_sum = {2: 0, 4: 0}
    for s in range(n_slots):
        H = make_channel(s, FC_UPPER6, 8, (2, 2), orient_rng).frequency_response(FREQS, 0.0)
        V = svd_precoder(H, 8)
        per_rank = {r: batch_mmse_sinr(H @ (V[:, :, :r] / np.sqrt(r)), nv)
                    for r in ranks}
        for n_cw in (2, 4):
            best_r, best = ranks[0], -1.0
            for r, sinr in per_rank.items():
                b = scheme_bits(sinr, n_cw)
                if b > best:
                    best, best_r = b, r
            bits[n_cw] += best
            rank_sum[n_cw] += best_r
    return ({c: se(bits[c], n_slots) for c in (2, 4)},
            {c: rank_sum[c] / n_slots for c in (2, 4)})


def fig_8layer(args):
    fr = args.fixed_rank
    tag = f"fixed rank {fr}" if fr else "up to 8 layers (adaptive)"
    snrs = np.arange(-5, 31, 2.5)
    se2, se4, rk2, rk4 = [], [], [], []
    print(f"32T8R UT CDL-C DS=30ns, upper 6 GHz -- 2 CW vs 4 CW ({tag})")
    print(f"{'SNR':>6} {'SE 2CW':>8} {'SE 4CW':>8} {'gain%':>7} "
          f"{'rank2':>6} {'rank4':>6}")
    for snr in snrs:
        s, rk = run_8layer(float(snr), fixed_rank=fr)
        se2.append(s[2]); se4.append(s[4]); rk2.append(rk[2]); rk4.append(rk[4])
        g = (s[4] / s[2] - 1) * 100 if s[2] > 0 else 0
        print(f"{snr:6.1f} {s[2]:8.3f} {s[4]:8.3f} {g:6.1f} "
              f"{rk[2]:6.1f} {rk[4]:6.1f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))
    ax[0].plot(snrs, se2, "o-", color="#2563eb", label="2 codewords (NR rank5-8)")
    ax[0].plot(snrs, se4, "s-", color="#dc2626", label="4 codewords")
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("spectral efficiency (b/s/Hz)")
    ax[0].set_title(f"2 CW vs 4 CW, {tag} (32T8R UT CDL-C)")
    ax[0].grid(alpha=0.3); ax[0].legend()
    gain = [(a / b - 1) * 100 for a, b in zip(se4, se2)]
    ax[1].plot(snrs, gain, "^-", color="#16a34a")
    ax[1].axhline(0, ls="--", color="gray")
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("4-CW gain over 2-CW (%)")
    ax[1].set_title("4-CW advantage vs SNR"); ax[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig, f"results/codeword_mapping_8layer{f'_fixed{fr}' if fr else ''}.png"


def fig_rank_compare(args):
    snrs = np.arange(-5, 31, 2.5)
    A = {2: [], 4: []}     # adaptive
    F = {2: [], 4: []}     # fixed rank 8
    rank_adapt = []
    print(f"{'SNR':>6} | {'adaptSE 2CW':>11} {'4CW':>7} rank | "
          f"{'fix8SE 2CW':>10} {'4CW':>7}")
    for snr in snrs:
        se_a, rk_a = run_8layer(float(snr), fixed_rank=None)
        se_f, _ = run_8layer(float(snr), fixed_rank=8)
        for c in (2, 4):
            A[c].append(se_a[c]); F[c].append(se_f[c])
        rank_adapt.append(rk_a[4])
        print(f"{snr:6.1f} | {se_a[2]:11.2f} {se_a[4]:7.2f} {rk_a[4]:4.1f} | "
              f"{se_f[2]:10.2f} {se_f[4]:7.2f}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 4.7))
    ax[0].plot(snrs, A[2], "o-", color="#2563eb", label="2-CW, adaptive rank")
    ax[0].plot(snrs, A[4], "s-", color="#dc2626", label="4-CW, adaptive rank")
    ax[0].plot(snrs, F[2], "o--", color="#93c5fd", label="2-CW, fixed rank 8")
    ax[0].plot(snrs, F[4], "s--", color="#fca5a5", label="4-CW, fixed rank 8")
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("spectral efficiency (b/s/Hz)")
    ax[0].set_title("Adaptive rank vs fixed rank 8 (32T8R UT CDL-C)")
    ax[0].grid(alpha=0.3); ax[0].legend(fontsize=8)
    ax[1].plot(snrs, rank_adapt, "o-", color="#16a34a", label="adaptive rank (RI)")
    ax[1].axhline(8, ls="--", color="gray", label="fixed rank 8")
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("layers transmitted")
    ax[1].set_title("Rank used: adaptation ramps 3 -> 8 with SNR")
    ax[1].set_ylim(0, 8.6); ax[1].grid(alpha=0.3); ax[1].legend()
    fig.tight_layout()
    return fig, "results/codeword_mapping_rank_compare.png"


# ---------------------------------------------------------------------------
# rank adaptation + delayed CSI + OLLA (olla, olla-convergence)
# ---------------------------------------------------------------------------
def run_olla(snr_db, scheme, seed, n_drops=14, slots_per_drop=40, warmup=WARMUP,
             history=False):
    """32T4R with rank adaptation, CSI delayed by DELAY slots and OLLA per
    codeword ("1cw": one codeword, "4cw": one per layer).

    The channel evolves within a drop, so the delayed precoder is slightly
    aged; drops add geometry diversity, and OLLA persists across them.  Block
    errors are drawn from the BLER on the current channel with the delayed
    precoder (first transmissions, no HARQ).  Returns SE, BLER and mean rank
    over the slots after ``warmup``; with ``history``, the mean OLLA offset
    and the fraction of failed codewords in every slot (None while the CSI
    pipeline fills).
    """
    nv = 10 ** (-snr_db / 10.0)
    rng = np.random.default_rng(seed)
    olla = np.zeros(4)                       # per layer position
    bits = 0.0; blocks = 0; errs = 0; ranks = []
    off_hist, err_hist = [], []
    gslot = 0
    for drop in range(n_drops):
        ch = make_channel(drop * 1009, FC_LOW)
        fb = deque()
        for s in range(slots_per_drop):
            H = ch.frequency_response(FREQS, s * SLOT_TIME)
            V = svd_precoder(H, 4)
            per_rank = {}
            for r in range(1, 5):
                W = V[:, :, :r] / np.sqrt(r)
                per_rank[r] = (W, batch_mmse_sinr(H @ W, nv))
            # RI: the rank with the most predicted bits, before OLLA
            best_r, best = 1, -1.0
            for r, (_, sinr) in per_rank.items():
                pred = (olla_mcs(sinr.reshape(-1), r, 0.0)[1] if scheme == "1cw"
                        else sum(olla_mcs(sinr[:, k], 1, 0.0)[1] for k in range(r)))
                if pred > best:
                    best, best_r = pred, r
            fb.append((best_r, *per_rank[best_r]))
            if len(fb) <= DELAY:
                if history:
                    off_hist.append(olla[:best_r].mean()); err_hist.append(None)
                gslot += 1
                continue
            rank, W, sinr_rep = fb.popleft()
            sinr_act = batch_mmse_sinr(H @ W, nv)              # current channel
            measure = gslot >= warmup
            if scheme == "1cw":
                cws = [(sinr_rep.reshape(-1), sinr_act.reshape(-1), rank, 0)]
            else:
                cws = [(sinr_rep[:, k], sinr_act[:, k], 1, k) for k in range(rank)]
            n_err = 0
            for s_rep, s_act, nl, k in cws:
                idx, tb = olla_mcs(s_rep, nl, olla[k])
                info = mcs_tables.get_mcs(idx, 2)
                bler = bler_from_effective_sinr(
                    effective_sinr_miesm(s_act, info.modulation_order),
                    info.modulation_order, info.target_code_rate, tb)
                ack = rng.random() > bler
                olla[k] += (-STEP_DN if ack else STEP_UP)
                n_err += 0 if ack else 1
                if measure:
                    blocks += 1; errs += 0 if ack else 1
                    if ack:
                        bits += tb
            if measure:
                ranks.append(rank)
            if history:
                off_hist.append(olla[:rank].mean()); err_hist.append(n_err / len(cws))
            gslot += 1
    if history:
        return np.array(off_hist), err_hist
    return (bits / (len(ranks) * SLOT_TIME) / MU_BW, errs / max(blocks, 1),
            float(np.mean(ranks)))


def fig_olla(args):
    snrs = np.arange(-5, 26, 2.5)
    rows = {"1cw": [], "4cw": []}
    print(f"32T4R CDL-C DS=30ns rank-adaptive + OLLA (target BLER {TARGET}), "
          f"CSI delay {DELAY} slots")
    print(f"{'SNR':>6} | {'SE 1CW':>7} {'BLER':>5} {'rank':>4} | "
          f"{'SE 4CW':>7} {'BLER':>5} {'rank':>4} | {'gain%':>6}")
    for snr in snrs:
        r1, r4 = (run_olla(float(snr), sch, zlib.crc32(f"{sch}:{snr:.1f}".encode()))
                  for sch in ("1cw", "4cw"))
        rows["1cw"].append(r1); rows["4cw"].append(r4)
        gain = (r4[0] / r1[0] - 1) * 100 if r1[0] > 0 else 0
        print(f"{snr:6.1f} | {r1[0]:7.2f} {r1[1]:5.2f} {r1[2]:4.1f} | "
              f"{r4[0]:7.2f} {r4[1]:5.2f} {r4[2]:4.1f} | {gain:6.1f}")

    a = {k: np.array(v) for k, v in rows.items()}
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.4))
    ax[0].plot(snrs, a["1cw"][:, 0], "o-", color="#2563eb", label="1 codeword")
    ax[0].plot(snrs, a["4cw"][:, 0], "s-", color="#dc2626", label="4 codewords")
    ax[0].set_xlabel("SNR (dB)"); ax[0].set_ylabel("spectral efficiency (b/s/Hz)")
    ax[0].set_title("SE — rank-adaptive + OLLA"); ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].plot(snrs, a["1cw"][:, 2], "o-", color="#2563eb", label="1 codeword")
    ax[1].plot(snrs, a["4cw"][:, 2], "s-", color="#dc2626", label="4 codewords")
    ax[1].set_xlabel("SNR (dB)"); ax[1].set_ylabel("mean selected rank (RI)")
    ax[1].set_title("Rank adaptation"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[2].plot(snrs, a["1cw"][:, 1], "o-", color="#2563eb", label="1 codeword")
    ax[2].plot(snrs, a["4cw"][:, 1], "s-", color="#dc2626", label="4 codewords")
    ax[2].axhline(TARGET, ls="--", color="gray", label=f"target {TARGET}")
    ax[2].set_xlabel("SNR (dB)"); ax[2].set_ylabel("achieved BLER")
    ax[2].set_title("OLLA holds BLER at target"); ax[2].legend(); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    return fig, "results/codeword_mapping_olla.png"


def fig_olla_convergence(args):
    """Check that OLLA settles (offset plateaus, BLER -> target) before the
    olla study's measurement window starts at WARMUP."""
    snr = 10.0
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    for scheme, c in [("1cw", "#2563eb"), ("4cw", "#dc2626")]:
        off, errs = run_olla(snr, scheme, seed=7, history=True)
        ax[0].plot(off, color=c, lw=1, label=scheme)
        e = np.array([x for x in errs if x is not None])
        post = e[np.arange(len(errs))[[x is not None for x in errs]] >= WARMUP]
        run_bler = np.cumsum(e) / np.arange(1, len(e) + 1)
        ax[1].plot(run_bler, color=c, label=scheme)
        print(f"{scheme}: final offset={off[-1]:+.2f} dB, "
              f"mean last 100={off[-100:].mean():+.2f} dB, "
              f"BLER after warmup={post.mean():.3f}")
    ax[0].axvline(WARMUP, ls="--", color="gray", label="warmup end")
    ax[0].set_xlabel("global slot"); ax[0].set_ylabel("mean OLLA offset (dB)")
    ax[0].set_title(f"OLLA offset trajectory (SNR={snr:.0f} dB)")
    ax[0].legend(); ax[0].grid(alpha=0.3)
    ax[1].axhline(TARGET, ls="--", color="gray", label="target 0.1")
    ax[1].set_xlabel("transmission #"); ax[1].set_ylabel("cumulative BLER")
    ax[1].set_title("Running BLER"); ax[1].legend(); ax[1].grid(alpha=0.3)
    ax[1].set_ylim(0, 0.3)
    fig.tight_layout()
    return fig, "results/olla_convergence.png"


STUDIES = {"study": fig_study, "ut": fig_ut, "1024qam": fig_1024qam,
           "8layer": fig_8layer, "rank-compare": fig_rank_compare,
           "olla": fig_olla, "olla-convergence": fig_olla_convergence}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study", choices=STUDIES)
    ap.add_argument("--fixed-rank", type=int, default=None,
                    help="8layer: force this rank every slot (e.g. 8); default adaptive")
    args = ap.parse_args()
    if args.fixed_rank and args.study != "8layer":
        ap.error("--fixed-rank applies to the 8layer study only")
    fig, out = STUDIES[args.study](args)
    os.makedirs("results", exist_ok=True)
    fig.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()

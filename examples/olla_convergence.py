#!/usr/bin/env python3
"""OLLA convergence diagnostic for the codeword-mapping study.

Logs the per-codeword OLLA offset trajectory and the running BLER over slots at
a single SNR, to check that the outer loop settles (offset plateaus, realized
BLER -> target) before the study's measurement window (warmup) begins.

Scenario matches examples/codeword_mapping_olla.py: 32T4R CDL-C DS=30 ns, 24 RB,
SVD, ~3 km/h, target BLER 0.1, CSI feedback delay 4 slots.
"""
import os, sys
import numpy as np
from collections import deque
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim import mcs_tables, tbs as tbs_mod
from nrdlsim.channel_models import CDLChannel
from nrdlsim.receiver import batch_mmse_sinr
from nrdlsim.link_abstraction import effective_sinr_miesm, bler_from_effective_sinr

N_RB, SCS = 24, 30e3
N_RE_PRB = tbs_mod.re_per_rb(13, 24, 0)
N_TX, N_RX, MAX_RANK = 32, 4, 4
FC = 3.5e9
FD = (3 / 3.6) * FC / 3e8
QMS = (2, 4, 6, 8)
TARGET, DELAY, WARMUP = 0.10, 4, 120
STEP_UP = 0.1
STEP_DN = STEP_UP * TARGET / (1 - TARGET)
freqs = (np.arange(N_RB) - N_RB / 2) * 12 * SCS


def best_mcs(sinr, nl, olla):
    eff = {q: effective_sinr_miesm(sinr, q) for q in QMS}
    ci, ctb = 0, 0
    for idx in range(mcs_tables.num_mcs(2)):
        info = mcs_tables.get_mcs(idx, 2)
        tb = tbs_mod.compute_tbs(N_RE_PRB, N_RB, info.modulation_order,
                                 info.target_code_rate, nl)
        bler = bler_from_effective_sinr(eff[info.modulation_order] - olla,
                                        info.modulation_order,
                                        info.target_code_rate, tb)
        if bler <= TARGET and tb > ctb:
            ci, ctb = idx, tb
    return ci, ctb


def run(scheme, snr_db, n_drops=14, spd=40):
    nv = 10 ** (-snr_db / 10)
    rng = np.random.default_rng(7)
    olla = np.zeros(MAX_RANK)
    off_hist, ack_hist = [], []
    for drop in range(n_drops):
        ch = CDLChannel("CDL-C", 30, FD, n_tx=N_TX, n_rx=N_RX, carrier_freq_hz=FC,
                        tx_pol=2, rx_pol=2, tx_layout=(2, 8), rx_layout=(1, 2),
                        rng=np.random.default_rng(drop * 1009))
        fb = deque()
        for s in range(spd):
            H = ch.frequency_response(freqs, s * 0.5e-3)
            _, _, vh = np.linalg.svd(H, full_matrices=False)
            V = np.conj(np.swapaxes(vh, -1, -2))
            per = {r: (V[:, :, :r] / np.sqrt(r),
                       batch_mmse_sinr(H @ (V[:, :, :r] / np.sqrt(r)), nv))
                   for r in range(1, 5)}
            br, bp = 1, -1
            for r, (W, sinr) in per.items():
                pred = (best_mcs(sinr.reshape(-1), r, 0)[1] if scheme == "1cw"
                        else sum(best_mcs(sinr[:, k], 1, 0)[1] for k in range(r)))
                if pred > bp:
                    bp, br = pred, r
            fb.append((br, *per[br]))
            if len(fb) <= DELAY:
                off_hist.append(olla[:br].mean()); ack_hist.append(None); continue
            rank, W, srep = fb.popleft()
            sact = batch_mmse_sinr(H @ W, nv)
            groups = ([(srep.reshape(-1), sact.reshape(-1), rank, 0)] if scheme == "1cw"
                      else [(srep[:, k], sact[:, k], 1, k) for k in range(rank)])
            acks = []
            for sr, sa, nl, k in groups:
                idx, tb = best_mcs(sr, nl, olla[k])
                info = mcs_tables.get_mcs(idx, 2)
                bl = bler_from_effective_sinr(
                    effective_sinr_miesm(sa, info.modulation_order),
                    info.modulation_order, info.target_code_rate, tb)
                ack = rng.random() > bl
                acks.append(ack)
                olla[k] += (-STEP_DN if ack else STEP_UP)
            off_hist.append(olla[:rank].mean())
            ack_hist.append(np.mean([0 if a else 1 for a in acks]))
    return np.array(off_hist), ack_hist


def main():
    snr = 10.0
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
    for scheme, c in [("1cw", "#2563eb"), ("4cw", "#dc2626")]:
        off, acks = run(scheme, snr)
        ax[0].plot(off, color=c, lw=1, label=scheme)
        e = np.array([a for a in acks if a is not None])
        post = e[np.arange(len(acks))[[a is not None for a in acks]] >= WARMUP]
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
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/olla_convergence.png", dpi=130)
    print("saved results/olla_convergence.png")


if __name__ == "__main__":
    main()

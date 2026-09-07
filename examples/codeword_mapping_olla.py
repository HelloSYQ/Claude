#!/usr/bin/env python3
"""Codeword mapping (1-CW vs per-layer-CW) WITH rank adaptation and OLLA.

Adds a realistic link-adaptation loop on top of the codeword-mapping study:

  * rank adaptation : each slot the UE reports the rank (1..4) that maximises
    predicted throughput for the scheme (RI).
  * CSI feedback delay : the reported rank/precoder/SINR are `DELAY` slots old.
  * outer-loop link adaptation (OLLA) : a per-codeword offset updated from
    HARQ ACK/NACK drives the *realized* BLER to the 10% target (step ratio
    step_down/step_up = target/(1-target)).
  * realized throughput : block errors are drawn from the actual SINR on the
    current channel using the delayed precoder (first transmission, no HARQ
    combining), so SE reflects operation at the target BLER.

Scenario: CDL-C DS=30 ns, 24 RB @ 30 kHz, 32T4R dual-pol UPA, SVD, ~3 km/h.
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
MU_BW = N_RB * 12 * SCS
SLOT_TIME = 0.5e-3
N_RE_PRB = tbs_mod.re_per_rb(13, 24, 0)
N_TX, N_RX, MAX_RANK = 32, 4, 4
FC = 3.5e9
FD = (3.0 / 3.6) * FC / 3e8
MCS_TABLE = 2
QMS = (2, 4, 6, 8)
TARGET = 0.10
DELAY = 4
STEP_UP = 0.1
STEP_DN = STEP_UP * TARGET / (1 - TARGET)   # -> steady-state BLER = TARGET


def make_channel(seed):
    return CDLChannel("CDL-C", 30, FD, n_tx=N_TX, n_rx=N_RX, carrier_freq_hz=FC,
                      tx_pol=2, rx_pol=2, tx_layout=(2, 8), rx_layout=(1, 2),
                      rng=np.random.default_rng(seed))


def best_mcs(sinr_set, n_layers, olla_db):
    """Highest-throughput MCS with predicted BLER <= TARGET after OLLA back-off."""
    eff = {qm: effective_sinr_miesm(sinr_set, qm) for qm in QMS}
    chosen, chosen_tb = 0, 0
    for idx in range(mcs_tables.num_mcs(MCS_TABLE)):
        info = mcs_tables.get_mcs(idx, MCS_TABLE)
        tb = tbs_mod.compute_tbs(N_RE_PRB, N_RB, info.modulation_order,
                                 info.target_code_rate, n_layers)
        bler = bler_from_effective_sinr(eff[info.modulation_order] - olla_db,
                                        info.modulation_order,
                                        info.target_code_rate, tb)
        if bler <= TARGET and tb > chosen_tb:
            chosen, chosen_tb = idx, tb
    return chosen, chosen_tb


def run_scheme(snr_db, scheme, n_drops=14, slots_per_drop=40, warmup=120, base=0):
    """Rank-adaptive + OLLA run.  The channel evolves continuously within a
    'drop' (so the delayed precoder is only slightly aged at low mobility);
    multiple drops give fading/geometry diversity.  OLLA persists across drops.
    """
    nv = 10 ** (-snr_db / 10.0)
    freqs = (np.arange(N_RB) - N_RB / 2) * 12 * SCS
    rng = np.random.default_rng(hash((scheme, round(snr_db, 1))) % 2**32)
    olla = np.zeros(MAX_RANK)               # per layer-position (persists)
    bits = 0.0; blocks = 0; errs = 0; ranks = []
    gslot = 0
    for drop in range(n_drops):
        ch = make_channel(base + drop * 1009)      # fresh geometry per drop
        fb = deque()                                # feedback buffer per drop
        for s in range(slots_per_drop):
            H = ch.frequency_response(freqs, s * SLOT_TIME)   # continuous
            _, _, vh = np.linalg.svd(H, full_matrices=False)
            V = np.conj(np.swapaxes(vh, -1, -2))              # [nrb,32,4]
            per_rank = {}
            for r in range(1, MAX_RANK + 1):
                Wr = V[:, :, :r] / np.sqrt(r)
                per_rank[r] = (Wr, batch_mmse_sinr(H @ Wr, nv))

            best_r, best_pred = 1, -1.0
            for r, (Wr, sinr) in per_rank.items():
                if scheme == "1cw":
                    _, tb = best_mcs(sinr.reshape(-1), r, 0.0); pred = tb
                else:
                    pred = sum(best_mcs(sinr[:, k], 1, 0.0)[1] for k in range(r))
                if pred > best_pred:
                    best_pred, best_r = pred, r
            fb.append((best_r, *per_rank[best_r]))
            if len(fb) <= DELAY:
                gslot += 1
                continue
            rank, W, sinr_rep = fb.popleft()
            sinr_act = batch_mmse_sinr(H @ W, nv)             # current channel

            measure = gslot >= warmup
            if scheme == "1cw":
                groups = [(sinr_rep.reshape(-1), sinr_act.reshape(-1), rank, 0)]
            else:
                groups = [(sinr_rep[:, k], sinr_act[:, k], 1, k) for k in range(rank)]
            for sinr_r, sinr_a, nl, k in groups:
                idx, tb = best_mcs(sinr_r, nl, olla[k])
                info = mcs_tables.get_mcs(idx, MCS_TABLE)
                bler_act = bler_from_effective_sinr(
                    effective_sinr_miesm(sinr_a, info.modulation_order),
                    info.modulation_order, info.target_code_rate, tb)
                ack = rng.random() > bler_act
                olla[k] += (-STEP_DN if ack else STEP_UP)
                if measure:
                    blocks += 1; errs += (0 if ack else 1)
                    if ack:
                        bits += tb
            if measure:
                ranks.append(rank)
            gslot += 1
    n_meas = len(ranks)
    dur = n_meas * SLOT_TIME
    return bits / dur / MU_BW, errs / max(blocks, 1), float(np.mean(ranks))


def main():
    snrs = np.arange(-5, 26, 2.5)
    rows = {"1cw": [], "4cw": []}
    print(f"32T4R CDL-C DS=30ns rank-adaptive + OLLA (target BLER {TARGET}), "
          f"CSI delay {DELAY} slots, f_d={FD:.1f} Hz")
    print(f"{'SNR':>6} | {'SE 1CW':>7} {'BLER':>5} {'rank':>4} | "
          f"{'SE 4CW':>7} {'BLER':>5} {'rank':>4} | {'gain%':>6}")
    for snr in snrs:
        r1 = run_scheme(float(snr), "1cw")
        r4 = run_scheme(float(snr), "4cw")
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
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/codeword_mapping_olla.png", dpi=130)
    print("saved results/codeword_mapping_olla.png")


if __name__ == "__main__":
    main()

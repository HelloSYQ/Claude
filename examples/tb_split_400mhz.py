#!/usr/bin/env python3
"""One TB over 400 MHz vs two TBs over 200 MHz each (same carrier).

  case 1 : one 400 MHz carrier (264 PRB), one TB across all PRBs.
  case 2 : the same carrier, TB1 on the lower 132 PRB and TB2 on the upper
           132 PRB, each with its own CQI, MCS, OLLA loop and HARQ process
           (a hypothetical frequency split, not an NR-standard mapping).

Assumptions
  * FR2, 28 GHz, 120 kHz SCS (0.125 ms slot); 264 PRB = 400 MHz.
  * CDL-A, DS 100 ns, 3 km/h; gNB 16T (dual-pol 2x4 panel), UE 2R
    (dual-pol), omni elements; rank up to 2.
  * Equal power per RE over the whole carrier (SNR per Rx antenna per RE).
  * Ideal channel estimation, CSI reported with a 4-slot delay; one wideband
    rank (RI) for the carrier; per-RB SVD precoding; MMSE receiver.
  * CQI/MCS from MCS/CQI table 2 at a 10% TB BLER target, OLLA per TB,
    TB-based HARQ (chase combining, up to 4 transmissions).
  * TB error model: TS 38.212 code-block segmentation; each code block gets
    the effective SINR (MIESM) of the REs it is mapped to (frequency-first);
    TB BLER = 1 - prod(1 - BLER_cb).  Run with the default code-block
    waterfall slope and a steep LDPC-like slope (11.8/dB: 10% -> 0.1% BLER in
    ~0.4 dB) to bound the code-block-count effect.
  * 13 PDSCH symbols, 24 DM-RS RE/PRB (132 data RE/PRB); PT-RS ignored.

SE is normalised by the full 400 MHz in both cases.
Writes results/tb_split_400mhz.png.
"""
import os, sys
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
from concurrent.futures import ProcessPoolExecutor
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim import mcs_tables, tbs as tbs_mod
from nrdlsim.channel_models import CDLChannel
from nrdlsim.receiver import batch_mmse_sinr
from nrdlsim.link_abstraction import (bicm_capacity, mi_to_eff_sinr_db, cb_segmentation,
                                      cb_re_weights, tb_bler_from_cb_sinrs,
                                      required_eff_sinr_db_tb)

FC, SCS, SLOT = 28e9, 120e3, 0.125e-3
N_RB, HALF = 264, 132
N_RE = tbs_mod.re_per_rb(13, 24, 0)
FD = (3.0 / 3.6) * FC / 3e8
FREQS = (np.arange(N_RB) - N_RB / 2) * 12 * SCS
DELAY, WARMUP, N_SLOTS, N_DROPS = 4, 40, 400, 16
TARGET, MAX_TX, OLLA_STEP = 0.1, 4, 0.5
MCS_TABLE = CQI_TABLE = 2
QMS = (2, 4, 6, 8)
RANKS = (1, 2)
SNRS = np.arange(-6, 31, 3)
DECOMP_SNRS = (3, 9, 15)     # decomposition runs (one TB-wide SINR per TB)
SLOPES = {"default CB slope": None, "steep LDPC-like CB slope": 11.8}
N_MCS = mcs_tables.num_mcs(MCS_TABLE)


class Tables:
    """TBS and target-BLER SINR thresholds for an allocation size and slope."""

    def __init__(self, n_rb, slope, per_cb=True):
        self.n_rb, self.slope, self.per_cb = n_rb, slope, per_cb
        self.tbs = {r: [tbs_mod.compute_tbs(N_RE, n_rb, *self._mcs(i), r) for i in range(N_MCS)]
                    for r in RANKS}
        self.mcs_req = {r: np.array([required_eff_sinr_db_tb(*self._mcs(i), self.tbs[r][i],
                                                             TARGET, slope)
                                     for i in range(N_MCS)]) for r in RANKS}
        self.cqi_req = {}
        for r in RANKS:
            req = []
            for c in range(1, 16):
                qm, rate, _ = mcs_tables.get_cqi(c, CQI_TABLE)
                tb = tbs_mod.compute_tbs(N_RE, n_rb, qm, rate, r)
                req.append(required_eff_sinr_db_tb(qm, rate, tb, TARGET, slope))
            self.cqi_req[r] = np.array(req)
        self._w = {}

    @staticmethod
    def _mcs(i):
        m = mcs_tables.get_mcs(i, MCS_TABLE)
        return m.modulation_order, m.target_code_rate

    def weights(self, c):
        if c not in self._w:
            self._w[c] = cb_re_weights(self.n_rb, c, N_RE)
        return self._w[c]

    def cqi(self, mi_rb, r):
        """Highest CQI whose wideband effective SINR meets the target."""
        eff = {q: float(mi_to_eff_sinr_db(mi_rb[q].mean(), q)) for q in QMS}
        for c in range(15, 0, -1):
            if eff[mcs_tables.get_cqi(c, CQI_TABLE)[0]] >= self.cqi_req[r][c - 1]:
                return c
        return 0

    def mcs(self, cqi, r, olla_db):
        if cqi == 0:
            return 0
        ok = np.nonzero(self.mcs_req[r] <= self.cqi_req[r][cqi - 1] - olla_db + 1e-9)[0]
        return int(ok[-1]) if ok.size else 0

    def bler(self, sinr, mcs, r):
        """TB BLER for per-(RB, layer) SINRs of this allocation."""
        qm, rate = self._mcs(mcs)
        tb = self.tbs[r][mcs]
        c, _ = cb_segmentation(tb, rate)
        mi_rb = bicm_capacity(10 * np.log10(np.maximum(sinr, 1e-9)), qm).mean(axis=1)
        if not self.per_cb:                  # decomposition: one TB-wide SINR
            return tb_bler_from_cb_sinrs(mi_to_eff_sinr_db(mi_rb.mean(), qm), qm, rate, tb,
                                         self.slope)
        eff_cb = mi_to_eff_sinr_db(self.weights(c) @ mi_rb, qm)
        return tb_bler_from_cb_sinrs(eff_cb, qm, rate, tb, self.slope)


class TBStream:
    """One TB stream: OLLA loop + single TB-based HARQ process."""

    def __init__(self, tables, rbs, rng):
        self.t, self.rbs, self.rng = tables, rbs, rng
        self.olla, self.harq = 0.0, None
        self.bits = self.first = self.first_err = 0
        self.mcs_sum = self.n = 0

    def step(self, H, rep, cqi, nv):
        if self.harq is None:
            r = rep["rank"]
            m = self.t.mcs(cqi, r, self.olla)
            self.harq = dict(r=r, mcs=m, tbs=self.t.tbs[r][m], acc=None, n=0)
        h = self.harq
        W = rep["V"][self.rbs, :, :h["r"]] / np.sqrt(h["r"])     # delayed precoder
        sinr = batch_mmse_sinr(H[self.rbs] @ W, nv)              # true channel
        h["acc"] = sinr if h["acc"] is None else h["acc"] + sinr  # chase combining
        ok = self.rng.random() > self.t.bler(h["acc"], h["mcs"], h["r"])
        if h["n"] == 0:
            self.first += 1; self.first_err += int(not ok)
            self.olla += -OLLA_STEP * TARGET / (1 - TARGET) if ok else OLLA_STEP
            self.olla = float(np.clip(self.olla, -10, 10))
        h["n"] += 1
        self.mcs_sum += h["mcs"]; self.n += 1
        if ok:
            self.bits += h["tbs"]; self.harq = None
        elif h["n"] >= MAX_TX:
            self.harq = None


def run(task):
    snr, drop, slope_name, per_cb = task
    slope = SLOPES[slope_name]
    nv = 10 ** (-snr / 10)
    full, half = Tables(N_RB, slope, per_cb), Tables(HALF, slope, per_cb)
    lo, hi = np.arange(HALF), np.arange(HALF, N_RB)
    ch = CDLChannel("CDL-A", 100.0, FD, n_tx=16, n_rx=2, carrier_freq_hz=FC,
                    tx_pol=2, rx_pol=2, tx_layout=(2, 4), rx_layout=(1, 1),
                    rng=np.random.default_rng(drop))
    c1 = TBStream(full, np.arange(N_RB), np.random.default_rng(10_000 + 97 * drop + snr))
    c2 = [TBStream(half, lo, np.random.default_rng(20_000 + 97 * drop + snr)),
          TBStream(half, hi, np.random.default_rng(30_000 + 97 * drop + snr))]
    queue, d_eff, rank_sum = [], [], 0
    for s in range(N_SLOTS):
        H = ch.frequency_response(FREQS, s * SLOT)
        # --- UE: CSI on the current channel (ideal estimate) -------------------
        _, _, vh = np.linalg.svd(H, full_matrices=False)
        V = np.conj(np.swapaxes(vh, -1, -2))
        mi = {}
        for r in RANKS:
            sinr = batch_mmse_sinr(H @ (V[:, :, :r] / np.sqrt(r)), nv)
            db = 10 * np.log10(np.maximum(sinr, 1e-9))
            mi[r] = {q: bicm_capacity(db, q).mean(axis=1) for q in QMS}   # per RB
        wb = {r: full.cqi(mi[r], r) for r in RANKS}
        rank = max(RANKS, key=lambda r: r * mcs_tables.get_cqi(wb[r], CQI_TABLE)[2])
        sub = lambda idx: {q: v[idx] for q, v in mi[rank].items()}
        rep = dict(rank=rank, V=V, cqi1=wb[rank],
                   cqi2=(half.cqi(sub(lo), rank), half.cqi(sub(hi), rank)))
        e_lo = mi_to_eff_sinr_db(mi[rank][6][lo].mean(), 6)
        e_hi = mi_to_eff_sinr_db(mi[rank][6][hi].mean(), 6)
        queue.append(rep)
        if len(queue) <= DELAY:
            continue
        rep = queue.pop(0)                                   # 4-slot-old report
        # --- gNB: both cases transmit on the same channel ----------------------
        c1.step(H, rep, rep["cqi1"], nv)
        for k in (0, 1):
            c2[k].step(H, rep, rep["cqi2"][k], nv)
        if s == DELAY + WARMUP:                              # end of warm-up
            for x in (c1, *c2):
                x.bits = x.first = x.first_err = x.mcs_sum = x.n = 0
            d_eff, rank_sum = [], 0
        d_eff.append(abs(e_lo - e_hi)); rank_sum += rank
    n_meas = N_SLOTS - DELAY - WARMUP - 1
    norm = n_meas * SLOT * N_RB * 12 * SCS
    return dict(snr=snr, drop=drop, slope=slope_name, per_cb=per_cb,
                se1=c1.bits / norm, se2=(c2[0].bits + c2[1].bits) / norm,
                bler1=c1.first_err / max(c1.first, 1),
                bler2=(c2[0].first_err + c2[1].first_err) / max(c2[0].first + c2[1].first, 1),
                mcs1=c1.mcs_sum / max(c1.n, 1),
                mcs2=(c2[0].mcs_sum + c2[1].mcs_sum) / max(c2[0].n + c2[1].n, 1),
                rank=rank_sum / max(len(d_eff), 1), d_eff=float(np.mean(d_eff)))


def main():
    tasks = [(int(s), d, k, True) for k in SLOPES for s in SNRS for d in range(N_DROPS)]
    tasks += [(s, d, k, False) for k in SLOPES for s in DECOMP_SNRS for d in range(N_DROPS)]
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        out = list(ex.map(run, tasks, chunksize=2))
    decomp = [o for o in out if not o["per_cb"]]
    out = [o for o in out if o["per_cb"]]
    res = {}
    for k in SLOPES:
        print(f"\n=== {k} ===")
        print(f"{'SNR':>4} | {'SE case1':>8} {'SE case2':>8} {'gain':>6} | {'BLER1':>5} "
              f"{'BLER2':>5} | {'MCS1':>5} {'MCS2':>5} | rank | |halves eff SINR diff|")
        for s in SNRS:
            rs = [o for o in out if o["slope"] == k and o["snr"] == s]
            m = {f: np.mean([o[f] for o in rs]) for f in rs[0]
                 if f not in ("snr", "drop", "slope", "per_cb")}
            res[(k, int(s))] = m
            print(f"{s:4d} | {m['se1']:8.3f} {m['se2']:8.3f} {100*(m['se2']/m['se1']-1):+5.1f}% | "
                  f"{m['bler1']:5.3f} {m['bler2']:5.3f} | {m['mcs1']:5.1f} {m['mcs2']:5.1f} | "
                  f"{m['rank']:4.2f} | {m['d_eff']:5.2f} dB")
    cap = max(res[(k, int(SNRS[-1]))]["se1"] for k in SLOPES)
    print(f"\nSNR advantage of case 2 at equal SE (SE below 90% of the {cap:.2f} b/s/Hz "
          "ceiling, where the curves are not flat):")
    for k in SLOPES:
        s1 = np.array([res[(k, int(s))]["se1"] for s in SNRS])
        s2 = np.array([res[(k, int(s))]["se2"] for s in SNRS])
        g = np.linspace(max(s1.min(), s2.min()), 0.9 * cap, 300)
        db = np.interp(g, s1, SNRS) - np.interp(g, s2, SNRS)
        print(f"  {k}: mean {db.mean():+.2f} dB, range {db.min():+.2f} .. {db.max():+.2f} dB")
    print(f"\nDecomposition ({N_DROPS} drops): gain with per-CB SINRs vs with one TB-wide SINR per TB")
    print(f"  {'slope':26} {'SNR':>4} | {'per-CB gain':>11} {'uniform gain':>12} | "
          f"{'weakest-CB part':>15}")
    for k in SLOPES:
        for s in DECOMP_SNRS:
            u = [o for o in decomp if o["slope"] == k and o["snr"] == s]
            g_u = 100 * (np.mean([o["se2"] for o in u]) / np.mean([o["se1"] for o in u]) - 1)
            g_p = 100 * (res[(k, s)]["se2"] / res[(k, s)]["se1"] - 1)
            print(f"  {k:26} {s:4d} | {g_p:+10.1f}% {g_u:+11.1f}% | {g_p - g_u:+13.1f} pts")
    plot(res)


def plot(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, ink2, grid = "#0b0b0b", "#52514e", "#e4e3df"
    c1, c2 = "#2a78d6", "#eb6834"                      # reference palette slots 1-2
    ls = dict(zip(SLOPES, ("-", "--")))
    fig, ax = plt.subplots(1, 2, figsize=(14, 5.2))
    for a in ax:
        a.grid(True, color=grid, lw=0.8); a.set_axisbelow(True)
        for s in ("top", "right"):
            a.spines[s].set_visible(False)
        a.tick_params(colors=ink2, labelsize=9); a.set_xlabel("SNR (dB)", color=ink2)
    for k in SLOPES:
        se1 = [res[(k, int(s))]["se1"] for s in SNRS]
        se2 = [res[(k, int(s))]["se2"] for s in SNRS]
        ax[0].plot(SNRS, se1, marker="o", color=c1, ls=ls[k], lw=2, ms=6,
                   mfc=c1 if ls[k] == "-" else "white", label=f"case 1: 1 TB x 400 MHz ({k})")
        ax[0].plot(SNRS, se2, marker="D", color=c2, ls=ls[k], lw=2, ms=6,
                   mfc=c2 if ls[k] == "-" else "white", label=f"case 2: 2 TB x 200 MHz ({k})")
        ax[1].plot(SNRS, [100 * (b / a - 1) for a, b in zip(se1, se2)], marker="D", color=c2,
                   ls=ls[k], lw=2, ms=6, mfc=c2 if ls[k] == "-" else "white", label=k)
    ax[1].axhline(0, color=ink2, lw=0.8)
    ax[0].set_ylabel("spectral efficiency over 400 MHz (b/s/Hz)", color=ink2)
    ax[0].set_title("SE vs SNR", color=ink); ax[0].legend(frameon=False, fontsize=8.5)
    ax[1].set_ylabel("case 2 SE gain over case 1 (%)", color=ink2)
    ax[1].set_title("Gain of two 200 MHz TBs over one 400 MHz TB", color=ink)
    ax[1].legend(frameon=False, fontsize=8.5)
    fig.suptitle("1 TB x 400 MHz vs 2 TB x 200 MHz — FR2 28 GHz, 120 kHz SCS, CDL-A 100 ns, "
                 "16T2R, CQI/OLLA per TB, TB-based HARQ", color=ink)
    fig.tight_layout()
    os.makedirs("results", exist_ok=True)
    fig.savefig("results/tb_split_400mhz.png", dpi=120, facecolor="#fcfcfb")
    print("saved results/tb_split_400mhz.png")


if __name__ == "__main__":
    main()

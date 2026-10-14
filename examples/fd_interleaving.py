#!/usr/bin/env python3
"""Frequency-domain (VRB-to-PRB) interleaving: 32T4R, CDL-A, DS 30 ns, 32 PRB.

The UE is allocated the whole 32-PRB bandwidth, so the VRB-to-PRB mapping
does not change which PRBs carry the transport block (TB).  It changes which
PRBs each LDPC code block (CB) lands on.  The CBs are mapped frequency first
over the VRBs.  Without interleaving, a CB that spans less than a few OFDM
symbols sits on one contiguous part of the band, and the TB fails when its
worst CB fails.  The mappings compared are:

  none    non-interleaved, PRB = VRB (reference)
  nr      FD interleaving 1: NR interleaved mapping (TS 38.211 7.3.1.6),
          bundle size L = 4 (L = 2 is not allowed with PRG size 4), R = 2
  blk4x8  FD interleaving 2: 1-PRB granularity, 4 x 8 block interleaver,
          VRBs written in rows and read in columns:
          VRB v -> PRB 4 (v mod 8) + floor(v / 8)
  ideal   every CB sees the slot-average mutual information, i.e. perfect
          per-CB frequency diversity (ideal interleaving)

Link model, per drop (independent CDL-A realisation) and SNR:

  * SVD precoding per PRG of 4 PRBs: eigenvectors of the PRG-summed channel
    covariance, ideal CSI; total power split over the layers;
  * LMMSE receiver, per-subcarrier per-layer SINR, ideal channel estimation;
  * one codeword (NR codeword-to-layer mapping up to rank 4).  TBS (TS 38.214
    5.1.3.2), LDPC base graph and CB segmentation (TS 38.212 7.2.2, 5.2.2),
    E_r per CB (5.4.2.1), layer mapping and frequency-first RE mapping in the
    VRB domain (TS 38.211 7.3.1.3, 7.3.1.5), then VRB-to-PRB mapping;
  * per-CB MIESM over the CB's REs and the simulator's LDPC waterfall at the
    CB's code rate K'/E_r and length K'.  TB BLER = 1 - prod(1 - BLER_CB);
  * ideal AMC with rank adaptation: the rank (1-4) and MCS (Table 2) with the
    most expected bits TBS (1 - BLER_TB), chosen separately for each mapping.

The waterfall slope matters here (a steep waterfall makes the worst CB
dominate), so every result is also given with a 4x steeper waterfall
(~0.2 dB per BLER decade at K' = 8448).

    python examples/fd_interleaving.py [--drops 1000] [--jobs 4] [--ds 30]

Writes results/fd_interleaving.json and results/fd_interleaving.png.  See
docs/fd-interleaving.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from nrdlsim import mcs_tables, tbs as tbs_mod                         # noqa: E402
from nrdlsim.channel_models import CDLChannel                           # noqa: E402
from nrdlsim.ldpc import segment_transport_block, rate_match_lengths    # noqa: E402
from nrdlsim.link_abstraction import (bicm_capacity, _bicm_table,       # noqa: E402
                                      _waterfall)
from nrdlsim.re_mapping import (vrb_to_prb_non_interleaved,             # noqa: E402
                                vrb_to_prb_nr_interleaved,
                                vrb_to_prb_block_interleaver,
                                code_block_spans, code_block_means)
from nrdlsim.receiver import batch_mmse_sinr                            # noqa: E402

OUT = os.path.join(ROOT, "results")

FC = 3.5e9
SCS = 30e3
N_RB = 32
N_SC = 12 * N_RB
SLOT = 0.5e-3
BW = N_RB * 12 * SCS
PRG = 4
N_RE_PRB = tbs_mod.re_per_rb(13, 24, 0)    # 13 symbols, 2 DM-RS symbols: 132 RE/PRB
N_RE = N_RE_PRB * N_RB                     # 4224 RE per layer
DS_NS = 30.0
SPEED_KMH = 3.0
MCS_TABLE = 2
RANKS = (1, 2, 3, 4)
QMS = (2, 4, 6, 8)
SNRS = np.arange(-10.0, 41.0, 1.0)
SLOPE_SCALES = (1.0, 4.0)
FREQS = (np.arange(N_SC) - N_SC / 2 + 0.5) * SCS

MAPPINGS = {
    "none": vrb_to_prb_non_interleaved(N_RB),
    "nr": vrb_to_prb_nr_interleaved(N_RB, 4),
    "blk4x8": vrb_to_prb_block_interleaver(4, 8),
}
VARIANTS = ("none", "nr", "blk4x8", "ideal")
LABELS = {
    "none": "no interleaving",
    "nr": "FD interleaving 1: NR, L = 4",
    "blk4x8": "FD interleaving 2: 1 PRB, 4 x 8",
    "ideal": "ideal interleaving (every CB sees the slot mean)",
}

ASSUMPTIONS = [
    ("Carrier", "3.5 GHz, 30 kHz SCS, 32 PRB (11.52 MHz), whole BW allocated"),
    ("Channel", "CDL-A, DS 30 ns, 3 km/h, one snapshot per drop"),
    ("gNB array", "32 ports, (M, N, P) = (2, 8, 2), +-45 deg, 0.5 wavelength, omni element"),
    ("UE array", "4 ports, (1, 2, 2), +-45 deg, 0.5 wavelength, omni element"),
    ("Precoding", "SVD per PRG of 4 PRBs (PRG-summed covariance), ideal CSI, "
                  "equal power per layer"),
    ("Receiver", "LMMSE, ideal channel estimation"),
    ("CW-to-layer mapping", "NR: one codeword, ranks 1-4"),
    ("Rank / MCS", "ideal AMC per drop and SNR: rank 1-4 and MCS (Table 2, 256QAM) "
                   "with the most expected bits, per mapping"),
    ("PDSCH", "13 symbols, 2 DM-RS symbols (type 1, 2 CDM groups without data): "
              "132 RE/PRB, 4224 RE per layer"),
    ("Coding", "TBS per TS 38.214, BG + CB segmentation + E_r per TS 38.212, "
               "frequency-first RE mapping over VRBs"),
    ("BLER", "per-CB MIESM + LDPC waterfall at K'/E_r; TB BLER = 1 - prod(1 - BLER_CB)"),
    ("FD interleaving 1", "NR VRB-to-PRB interleaving, L = 4, R = 2 (TS 38.211 7.3.1.6)"),
    ("FD interleaving 2", "1-PRB granularity, 4 x 8 block interleaver, write rows / "
                          "read columns: VRB v -> PRB 4 (v mod 8) + floor(v / 8)"),
]


# ---------------------------------------------------------------------------
# static tables
# ---------------------------------------------------------------------------
def cb_tables() -> dict:
    """Per rank: TBS of every MCS and the CBs of every MCS, concatenated."""
    tabs = {}
    for r in RANKS:
        tbs, first, qi, spans, k_prime, thr, slope = [], [], [], [], [], [], []
        n_cb = 0
        for i in range(mcs_tables.num_mcs(MCS_TABLE)):
            m = mcs_tables.get_mcs(i, MCS_TABLE)
            qm, rate = m.modulation_order, m.target_code_rate
            tb = tbs_mod.compute_tbs(N_RE_PRB, N_RB, qm, rate, r)
            seg = segment_transport_block(tb, rate)
            e = rate_match_lengths(N_RE * qm * r, seg.c, r, qm)
            tbs.append(tb)
            first.append(n_cb)
            n_cb += seg.c
            for e_r in e:
                t, s = _waterfall(qm, seg.k_prime / e_r, seg.k_prime)
                thr.append(t)
                slope.append(s)
            qi += [QMS.index(qm)] * seg.c
            spans.append(code_block_spans(e, qm, r))
            k_prime.append(seg.k_prime)
        tabs[r] = dict(tbs=np.array(tbs, float), first=np.array(first), qi=np.array(qi),
                       spans=np.concatenate(spans), thr=np.array(thr),
                       slope=np.outer(SLOPE_SCALES, slope),
                       n_cb=np.diff(first + [n_cb]), k_prime=np.array(k_prime))
    return tabs


def eff_sinr_db(mean_mi: np.ndarray, qi: np.ndarray) -> np.ndarray:
    """Invert the BICM curve of each CB's modulation (MIESM, as in
    link_abstraction.effective_sinr_miesm)."""
    out = np.empty(mean_mi.shape)
    for k, qm in enumerate(QMS):
        m = qi == k
        if m.any():
            grid_db, cap = _bicm_table(qm)
            out[..., m] = np.interp(np.clip(mean_mi[..., m], 1e-6, qm - 1e-6), cap, grid_db)
    return out


# ---------------------------------------------------------------------------
# channel and per-RE SINR
# ---------------------------------------------------------------------------
def make_channel(seed: int, ds_ns: float = DS_NS) -> CDLChannel:
    return CDLChannel("CDL-A", ds_ns, SPEED_KMH / 3.6 * FC / 3e8, n_tx=32, n_rx=4,
                      carrier_freq_hz=FC, tx_pol=2, rx_pol=2, tx_layout=(2, 8),
                      rx_layout=(1, 2), rng=np.random.default_rng(seed))


def prg_precoder(H: np.ndarray, rank: int) -> np.ndarray:
    """SVD precoder per PRG: top eigenvectors of sum_{sc in PRG} H^H H.

    H: (n_sc, n_rx, n_tx).  Returns unit-norm columns W (n_sc, n_tx, rank).
    """
    n_prg = N_RB // PRG
    Hg = H.reshape(n_prg, -1, *H.shape[1:])
    Rg = np.einsum("gfus,gfut->gst", Hg.conj(), Hg)
    _, V = np.linalg.eigh(Rg)                             # ascending eigenvalues
    W = V[:, :, ::-1][:, :, :rank]
    return np.repeat(W, H.shape[0] // n_prg, axis=0)


def drop_channel(seed: int, ds_ns: float = DS_NS) -> np.ndarray:
    """Per-subcarrier channel of one drop, (n_sc, n_rx, n_tx)."""
    return make_channel(seed, ds_ns).frequency_response(FREQS, 0.0)


def layer_mi(H: np.ndarray, rank: int, snrs=SNRS):
    """Per-subcarrier, per-layer BICM MI (n_snr, n_qm, n_sc, rank) and SINR."""
    Heff = H @ (prg_precoder(H, rank) / np.sqrt(rank))
    sinr = np.stack([batch_mmse_sinr(Heff, 10 ** (-s / 10)) for s in snrs])
    sdb = 10 * np.log10(np.clip(sinr, 1e-6, 1e6))
    return np.stack([bicm_capacity(sdb, q) for q in QMS], axis=1), sinr


def cb_eff_sinr(mi: np.ndarray, t: dict, variant: str) -> np.ndarray:
    """Effective SINR (dB) of every CB of every MCS: (..., n_cb_total)."""
    if variant == "ideal":
        mean = mi.mean(axis=(-2, -1))[..., t["qi"]]
    else:
        cbm = code_block_means(mi, t["spans"], MAPPINGS[variant])   # (..., n_qm, n_cb)
        mean = cbm[..., t["qi"], np.arange(t["qi"].size)]
    return eff_sinr_db(mean, t["qi"])


def tb_bler(eff: np.ndarray, t: dict) -> np.ndarray:
    """TB BLER (n_scale, ..., n_mcs) from the CB effective SINRs (..., n_cb)."""
    p = 1.0 / (1.0 + np.exp(t["slope"].reshape(len(SLOPE_SCALES), *([1] * (eff.ndim - 1)), -1)
                            * (eff - t["thr"])))
    ok = np.add.reduceat(np.log1p(-np.minimum(p, 1 - 1e-15)), t["first"], axis=-1)
    return -np.expm1(ok)


# ---------------------------------------------------------------------------
# simulation
# ---------------------------------------------------------------------------
def _worker(args):
    seeds, ds_ns = args
    tabs = cb_tables()
    n_mcs = mcs_tables.num_mcs(MCS_TABLE)
    nv, ns, nr, nq = len(VARIANTS), len(SLOPE_SCALES), len(RANKS), len(SNRS)
    se = np.zeros((len(seeds), nq, nv, ns, nr))           # best SE per rank
    mcs = np.zeros((len(seeds), nq, nv, ns, nr), dtype=np.int16)
    bler = np.zeros((nq, nv, ns, nr, n_mcs))              # sum over drops
    spread = np.zeros((nq, nv, nr, n_mcs))                # max - min CB eff SINR (dB)
    for d, seed in enumerate(seeds):
        H = drop_channel(seed, ds_ns)
        for ir, r in enumerate(RANKS):
            t = tabs[r]
            mi, _ = layer_mi(H, r)
            for iv, v in enumerate(VARIANTS):
                eff = cb_eff_sinr(mi, t, v)                    # (nq, n_cb)
                b = tb_bler(eff, t)                            # (ns, nq, n_mcs)
                bits = t["tbs"] * (1 - b)
                se[d, :, iv, :, ir] = bits.max(-1).T / SLOT / BW
                mcs[d, :, iv, :, ir] = bits.argmax(-1).T
                bler[:, iv, :, ir] += np.moveaxis(b, 0, 1)
                spread[:, iv, ir] += (np.maximum.reduceat(eff, t["first"], axis=-1)
                                      - np.minimum.reduceat(eff, t["first"], axis=-1))
    return se, mcs, bler, spread


def run(n_drops: int, n_jobs: int, ds_ns: float = DS_NS):
    seeds = np.arange(n_drops)
    chunks = [seeds[k::n_jobs] for k in range(n_jobs) if len(seeds[k::n_jobs])]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        parts = list(ex.map(_worker, [(c, ds_ns) for c in chunks]))
    order = np.argsort(np.concatenate(chunks))
    se = np.concatenate([p[0] for p in parts])[order]
    mcs = np.concatenate([p[1] for p in parts])[order]
    bler = sum(p[2] for p in parts) / n_drops
    spread = sum(p[3] for p in parts) / n_drops
    return se, mcs, bler, spread


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------
def snr_at_bler(snr: np.ndarray, bler: np.ndarray, target: float = 0.1):
    """SNR where a decreasing mean BLER curve first crosses ``target``."""
    below = np.flatnonzero(bler <= target)
    if below.size == 0 or below[0] == 0:
        return None
    i = below[0]
    y0, y1 = np.log10(max(bler[i - 1], 1e-12)), np.log10(max(bler[i], 1e-12))
    return float(snr[i - 1] + (snr[i] - snr[i - 1]) * (y0 - np.log10(target)) / (y0 - y1))


def summarize(se, mcs, bler, spread, tabs, n_boot: int = 1000) -> dict:
    n = se.shape[0]
    best = se.max(-1)                                     # (drops, nq, nv, ns)
    rank = se.argmax(-1) + 1
    mcs_sel = np.take_along_axis(mcs, se.argmax(-1)[..., None], -1)[..., 0]
    mean = best.mean(0)                                   # (nq, nv, ns)
    # paired bootstrap over drops of the ratio of mean SEs
    w = np.random.default_rng(7).multinomial(n, np.full(n, 1 / n), size=n_boot) / n
    boot = (w @ best.reshape(n, -1)).reshape(n_boot, *best.shape[1:])
    ratio = boot / np.maximum(boot[:, :, :1], 1e-12)
    gain = 100 * (mean / np.maximum(mean[:, :1], 1e-12) - 1)
    lo, hi = (100 * (np.percentile(ratio, p, axis=0) - 1) for p in (2.5, 97.5))
    res = {"se": {}, "gain_pct": {}, "gain_ci95_pct": {}, "avg_rank": {},
           "avg_mcs": {}, "rank_pmf": {}, "snr10_db": {}, "cb_spread_db": {}}
    for iv, v in enumerate(VARIANTS):
        for js, s in enumerate(SLOPE_SCALES):
            key = f"{v}@x{s:g}"
            res["se"][key] = mean[:, iv, js].round(4).tolist()
            res["gain_pct"][key] = gain[:, iv, js].round(3).tolist()
            res["gain_ci95_pct"][key] = [lo[:, iv, js].round(3).tolist(),
                                         hi[:, iv, js].round(3).tolist()]
            res["avg_rank"][key] = rank[:, :, iv, js].mean(0).round(3).tolist()
            res["avg_mcs"][key] = mcs_sel[:, :, iv, js].mean(0).round(3).tolist()
            res["rank_pmf"][key] = [np.mean(rank[:, :, iv, js] == r, axis=0).round(3).tolist()
                                    for r in RANKS]
            res["snr10_db"][key] = {
                str(r): [snr_at_bler(SNRS, bler[:, iv, js, ir, m])
                         for m in range(bler.shape[-1])] for ir, r in enumerate(RANKS)}
        res["cb_spread_db"][v] = {str(r): spread[:, iv, ir].round(3).tolist()
                                  for ir, r in enumerate(RANKS)}
    res["bler"] = {f"{v}@x{s:g}": {str(r): bler[:, iv, js, ir].T.round(5).tolist()
                                   for ir, r in enumerate(RANKS)}
                   for iv, v in enumerate(VARIANTS) for js, s in enumerate(SLOPE_SCALES)}
    res["n_cb"] = {str(r): tabs[r]["n_cb"].tolist() for r in RANKS}
    res["tbs"] = {str(r): tabs[r]["tbs"].astype(int).tolist() for r in RANKS}
    res["k_prime"] = {str(r): tabs[r]["k_prime"].tolist() for r in RANKS}
    return res


def print_summary(res: dict) -> None:
    pts = (0.0, 10.0, 20.0, 25.0, 30.0, 35.0)
    ii = [int(np.argmin(np.abs(SNRS - p))) for p in pts]
    for s in SLOPE_SCALES:
        print(f"\nwaterfall slope x{s:g}: SE (b/s/Hz) and gain over no interleaving (%)")
        print("  SNR (dB)          " + "".join(f"{p:>13.0f}" for p in pts))
        for v in VARIANTS:
            k = f"{v}@x{s:g}"
            print(f"  {v:8s} SE        " + "".join(f"{res['se'][k][i]:13.3f}" for i in ii))
            if v != "none":
                lo, hi = res["gain_ci95_pct"][k]
                print(f"  {v:8s} gain %    "
                      + "".join(f"{res['gain_pct'][k][i]:13.2f}" for i in ii))
                print(f"  {'':8s} 95% CI    "
                      + "".join(f"{lo[i]:+6.2f}..{hi[i]:+5.2f}".rjust(13) for i in ii))
        for v in VARIANTS:
            print(f"  {v:8s} avg rank  " + "".join(
                f"{res['avg_rank'][f'{v}@x{s:g}'][i]:13.2f}" for i in ii))
    print("\nSNR at 10% TB BLER (dB) and gain over no interleaving, rank 4 / MCS 19-27")
    for s in SLOPE_SCALES:
        for m in (19, 23, 27):
            ref = res["snr10_db"][f"none@x{s:g}"]["4"][m]
            row = [f"{ref:6.2f}" if ref is not None else "   n/a"]
            for v in VARIANTS[1:]:
                x = res["snr10_db"][f"{v}@x{s:g}"]["4"][m]
                row.append("  n/a" if x is None or ref is None else f"{ref - x:+5.2f}")
            print(f"  x{s:g} MCS {m:2d} (C = {res['n_cb']['4'][m]:2d}): none {row[0]}  "
                  + "  ".join(f"{v} {g}" for v, g in zip(VARIANTS[1:], row[1:])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drops", type=int, default=1000)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--ds", type=float, default=DS_NS,
                    help="delay spread (ns); other values are a cross-check, "
                         "written to fd_interleaving_ds<DS>.json without a plot")
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()
    t0 = time.time()
    se, mcs, bler, spread = run(args.drops, args.jobs, args.ds)
    res = {"assumptions": ASSUMPTIONS, "snr_db": SNRS.tolist(), "n_drops": args.drops,
           "ds_ns": args.ds,
           "variants": {v: LABELS[v] for v in VARIANTS},
           "slope_scales": list(SLOPE_SCALES),
           "mappings": {k: v.tolist() for k, v in MAPPINGS.items()}}
    res.update(summarize(se, mcs, bler, spread, cb_tables()))
    res["runtime_s"] = round(time.time() - t0, 1)
    print(f"{args.drops} drops in {res['runtime_s']} s")
    print_summary(res)
    os.makedirs(OUT, exist_ok=True)
    name = "fd_interleaving" if args.ds == DS_NS else f"fd_interleaving_ds{args.ds:g}"
    with open(os.path.join(OUT, name + ".json"), "w") as fh:
        json.dump(res, fh, separators=(",", ":"))
    if not args.no_plot and args.ds == DS_NS:
        from fd_interleaving_plot import plot   # noqa: E402  (examples/ on sys.path)
        plot(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

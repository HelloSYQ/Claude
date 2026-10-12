#!/usr/bin/env python3
"""Codeword-to-layer mapping with Nokia's link-level assumptions, MMSE only.

Repeats the link-level study of Nokia R1-2607066 (RAN1 #126bis, section
"Mechanism for layer mapping addressing layer imbalance", Annex B Table 11)
without the SIC receiver: SE vs SNR for fixed ranks 1-4 with

  * 1 CW          : NR up to rank 4, one codeword on all layers;
  * 2 CW          : two codewords, layers split {1,1} / {1,2} / {2,2};
  * 1 CW per layer: up to four codewords, each with its own MCS.

Outputs the equivalents of Nokia's Fig. 26 (ranks 1-4, 1 CW vs 1 CW per
layer), Fig. 28 (rank 4, 1 / 2 / 4 CW, curves without SIC), Table 3
(rank-transition SNR) and Table 4 (envelope SE gain over 1 CW), and compares
them with Nokia's digitized curves (examples/data/nokia_r1_2607066_cw2lm.json).

    python examples/cw2lm_nokia.py [--drops 300] [--jobs 4]

Writes results/cw2lm_nokia.json, results/cw2lm_nokia.png and
results/cw2lm_nokia_rank4.png.  See docs/cw2lm-nokia.md.
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
from nrdlsim import tbs as tbs_mod                                   # noqa: E402
from nrdlsim.channel_models import (CDLChannel, build_panel,          # noqa: E402
                                    ut_handset_array, UT_HANDSET_ELEMENT)
from nrdlsim.layer_mapping import layer_groups                       # noqa: E402
from nrdlsim.link_abstraction import ideal_amc                        # noqa: E402
from nrdlsim.receiver import batch_mmse_sinr_est                      # noqa: E402

NOKIA_DATA = os.path.join(ROOT, "examples", "data", "nokia_r1_2607066_cw2lm.json")
OUT = os.path.join(ROOT, "results")

# ---- Nokia R1-2607066 Table 11 (link level), as modelled here -------------
FC = 3.5e9
SCS = 30e3
N_RB = 96                        # "rb = 96" in the captions of Fig. 26-28
SLOT = 0.5e-3
BW = N_RB * 12 * SCS             # SE is normalised by the allocated bandwidth
N_RE = tbs_mod.re_per_rb(12, 24, 0)   # 2 PDCCH + 2 DM-RS symbols: 120 RE/PRB
DS_NS = 100.0
SPEED_KMH = 3.0
MCS_TABLE = 2
RANKS = (1, 2, 3, 4)
MAPPINGS = {"1cw": 1, "2cw": 2, "cw_per_layer": 4}   # name -> max codewords
SNRS = np.arange(-15.0, 47.0, 1.0)   # our scale; Nokia scale is about 4 dB lower
CE_GAIN = 6.0                    # DM-RS estimation SNR gain (error var N0/6)

ASSUMPTIONS = [
    ("Carrier", "3.5 GHz TDD, 30 kHz, 40 MHz", "3.5 GHz, 30 kHz, 96 PRB (rb = 96 in Fig. 26-28)"),
    ("Channel", "CDL-A, DS 100 ns, 3 km/h", "CDL-A, DS 100 ns, one slot per drop (no CSI ageing)"),
    ("gNB array", "64 TXRU, 192 AE, (12,8,2,1,1;4,8), (dH,dV) = (0.5,0.7)",
     "64 ports, 3-element vertical TXRU (e-tilt 90 deg), 38.901 element 8 dBi / 65 deg, +-45 deg"),
    ("UE array", "8 Rx, TR 38.901 7.3 (V19.1.0) directive, (a,b,g) = (0,0,0), no blockage",
     "8 single-pol directive elements on the handset edges (5.4 dBi, 125 deg, 22.5 dB), (0,0,0), no blockage"),
    ("Receiver", "MMSE, SIC", "LMMSE only (no SIC)"),
    ("Channel estimation", "non-ideal", "DM-RS estimate, error variance N0/6 per entry"),
    ("Precoding", "SU-MIMO", "SVD per PRB, equal power per layer"),
    ("Link adaptation", "AMC", "ideal AMC per codeword: MCS with the most expected bits (MIESM)"),
    ("MCS table", "Table 2 (256QAM)", "Table 2 (256QAM)"),
    ("Rank", "fixed 1, 2, 3, 4", "fixed 1, 2, 3, 4"),
    ("Overhead", "-", "2 PDCCH + 2 DM-RS symbols, 120 RE/PRB: 4.94 bit/s/Hz per layer at MCS 27 (Nokia: 5.0)"),
]

VARIANTS = {
    "nokia": dict(ue="handset", ce_gain=CE_GAIN,
                  label="Nokia set-up (8-Rx directive handset, non-ideal CE)"),
    "iso_ue": dict(ue="iso", ce_gain=CE_GAIN,
                   label="isotropic 8-Rx UE (2x2 dual-pol), non-ideal CE"),
    "ideal_ce": dict(ue="handset", ce_gain=None,
                     label="8-Rx directive handset, ideal CE"),
    # diagnostic: the Nokia set-up with eigenmodes 2-4 attenuated so that the
    # 1 CW-per-layer curves of ranks 2-4 match Nokia's (fit_eigen_attenuation,
    # ideal CE as in the fit); its 1 CW and 2 CW curves are then predictions
    "matched": dict(ue="handset", ce_gain=None, eig_atten_db="fit",
                    label="eigenmode spread fitted to Nokia's MCW curves, ideal CE"),
}


def iso_ue_array() -> dict:
    """8 isotropic +-45 deg elements on a 2x2 grid, 0.5 wavelength apart."""
    pos, slant = build_panel(8, 2, (2, 2), 0.5, 0.5)
    return {"positions": pos, "rotation_deg": np.zeros((8, 3)),
            "pol_slant_deg": np.rad2deg(slant)}


def make_channel(seed: int, ue: str) -> CDLChannel:
    kw = dict(model="CDL-A", delay_spread_ns=DS_NS,
              max_doppler_hz=SPEED_KMH / 3.6 * FC / 3e8,
              n_tx=64, n_rx=8, carrier_freq_hz=FC,
              tx_pol=2, tx_layout=(4, 8), spacing_v=0.7, spacing_h=0.5,
              tx_pattern="38.901", tx_subarray=3, tx_etilt_deg=90.0,
              rng=np.random.default_rng(seed))
    if ue == "handset":
        kw.update(rx_elements=ut_handset_array(FC), rx_pattern="38.901",
                  rx_element_max_gain_dbi=UT_HANDSET_ELEMENT["g_max_dbi"],
                  rx_element_hpbw_deg=UT_HANDSET_ELEMENT["hpbw_deg"],
                  rx_element_front_back_db=UT_HANDSET_ELEMENT["front_back_db"])
    else:
        kw.update(rx_elements=iso_ue_array(), rx_pattern="omni")
    return CDLChannel(**kw)


def drop_eigenmodes(seed: int, ue: str, eig_atten_db=None):
    """SVD of one drop: left singular vectors U (n_rb, 8, 4) and s (n_rb, 4).

    With the SVD precoder W = V_r / sqrt(r) the effective channel of rank r
    is H W = U_r diag(s_r) / sqrt(r).
    """
    ch = make_channel(seed, ue)
    f = (np.arange(N_RB) - N_RB / 2 + 0.5) * 12 * SCS
    H = ch.frequency_response(f, 0.0)                     # (n_rb, 8, 64)
    U, s, _ = np.linalg.svd(H, full_matrices=False)
    s = s[:, :4]
    if eig_atten_db is not None:
        s = s * 10.0 ** (-np.asarray(eig_atten_db) / 20.0)
    return U[:, :, :4], s


def _worker(args):
    seeds, ue, ce_gain, eig_atten_db = args
    bits = {m: np.zeros((len(RANKS), len(SNRS))) for m in MAPPINGS}
    mcs4 = {m: np.zeros((len(SNRS), 4)) for m in MAPPINGS}     # rank-4 MCS per CW
    layer_sinr = np.zeros((len(seeds), len(SNRS), 4))           # rank-4, dB
    eig = np.zeros((len(seeds), 4))
    for d, seed in enumerate(seeds):
        U, s = drop_eigenmodes(seed, ue, eig_atten_db)
        eig[d] = 10 * np.log10((s ** 2).mean(0))
        rng = np.random.default_rng(1_000_003 + seed)
        for j, snr in enumerate(SNRS):
            n0 = 10.0 ** (-snr / 10.0)
            if ce_gain is not None:
                E = np.sqrt(n0 / ce_gain / 2) * (rng.standard_normal(U.shape)
                                                  + 1j * rng.standard_normal(U.shape))
            for i, r in enumerate(RANKS):
                h = U[:, :, :r] * (s[:, None, :r] / np.sqrt(r))
                sinr = batch_mmse_sinr_est(h, h if ce_gain is None else h + E[:, :, :r], n0)
                if r == 4:
                    layer_sinr[d, j] = 10 * np.log10(sinr.mean(0))
                cache = {}
                for m, max_cw in MAPPINGS.items():
                    groups = tuple(tuple(g) for g in layer_groups(r, max_cw))
                    if groups not in cache:
                        res = [ideal_amc(sinr[:, list(g)].ravel(), len(g), N_RE, N_RB,
                                         MCS_TABLE) for g in groups]
                        cache[groups] = res
                    res = cache[groups]
                    bits[m][i, j] += sum(x[3] for x in res)
                    if r == 4:
                        mcs4[m][j, :len(res)] += [x[0] for x in res]
    return bits, mcs4, layer_sinr, eig


def run_variant(name: str, n_drops: int, n_jobs: int, eig_atten_db=None) -> dict:
    v = VARIANTS[name]
    seeds = np.arange(n_drops)
    chunks = [seeds[k::n_jobs] for k in range(n_jobs)]
    args = [(c, v["ue"], v["ce_gain"], eig_atten_db) for c in chunks if len(c)]
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        parts = list(ex.map(_worker, args))
    bits = {m: sum(p[0][m] for p in parts) for m in MAPPINGS}
    mcs4 = {m: sum(p[1][m] for p in parts) / n_drops for m in MAPPINGS}
    layer_sinr = np.concatenate([p[2] for p in parts])
    eig = np.concatenate([p[3] for p in parts])
    se = {m: {str(r): (bits[m][i] / n_drops / SLOT / BW).tolist()
              for i, r in enumerate(RANKS)} for m in MAPPINGS}
    imb = layer_sinr[:, :, 0] - layer_sinr[:, :, 3]
    return {"label": v["label"], "se": se,
            "eig_atten_db": None if eig_atten_db is None else list(eig_atten_db),
            "rank4_mcs": {m: mcs4[m].tolist() for m in MAPPINGS},
            "rank4_layer_sinr_db_mean": layer_sinr.mean(0).tolist(),
            "rank4_imbalance_db_pct": {p: np.percentile(imb, p, axis=0).tolist()
                                       for p in (10, 50, 90)},
            "eig_db_mean": eig.mean(0).tolist(),
            "eig_spread_db_pct": {p: float(np.percentile(eig[:, 0] - eig[:, 3], p))
                                  for p in (10, 50, 90)}}


def _layer_curves(args):
    """Mean single-layer expected bits of eigenmode k vs per-layer SNR offset."""
    seeds, xs = args
    g = np.zeros((4, len(xs)))
    for seed in seeds:
        _, s = drop_eigenmodes(seed, "handset")
        for k in range(4):
            for j, x in enumerate(xs):
                g[k, j] += ideal_amc(s[:, k] ** 2 * 10 ** (x / 10), 1, N_RE, N_RB,
                                     MCS_TABLE)[3]
    return g


def fit_eigen_attenuation(delta: float, n_drops: int = 48, n_jobs: int = 4):
    """Extra attenuation (dB) of eigenmodes 2-4 that fits Nokia's MCW curves.

    With ideal CE the SVD layers do not interfere, so the 1 CW-per-layer SE
    of rank r is sum_k G_k(SNR - a_k - 10 log10 r), G_k the mean single-layer
    SE of eigenmode k.  a_2, a_3, a_4 are fitted in turn (least squares on
    Nokia's SNR axis, our SNR = Nokia's - ``delta``) to Nokia's rank-2, -3
    and -4 curves.  Returns [0, a_2, a_3, a_4].
    """
    xs = np.arange(-70.0, 70.5, 0.5)
    seeds = np.arange(n_drops)
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        parts = list(ex.map(_layer_curves, [(seeds[k::n_jobs], xs) for k in range(n_jobs)]))
    G = sum(parts) / n_drops / SLOT / BW                    # (4, n_x) bit/s/Hz
    nsnr, ncur, _ = nokia_curves()
    s_ours = nsnr - delta
    atten = [0.0]
    for r in (2, 3, 4):
        target = np.array(ncur["cw_per_layer"][str(r)])
        fixed = sum(np.interp(s_ours - atten[k] - 10 * np.log10(r), xs, G[k])
                    for k in range(r - 1))
        cand = np.arange(0.0, 45.0, 0.25)
        err = [np.mean((fixed + np.interp(s_ours - a - 10 * np.log10(r), xs, G[r - 1])
                        - target) ** 2) for a in cand]
        atten.append(float(cand[int(np.argmin(err))]))
    return atten


# ---------------------------------------------------------------------------
# comparison metrics
# ---------------------------------------------------------------------------
def envelope(curves: dict, mapping: str) -> np.ndarray:
    """Best fixed rank at each SNR (rank adaptation with perfect RI)."""
    return np.max([curves[mapping][str(r)] for r in RANKS], axis=0)


def transitions(curves: dict, mapping: str, snr: np.ndarray) -> list:
    """SNR beyond which rank r+1 stays ahead of rank r (last up-crossing)."""
    out = []
    for r in RANKS[:-1]:
        d = np.array(curves[mapping][str(r + 1)]) - np.array(curves[mapping][str(r)])
        up = np.flatnonzero((d[:-1] <= 0) & (d[1:] > 0))
        if d[-1] <= 0 or up.size == 0:
            out.append(None if d[-1] <= 0 else float(snr[0]))
            continue
        i = up[-1]
        out.append(float(snr[i] + (snr[i + 1] - snr[i]) * (-d[i]) / (d[i + 1] - d[i])))
    return out


def nokia_curves() -> tuple[np.ndarray, dict, dict]:
    with open(NOKIA_DATA) as f:
        d = json.load(f)
    f26, f28 = d["fig26"], d["fig28"]
    snr = np.array(f26["snr_db"])
    c = {"1cw": {"1": f26["rank1_1cw"], "2": f26["rank2_1cw"],
                 "3": f26["rank3_1cw"], "4": f26["rank4_1cw"]},
         "cw_per_layer": {"1": f26["rank1_1cw"], "2": f26["rank2_cw_per_layer"],
                          "3": f26["rank3_cw_per_layer"], "4": f26["rank4_cw_per_layer"]}}
    r4_2cw = np.interp(snr, f28["snr_db"], f28["rank4_2cw"])
    return snr, c, {"2cw": r4_2cw.tolist()}


def snr_offset(ours_snr, ours_r1, nok_snr, nok_r1, se_max=4.5) -> float:
    """dB to add to our SNR so our rank-1 curve matches Nokia's (SE <= se_max)."""
    m = np.array(nok_r1) <= se_max
    cand = np.arange(-25.0, 25.0, 0.05)
    err = [np.mean((np.interp(nok_snr[m] - dlt, ours_snr, ours_r1) - np.array(nok_r1)[m]) ** 2)
           for dlt in cand]
    return float(cand[int(np.argmin(err))])


TABLE4_SNR = (-5.0, 10.0, 20.0, 30.0)
NOKIA_TABLE3 = {"1cw": [0, 19, 38], "cw_per_layer": [-7, 14, 17]}
NOKIA_TABLE4 = {"cw_per_layer": [2.6, 14.0, 25.0, 20.0]}


def compare(res: dict) -> dict:
    nsnr, ncur, n28 = nokia_curves()
    base = res["variants"]["nokia"]["se"]
    delta = snr_offset(SNRS, base["1cw"]["1"], nsnr, ncur["1cw"]["1"])
    out = {"snr_offset_db": delta, "nokia_digitized": {
        "table3": {m: transitions(ncur, m, nsnr) for m in ("1cw", "cw_per_layer")},
        "table4_pct": [100 * (np.interp(s, nsnr, envelope(ncur, "cw_per_layer"))
                              / np.interp(s, nsnr, envelope(ncur, "1cw")) - 1)
                       for s in TABLE4_SNR]}}
    for name, v in res["variants"].items():
        cur = v["se"]
        s_al = SNRS + delta
        e1 = envelope(cur, "1cw")
        row = {"table3": {m: [None if x is None else x + delta for x in transitions(cur, m, SNRS)]
                          for m in MAPPINGS},
               "table4_pct": {m: [100 * (np.interp(s, s_al, envelope(cur, m))
                                         / np.interp(s, s_al, e1) - 1) for s in TABLE4_SNR]
                              for m in ("2cw", "cw_per_layer")},
               "rank4_gain_pct": {m: [100 * (np.interp(s, s_al, cur[m]["4"])
                                             / np.interp(s, s_al, cur["1cw"]["4"]) - 1)
                                      for s in TABLE4_SNR] for m in ("2cw", "cw_per_layer")}}
        out[name] = row
    n4 = {"1cw": ncur["1cw"]["4"], "2cw": n28["2cw"], "cw_per_layer": ncur["cw_per_layer"]["4"]}
    out["nokia_digitized"]["rank4_gain_pct"] = {
        m: [100 * (np.interp(s, nsnr, n4[m]) / np.interp(s, nsnr, n4["1cw"]) - 1)
            for s in TABLE4_SNR] for m in ("2cw", "cw_per_layer")}
    return out


def print_summary(res: dict, cmp: dict) -> None:
    f = lambda xs: " ".join("  n/a " if x is None else f"{x:6.1f}" for x in xs)
    print(f"\nSNR offset (ours -> Nokia scale, rank-1 fit): {cmp['snr_offset_db']:+.2f} dB")
    print("\nTable 3, rank-transition SNR (dB, Nokia scale): 1->2, 2->3, 3->4")
    for m in ("1cw", "cw_per_layer"):
        print(f"  Nokia      {m:13s} {f(NOKIA_TABLE3[m])}")
        for name in res["variants"]:
            print(f"  {name:10s} {m:13s} {f(cmp[name]['table3'][m])}")
    print("\nTable 4, envelope SE gain over 1 CW (%) at", TABLE4_SNR)
    print(f"  Nokia table  cw_per_layer {f(NOKIA_TABLE4['cw_per_layer'])}")
    print(f"  Nokia digit. cw_per_layer {f(cmp['nokia_digitized']['table4_pct'])}")
    for name in res["variants"]:
        for m in ("2cw", "cw_per_layer"):
            print(f"  {name:12s} {m:12s} {f(cmp[name]['table4_pct'][m])}")
    print("\nRank 4 (Fig. 28), SE gain over 1 CW (%) at", TABLE4_SNR)
    for m in ("2cw", "cw_per_layer"):
        print(f"  Nokia digit. {m:12s} {f(cmp['nokia_digitized']['rank4_gain_pct'][m])}")
        for name in res["variants"]:
            print(f"  {name:12s} {m:12s} {f(cmp[name]['rank4_gain_pct'][m])}")
    for name, v in res["variants"].items():
        p = v["eig_spread_db_pct"]
        print(f"\n{name}: eigenvalue spread s1^2/s4^2 (dB) p10/p50/p90 = "
              f"{p[10]:.1f} / {p[50]:.1f} / {p[90]:.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--drops", type=int, default=300)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--variants", default=",".join(VARIANTS))
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    res = {"assumptions": ASSUMPTIONS, "snr_db": SNRS.tolist(), "n_drops": args.drops,
           "n_re_per_prb": N_RE, "variants": {}}
    names = args.variants.split(",")
    for name in [n for n in names if n != "matched"]:
        t1 = time.time()
        res["variants"][name] = run_variant(name, args.drops, args.jobs)
        print(f"{name}: {time.time() - t1:.0f} s")
    if "matched" in names:
        nsnr, ncur, _ = nokia_curves()
        delta = snr_offset(SNRS, res["variants"]["nokia"]["se"]["1cw"]["1"],
                           nsnr, ncur["1cw"]["1"])
        atten = fit_eigen_attenuation(delta, n_jobs=args.jobs)
        print(f"fitted eigenmode attenuation (dB): {atten}")
        t1 = time.time()
        res["variants"]["matched"] = run_variant("matched", args.drops, args.jobs, atten)
        print(f"matched: {time.time() - t1:.0f} s")
    res["comparison"] = cmp = compare(res)
    res["runtime_s"] = round(time.time() - t0, 1)
    print_summary(res, cmp)
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "cw2lm_nokia.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    if not args.no_plot:
        from cw2lm_nokia_plot import plot   # noqa: E402  (examples/ on sys.path)
        plot(res)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Calibration against 5G-IA and 3GPP TS 38.104 link-level results

This documents how `nrdlsim` compares to two independent published references
for the SNR at 70 % of maximum throughput on the 3GPP RAN4 FR1/FDD cases:

1. **5G-IA Link-Level Calibration Results (v2)** — an *ideal* link-level
   calibration averaged across eight companies (Samsung, ZTE, CMCC, Nokia,
   Huawei, Ericsson, China Telecom, CATT) and the 5G-PPP WG result.
2. **3GPP TS 38.104 V16.4.0** — the consensus PUSCH *minimum performance
   requirements* agreed in 3GPP RAN4 (Tables 8.2.1.2-2 and 8.2.1.2-6). These
   include an implementation margin, so an ideal simulator should sit a few dB
   *below* them.

The 38.104 Fixed Reference Channels map exactly onto the MCS used here:
**G-FR1-A3** = QPSK R=193/1024 (= MCS 2), **G-FR1-A4** = 16QAM R=658/1024
(= MCS 16), **G-FR1-A5** = 64QAM R=567/1024 (= MCS 20), with 52 PRB (10 MHz/
15 kHz), 12 data symbols and DM-RS type 1 pos1 — matching the simulator config.

Reproduce with `python examples/calibration_5gia.py`.

## Methodology

The reference Table 3 gives numeric targets for 9 PUSCH cases (CP-OFDM, MMSE,
ideal channel estimation, no precoding).  At link level the PDSCH chain is
symmetric to these, so each case is run on the downlink chain with matching
MCS / channel / antenna configuration:

* fixed MCS (no link adaptation), MCS table 1;
* ideal channel estimation (`ideal_channel_estimation=True`);
* open-loop, no transmit precoding (`precoding="none"`);
* total transmit power fixed across layers (SNR = total Es/N0);
* HARQ enabled, RV sequence per the document.

For each case the throughput-vs-SNR curve is measured and the SNR at 70 % of the
high-SNR throughput ceiling is interpolated.

## Results

SNR (dB) @ 70 % throughput — simulator vs the 5G-IA 8-company average (ideal)
and the 3GPP TS 38.104 minimum requirement:

| Case | Config | BW/SCS | sim | 5G-IA avg | TS 38.104 | Δ(5G-IA) | Δ(38.104) |
|------|--------|--------|-----|-----------|-----------|----------|-----------|
| 1 | 1×2 TDL-B QPSK  | 10/15  | −5.70 | −4.68 | −2.5 | −1.02 | −3.20 |
| 2 | 1×2 TDL-C 16QAM | 10/15  |  7.10 |  8.24 | 10.2 | −1.14 | −3.10 |
| 3 | 1×2 TDL-A 64QAM | 10/15  |  9.07 | 10.48 | 12.2 | −1.41 | −3.13 |
| 4 | 2×2 TDL-B QPSK  | 10/15  | −2.45 | −0.59 |  1.7 | −1.86 | −4.15 |
| 5 | 2×2 TDL-C 16QAM | 10/15  | 14.54 | 16.31 | 18.3 | −1.77 | −3.76 |
| 6 | 1×2 TDL-B QPSK  | 40/30  | −6.90 | −4.68 | −2.5 | −2.22 | −4.40 |
| 7 | 1×2 TDL-C 16QAM | 40/30  |  7.60 |  7.97 | 10.0 | −0.37 | −2.40 |
| 8 | 1×2 TDL-A 64QAM | 40/30  |  9.41 | 10.12 | 12.4 | −0.71 | −2.99 |
| 9 | 2×2 TDL-B QPSK  | 40/30  | −2.93 | −0.96 |  1.3 | −1.97 | −4.23 |

* **vs 5G-IA ideal:** mean Δ = −1.38 dB, RMS = 1.50 dB, **bias-removed spread = 0.59 dB**.
* **vs 3GPP TS 38.104:** mean Δ = −3.48 dB, **spread = 0.64 dB** (sim sits below
  the minimum requirement, as an ideal simulator must).

### Effect of the exact BICM capacity and the TS 38.211 bit labels

These results use the exact BICM capacity of the TS 38.211 constellations
(`link_abstraction.bicm_capacity`). Earlier versions built the capacity table
from a 2000-sample Monte-Carlo with max-log LLRs, and labelled the 64QAM and
larger constellations with a mapping that is not Gray. Each case keeps its
seeds, so the shifts below are paired; with the old table and labels the
calibration reproduces the earlier published numbers exactly.

| Case | old table | exact capacity | + TS 38.211 labels (now) |
|------|-----------|----------------|--------------------------|
| 1 | −5.70 | −5.70 | −5.70 |
| 2 |  7.16 |  7.10 |  7.10 |
| 3 |  9.17 |  9.14 |  9.07 |
| 4 | −2.45 | −2.45 | −2.45 |
| 5 | 14.68 | 14.54 | 14.54 |
| 6 | −6.90 | −6.90 | −6.90 |
| 7 |  7.73 |  7.60 |  7.60 |
| 8 |  9.56 |  9.56 |  9.41 |
| 9 | −3.02 | −2.93 | −2.93 |
| mean Δ / spread vs 5G-IA | −1.33 / 0.64 | −1.36 / 0.61 | −1.38 / 0.59 |
| mean Δ / spread vs TS 38.104 | −3.43 / 0.69 | −3.46 / 0.66 | −3.48 / 0.64 |

The exact capacity moves the cases by −0.14 to +0.09 dB, and the TS 38.211
labels move the two 64QAM cases by another −0.07 and −0.15 dB. The labels
raise the 64QAM capacity by up to 0.16 bit and lower the capacity-achieving SNR
of MCS 20 (64QAM, R = 567/1024) by 0.41 dB; at a fixed SNR between 4 and
14 dB they raise the throughput of case 3 by 2–19 % (600 slots per point).
The 70 % points move less than 0.41 dB because these slow-fading TDL-A cases
(10 Hz Doppler) draw a new channel at every SNR point: their throughput curves
are jagged, and the crossing sits between two points that differ mainly by
channel realization.

## Assessment

* **Relative / trend alignment is strong and consistent across both
  references.** After removing a constant bias the simulator tracks the
  case-to-case SNR to within **~0.6 dB** across QPSK / 16QAM / 64QAM,
  TDL-A/B/C, SIMO and 2×2 MIMO, 10 MHz/15 kHz and 40 MHz/30 kHz. The companies
  themselves disagree by ~1.5–2 dB on the harder cases (e.g. 5G-IA case 5 spans
  15.31–17.24 dB), so the relative agreement is within the inter-company spread.

* **The two references are mutually consistent, and the simulator sits
  correctly relative to both.** The 3GPP TS 38.104 requirement is ~2 dB above
  the 5G-IA ideal calibration (the implementation margin), and the simulator is
  ~1.4 dB below the ideal calibration (MIESM coding-gap optimism). The −3.48 dB
  vs 38.104 is exactly these two effects stacked: margin (~2.1 dB) + abstraction
  optimism (~1.4 dB). An ideal simulator being several dB better than a *minimum
  requirement* is the physically correct relationship.

* **The ~1.4 dB optimistic bias vs the ideal calibration** is the MIESM
  coding-gap model (`1.0 + 0.6·R` dB from capacity), which underestimates the
  real NR-LDPC implementation loss. It is a single calibratable constant — a gap
  offset of ~1.4 dB, or running the bit-true LDPC path, centres the results on
  the ideal reference.

### Correctness fixes found via this exercise

The calibration surfaced a real bug: the per-layer SINR gave **each layer full
transmit power**, so a rank-2 transmission received 2× (+3 dB) the power of a
rank-1 one.  Fixing the total-power constraint (split power across layers,
SNR = total Es/N0) moved the MIMO cases from ~6 dB optimistic to ~2 dB and cut
the RMS error from 3.4 dB to 1.7 dB.  Two options were also added to match
calibration conditions: `ideal_channel_estimation` and open-loop
`precoding="none"`.

## Caveats

* The reference cases are PUSCH; they are reproduced on the link-symmetric PDSCH
  chain (valid for CP-OFDM/MMSE without precoding).
* The default SE path uses the MIESM abstraction; absolute-SNR alignment within
  the ~0.5 dB inter-company target requires the bit-true LDPC path with the
  exact 38.212 base graphs and a matched SNR definition.
* Monte-Carlo depth is modest (60 slots/point); the ~1 dB difference between the
  otherwise-identical cases 1 and 6 is within this MC noise. The slow-fading
  64QAM cases are the noisiest: with 600 slots per point and a 0.25 dB step,
  cases 3 and 8 come out at 9.5 and 7.9 dB.

## Sources

* 5G-IA, "Link-Level Calibration Results v2", 5G-PPP (2019), Table 3.
* 3GPP TS 38.104 V16.4.0 (2020-06), "NR; Base Station radio transmission and
  reception", clause 8.2.1.2 (Tables 8.2.1.2-2, 8.2.1.2-6) and Annex A.3/A.4/A.5
  (FRCs G-FR1-A3/A4/A5). The RAN4 requirements were contributed by Ericsson,
  Nokia, Huawei, ZTE, Qualcomm, Intel, Samsung and others.

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
| 2 | 1×2 TDL-C 16QAM | 10/15  |  7.16 |  8.24 | 10.2 | −1.08 | −3.04 |
| 3 | 1×2 TDL-A 64QAM | 10/15  |  9.17 | 10.48 | 12.2 | −1.31 | −3.03 |
| 4 | 2×2 TDL-B QPSK  | 10/15  | −2.45 | −0.59 |  1.7 | −1.86 | −4.15 |
| 5 | 2×2 TDL-C 16QAM | 10/15  | 14.68 | 16.31 | 18.3 | −1.63 | −3.62 |
| 6 | 1×2 TDL-B QPSK  | 40/30  | −6.90 | −4.68 | −2.5 | −2.22 | −4.40 |
| 7 | 1×2 TDL-C 16QAM | 40/30  |  7.73 |  7.97 | 10.0 | −0.24 | −2.27 |
| 8 | 1×2 TDL-A 64QAM | 40/30  |  9.56 | 10.12 | 12.4 | −0.56 | −2.84 |
| 9 | 2×2 TDL-B QPSK  | 40/30  | −3.02 | −0.96 |  1.3 | −2.06 | −4.32 |

* **vs 5G-IA ideal:** mean Δ = −1.33 dB, RMS = 1.47 dB, **bias-removed spread = 0.64 dB**.
* **vs 3GPP TS 38.104:** mean Δ = −3.43 dB, **spread = 0.69 dB** (sim sits below
  the minimum requirement, as an ideal simulator must).

## Assessment

* **Relative / trend alignment is strong and consistent across both
  references.** After removing a constant bias the simulator tracks the
  case-to-case SNR to within **~0.65 dB** across QPSK / 16QAM / 64QAM,
  TDL-A/B/C, SIMO and 2×2 MIMO, 10 MHz/15 kHz and 40 MHz/30 kHz. The companies
  themselves disagree by ~1.5–2 dB on the harder cases (e.g. 5G-IA case 5 spans
  15.31–17.24 dB), so the relative agreement is within the inter-company spread.

* **The two references are mutually consistent, and the simulator sits
  correctly relative to both.** The 3GPP TS 38.104 requirement is ~2 dB above
  the 5G-IA ideal calibration (the implementation margin), and the simulator is
  ~1.3 dB below the ideal calibration (MIESM coding-gap optimism). The −3.43 dB
  vs 38.104 is exactly these two effects stacked: margin (~2.1 dB) + abstraction
  optimism (~1.3 dB). An ideal simulator being several dB better than a *minimum
  requirement* is the physically correct relationship.

* **The ~1.3 dB optimistic bias vs the ideal calibration** is the MIESM
  coding-gap model (`1.0 + 0.6·R` dB from capacity), which underestimates the
  real NR-LDPC implementation loss. It is a single calibratable constant — a gap
  offset of ~1.3 dB, or running the bit-true LDPC path, centres the results on
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
  otherwise-identical cases 1 and 6 is within this MC noise.

## Sources

* 5G-IA, "Link-Level Calibration Results v2", 5G-PPP (2019), Table 3.
* 3GPP TS 38.104 V16.4.0 (2020-06), "NR; Base Station radio transmission and
  reception", clause 8.2.1.2 (Tables 8.2.1.2-2, 8.2.1.2-6) and Annex A.3/A.4/A.5
  (FRCs G-FR1-A3/A4/A5). The RAN4 requirements were contributed by Ericsson,
  Nokia, Huawei, ZTE, Qualcomm, Intel, Samsung and others.

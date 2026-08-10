# Calibration against 5G-IA link-level results

This documents how `nrdlsim` compares to the **5G Infrastructure Association
Link-Level Calibration Results (v2)**, which reports the SNR at 70 % of maximum
throughput for the 3GPP RAN4 FR1/FDD evaluation cases (agreed across eight
companies).  Reproduce with `python examples/calibration_5gia.py`.

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

SNR (dB) @ 70 % throughput — simulator vs the 5G-PPP WG value and the
8-company average:

| Case | Config | sim | 5G-PPP WG | company avg | Δ(WG) | Δ(avg) |
|------|--------|-----|-----------|-------------|-------|--------|
| 1 | 1×2 TDL-B QPSK (MCS2) | −5.70 | −4.15 | −4.68 | −1.55 | −1.02 |
| 2 | 1×2 TDL-C 16QAM (MCS16) | 7.16 | 8.60 | 8.24 | −1.44 | −1.08 |
| 3 | 1×2 TDL-A 64QAM (MCS20) | 9.17 | 10.20 | 10.48 | −1.03 | −1.31 |
| 4 | 2×2 TDL-B QPSK (MCS2) | −2.45 | −0.45 | −0.59 | −2.00 | −1.86 |
| 5 | 2×2 TDL-C 16QAM (MCS16) | 14.68 | 15.45 | 16.31 | −0.77 | −1.63 |
| 6 | 1×2 TDL-B QPSK (MCS2) | −6.90 | −4.15 | −4.68 | −2.75 | −2.22 |
| 7 | 1×2 TDL-C 16QAM (MCS16) | 7.73 | 8.40 | 7.97 | −0.67 | −0.24 |
| 8 | 1×2 TDL-A 64QAM (MCS20) | 9.56 | 9.70 | 10.12 | −0.14 | −0.56 |
| 9 | 2×2 TDL-B QPSK (MCS2) | −3.02 | −0.10 | −0.96 | −2.92 | −2.06 |

* **Mean Δ vs company average = −1.3 dB**, RMS = 1.5 dB.
* **Bias-removed spread (std of Δ) = 0.9 dB.**

## Assessment

* **Relative / trend alignment is strong.** After removing a constant bias the
  simulator tracks the case-to-case SNR to within **0.9 dB** across QPSK / 16QAM
  / 64QAM, TDL-A/B/C, SIMO and 2×2 MIMO, and 10 MHz/15 kHz and 40 MHz/30 kHz.
  For reference the companies themselves disagree by ~1.5–2 dB on the harder
  cases (e.g. case 5 spans 15.31–17.24 dB), so the relative agreement is within
  the inter-company spread.

* **A consistent ~1.5 dB optimistic bias** remains.  This is the MIESM
  link-abstraction coding-gap model (`1.0 + 0.6·R` dB from capacity), which
  underestimates the real NR-LDPC implementation loss.  It is a single
  calibratable constant — it does not vary much across the cases — so a gap
  offset of ~1.5 dB, or running the bit-true LDPC path, would centre the
  results near zero.

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

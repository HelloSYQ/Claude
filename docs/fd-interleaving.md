# Frequency-domain interleaving: 32T4R, CDL-A, DS 30 ns, 32 PRB

This study compares two frequency-domain (FD), VRB-to-PRB, interleavers for a PDSCH that uses the whole 32-PRB bandwidth. It uses rank adaptation and the NR codeword-to-layer mapping.

| Mapping | VRB-to-PRB mapping |
|---|---|
| no interleaving (reference) | PRB = VRB |
| **FD interleaving 1**: NR | TS 38.211 7.3.1.6 interleaved mapping with bundle size L = 4 and R = 2. VRB bundle j = 2c + r goes to PRB bundle 4r + c: VRBs 0–3 → PRBs 0–3, 4–7 → 16–19, 8–11 → 4–7, … |
| **FD interleaving 2**: 1 PRB, 4 × 8 | VRBs are written row by row into a 4 × 8 matrix and read column by column: VRB v → PRB 4 (v mod 8) + ⌊v/8⌋. VRBs 0–7 → PRBs 0, 4, …, 28; VRB 8 → PRB 1 |
| ideal interleaving (reference) | every code block sees the slot-average mutual information |

L = 4 is the only NR choice here: TS 38.211 does not allow L = 2 together with a PRG size of 4.

```bash
python examples/fd_interleaving.py                   # 1000 drops, ~1 min on 4 cores
python examples/fd_interleaving_plot.py             # re-plot results/fd_interleaving.json
python examples/fd_interleaving.py --ds 300 --drops 400 --no-plot   # cross-check
python -m pytest tests/test_fd_interleaving.py
```

Files:

- `examples/fd_interleaving.py`: the simulation, which writes `results/fd_interleaving.json`
- `examples/fd_interleaving_plot.py`: the figure `results/fd_interleaving.png`
- `nrdlsim/re_mapping.py`: the VRB-to-PRB mappings and the RE mapping of code blocks
- `nrdlsim/ldpc.py`: `segment_transport_block` and `rate_match_lengths` (TS 38.212 5.2.2 / 5.4.2.1)
- `tests/test_fd_interleaving.py`

## Result

**With this set-up, FD interleaving brings no measurable gain.** Both interleavers improve spectral efficiency (SE) by at most 0.05 % with the simulator's LDPC waterfall, and by 0.18 % with a 4× steeper waterfall. At 10 % TB BLER they gain at most 0.06 dB. The ideal interleaver gains no more: there is almost nothing for any interleaver to recover. The link adaptation makes the same choices for every mapping: average rank within 0.005 and average MCS within 0.06.

![FD interleaving results](../results/fd_interleaving.png)

### Why the gain is so small

Interleaving matters only when the code blocks (CBs) of a transport block (TB) see different SINRs. Two things keep them alike here:

1. **The beamformed channel is nearly flat over 32 PRBs.** The raw per-element channel |h|² of CDL-A at 30 ns varies by a median of 8 dB (7–10 dB, 10th–90th percentile) across the 32 PRBs. After per-PRG SVD precoding over 32 ports and the 4-Rx LMMSE receiver, the per-PRB SINR of each layer varies by a median of only 0.5, 0.7, 1.2 and 1.5 dB for layers 1–4 (90th percentile 0.9, 1.2, 2.1, 2.6 dB). These figures are at rank 4 and 25 dB over 200 drops.
2. **Each CB covers most of the band and all layers.** Layer mapping deals each CB's symbols out over all v layers, so layer imbalance is averaged inside every CB. The CBs are also long compared with the band. Even at rank 4, MCS 27 (TBS 125016, 15 CBs, the most CBs in this set-up), each CB covers 281–282 REs per layer, i.e. 0.73 OFDM symbol or about 23.5 of the 32 PRBs. At lower ranks a CB covers more than a whole OFDM symbol:

| Rank (MCS 27) | CBs per TB | REs per layer per CB | OFDM symbols per CB |
|---|---|---|---|
| 1 | 4 | 1056 | 2.75 |
| 2 | 8 | 528 | 1.38 |
| 3 | 12 | 352 | 0.92 |
| 4 | 15 | 281 | 0.73 |

Panels (e)–(g) of the figure show where each CB lands at rank 4, MCS 27. Without interleaving, each CB occupies one contiguous run of PRBs. The NR interleaver moves runs of 4 PRBs (one PRG) to the two halves of the band. The 4 × 8 interleaver spreads each CB over almost every PRG.

The spread (max − min) of the 15 CB effective SINRs shows how well each interleaver equalises the CBs. The table gives the drop average at rank 4, MCS 27; it is panel (c) of the figure:

| SNR | no interleaving | FD interleaving 1 (NR) | FD interleaving 2 (4 × 8) |
|---|---|---|---|
| 10 dB | 0.16 dB | 0.09 dB | 0.03 dB |
| 20 dB | 0.25 dB | 0.14 dB | 0.05 dB |
| 25 dB | 0.34 dB | 0.19 dB | 0.05 dB |

The 4 × 8 interleaver brings the CBs within 0.05 dB of each other and the NR interleaver halves the spread. Since the spread is only about 0.3 dB to begin with, neither changes the TB BLER enough to matter.

### Spectral efficiency (ideal AMC, rank adaptation)

The first data row is the SE without interleaving. The other rows are SE gains over no interleaving in %, with the 95 % bootstrap confidence interval over the 1000 drops (paired) in brackets.

Simulator waterfall (x1):

| | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| SE, no interleaving (b/s/Hz) | 5.14 | 8.25 | 11.59 | 15.63 | 19.23 | 21.60 | 21.70 |
| FD interleaving 1 (NR) | +0.01 (0.00..0.01) | +0.02 (0.01..0.02) | +0.01 (0.01..0.01) | +0.00 | +0.03 (0.02..0.03) | +0.01 (0.00..0.01) | 0.00 |
| FD interleaving 2 (4 × 8) | +0.01 (0.01..0.01) | +0.02 (0.02..0.02) | +0.01 (0.01..0.02) | +0.00 | +0.04 (0.03..0.04) | +0.01 (0.01..0.01) | 0.00 |
| ideal interleaving | +0.01 (0.01..0.02) | +0.02 (0.02..0.03) | +0.02 (0.01..0.02) | +0.00 | +0.03 (0.02..0.04) | +0.00 (−0.00..0.01) | 0.00 |

Steep waterfall (x4):

| | 0 dB | 5 dB | 10 dB | 15 dB | 20 dB | 25 dB | 30 dB |
|---|---|---|---|---|---|---|---|
| SE, no interleaving (b/s/Hz) | 5.86 | 9.31 | 12.92 | 17.24 | 20.77 | 21.70 | 21.70 |
| FD interleaving 1 (NR) | +0.02 (0.01..0.02) | +0.05 (0.04..0.06) | +0.02 (0.02..0.03) | +0.01 (0.01..0.01) | +0.13 (0.11..0.16) | 0.00 | 0.00 |
| FD interleaving 2 (4 × 8) | +0.02 (0.02..0.03) | +0.06 (0.05..0.08) | +0.03 (0.02..0.04) | +0.01 (0.01..0.02) | +0.18 (0.15..0.21) | 0.00 | 0.00 |
| ideal interleaving | +0.02 (0.02..0.03) | +0.07 (0.05..0.09) | +0.03 (0.03..0.04) | +0.02 (0.02..0.03) | +0.18 (0.15..0.22) | 0.00 | 0.00 |

- **Rank adaptation.** The average rank is 3.5 at 0 dB (48 % rank 3, 52 % rank 4) and 4.0 from 5 dB on, for every mapping. The 32-port array makes the 4th eigenmode strong enough for rank 4 almost everywhere.
- **Where the gain peaks.** The gains are largest around 20 dB, where the AMC uses MCS 24–26 at rank 4 with 14–15 CBs. From about 25 dB the TB sits at MCS 27 with near-zero BLER, so there is nothing left to gain.
- **The 4 × 8 interleaver.** At every SNR it is as good as NR or slightly better, and it essentially reaches the ideal reference. The difference from NR is at most 0.01 % of SE (x1) or 0.05 % (x4).

### TB BLER at fixed rank and MCS

The table gives the SNR (dB) at 10 % TB BLER, mean over drops, without interleaving, and how much lower (dB) the other mappings reach it.

| Rank, MCS (CBs) | waterfall | no interleaving | FD interleaving 1 (NR) | FD interleaving 2 (4 × 8) | ideal |
|---|---|---|---|---|---|
| 4, MCS 19 (11) | x1 | 14.83 | 0.00 | 0.00 | 0.00 |
| 4, MCS 23 (13) | x1 | 17.64 | 0.00 | 0.00 | 0.00 |
| 4, MCS 27 (15) | x1 | 23.23 | 0.02 | 0.03 | 0.03 |
| 4, MCS 19 (11) | x4 | 13.62 | 0.00 | 0.00 | 0.00 |
| 4, MCS 23 (13) | x4 | 16.32 | 0.01 | 0.02 | 0.02 |
| 4, MCS 27 (15) | x4 | 21.83 | 0.05 | 0.06 | 0.06 |

### Cross-check: a more frequency-selective channel

The set-up is unchanged except for the delay spread (400 drops). This check confirms that the model does see interleaving gain when the beamformed channel is selective.

| DS | Max SE gain, NR / 4 × 8 / ideal (x1) | Max SE gain, NR / 4 × 8 / ideal (x4) | 10 %-BLER gain at rank 4, MCS 27, NR / 4 × 8 / ideal (x4) |
|---|---|---|---|
| 30 ns (requested) | 0.04 / 0.05 / 0.05 % | 0.13 / 0.18 / 0.18 % | 0.05 / 0.06 / 0.06 dB |
| 100 ns | 0.03 / 0.03 / 0.04 % | 0.08 / 0.10 / 0.19 % | 0.01 / 0.00 / 0.04 dB |
| 300 ns | 0.15 / 0.11 / 0.36 % | 0.42 / 0.38 / 0.75 % | 0.03 / 0.28 / 0.89 dB |

At 300 ns, ideal interleaving gains up to 0.9 dB at rank 4, MCS 27. Neither interleaver gets close to that, and the 4 × 8 interleaver then shows a side effect of its geometry:

- **The 4 × 8 row index is the PRB's position inside its PRG.** VRBs 0–7 land on the first PRB of every PRG, VRBs 8–15 on the second, and so on. The NR interleaver keeps whole PRGs together.
- **Position inside the PRG matters once the channel varies within a PRG.** The per-PRG SVD precoder fits the PRG centre better than its edges. At 300 ns the edge PRBs (positions 0 and 3) are 1.5 dB worse than the centre PRBs (positions 1 and 2). A CB made of a few rows of the 4 × 8 matrix therefore sits mostly on edge PRBs, or mostly on centre PRBs.
- **At 30 ns the effect is negligible.** The edge–centre difference is 0.05 dB, so it does not affect the requested set-up.

## Model

| Item | Assumption |
|---|---|
| Carrier | 3.5 GHz, 30 kHz SCS, 32 PRB (11.52 MHz); the whole BWP is allocated |
| Channel | CDL-A, DS 30 ns, 3 km/h; one snapshot per drop (the channel is constant over the slot); 1000 drops |
| gNB | 32 ports, (M, N, P) = (2, 8, 2), ±45° cross-polar, 0.5 λ spacing, omni elements |
| UE | 4 ports, (1, 2, 2), ±45° cross-polar, 0.5 λ spacing, omni elements |
| Precoding | SVD per PRG of 4 PRBs: the strongest eigenvectors of Σ_PRG HᴴH, ideal CSI; equal power per layer |
| Receiver | LMMSE per subcarrier, ideal channel estimation |
| CW-to-layer mapping | NR: one codeword for ranks 1–4; symbol i goes to layer i mod v (TS 38.211 7.3.1.3) |
| PDSCH | 13 symbols, 2 DM-RS symbols (type 1, 2 CDM groups without data): 132 RE/PRB, 4224 RE per layer |
| Coding | TBS per TS 38.214 5.1.3.2, MCS Table 2. LDPC base graph, TB CRC and code-block segmentation per TS 38.212 7.2.2 / 5.2.2; E_r per CB per 5.4.2.1 |
| RE mapping | each layer frequency first over the VRBs, then symbol by symbol (TS 38.211 7.3.1.5), then VRB-to-PRB (7.3.1.6) |
| CB BLER | MIESM over the CB's REs, then the simulator's LDPC waterfall at the CB code rate K′/E_r and length K′ |
| TB BLER | 1 − Π (1 − BLER_CB) |
| Link adaptation | ideal AMC per drop and SNR: the rank (1–4) and MCS with the most expected bits TBS (1 − BLER_TB), chosen separately for each mapping |

The NR interleaver uses C = ⌊N_bundle / 2⌋ columns, which makes the mapping a permutation for every BWP size and offset (see the tests). For 32 PRB and L = 4 there are 8 bundles, so C = 4 whichever rounding is used.

### Waterfall slope

How much the worst CB dominates the TB depends on how steep each CB's waterfall is.

- **x1 (all main results).** The simulator's waterfall (`link_abstraction._waterfall`), as in every other study in this repo. It is shallow: about 0.8 dB per BLER decade at K′ = 8448.
- **x4 (sensitivity check).** A 4× steeper slope with the same 50 % point, about 0.2 dB per decade at K′ = 8448. This is closer to an AWGN LDPC curve.

The conclusion holds for both.

## Limitations

- **Ideal CSI, ideal channel estimation, no CSI delay.** Precoder ageing or a codebook (Type I/II) precoder would leave more frequency selectivity in the effective channel, so these results lean towards the low side of the interleaving gain.
- **Ideal AMC.** There is no CQI mismatch, OLLA or HARQ. With a wideband CQI that cannot see CB-level effects, OLLA would absorb the small per-CB SINR differences as a back-off of a similar (tiny) size.
- **Full allocation only.** With a partial allocation inside a larger BWP, interleaving also spreads the allocated PRBs over the band, which is a different and usually larger effect.
- **Static channel within the slot.** At 3 km/h the channel does not change within the 0.5 ms slot, so CBs differ only in frequency.

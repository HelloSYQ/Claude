# NR Downlink Link-Level Simulation — requirements checklist

What a *well-functioning* downlink (PDSCH) link-level simulator (LLS) must
resolve, and the parameters it must expose.  Use this as a design/coverage
checklist.  The final column marks how `nrdlsim` currently stands:
✅ implemented · ◐ partial / abstracted · ❌ not yet.

---

## 1. Granularity — the resolution the LLS must reach

| Axis | Minimum bar (must have) | Reference-grade (full fidelity) | nrdlsim |
|------|-------------------------|---------------------------------|---------|
| **Frequency** | per-resource-element signal processing; channel at least per-RB | per-subcarrier channel response | ◐ RE proc, per-RB channel |
| **Time** | per-OFDM-symbol, with intra-slot channel (Doppler) evolution | **time-domain sample waveform**: IFFT/CP/FFT, oversampling, sync | ✅ waveform module (IFFT/CP/FFT) + ◐ slot-level SE loop |
| **Coding** | code-block-level, bit-true FEC (or calibrated L2S abstraction) | bit-true always, per-RV soft HARQ buffer | ✅ bit-true LDPC + MIESM |
| **Space** | per-antenna-port and per-layer | per-element with array geometry & coupling | ✅ geometric CDL panels |
| **Statistics** | enough MC blocks for ~100 block errors at target BLER | confidence intervals reported | ◐ configurable depth |

**The bar in one line:** resource-element granularity in frequency,
OFDM-symbol granularity in time (with Doppler within the slot), and
code-block granularity in the FEC.  The step up to a **time-domain waveform**
(samples, CP, sync) is what lets you model synchronisation, CP overrun and RF
impairments.

---

## 2. Essential parameters, by category

### A. Carrier / numerology
- [x] Subcarrier spacing (μ → 15·2^μ kHz)
- [x] Bandwidth / N_RB (BWP size)
- [x] Cyclic prefix (normal/extended)
- [x] Carrier frequency (Doppler, phase noise)
- [x] FFT size & sample rate (waveform path, `nrdlsim/ofdm.py`)
- [ ] Duplex mode + TDD DL/UL slot pattern

### B. PDSCH resource allocation
- [x] RB allocation (start, length)
- [x] Symbol allocation + PDSCH mapping type A/B
- [x] DM-RS: type 1/2, additional positions, CDM groups, front-load
- [ ] PT-RS (phase tracking, FR2)
- [x] xOverhead
- [ ] Rate-matching around SSB / CORESET / CSI-RS
- [ ] VRB→PRB mapping / PRB bundling

### C. MCS & coding
- [x] MCS table + index → (Qm, target code rate)
- [x] Transport block size (TBS)
- [x] LDPC base graph + lifting size, code-block segmentation
- [x] CRC (24A / 24B / 16)
- [x] Rate matching (◐ RV sequence / HARQ redundancy versions)
- [ ] Scrambling (nRNTI, nID)
- [ ] Limited-buffer rate matching (LBRM)

### D. MIMO / precoding / antennas
- [x] Number of layers / rank, codeword→layer mapping
- [x] Precoding (SVD / reciprocity; ◐ Type-I/II codebook + PMI)
- [x] Antenna ports & panel geometry (UPA, polarization)
- [x] Antenna element radiation pattern (TR 38.901 Table 7.3-1)

### E. Channel
- [x] Model: TDL-A…E, CDL-A…E, AWGN
- [x] Delay spread
- [x] Doppler / UE speed
- [x] K-factor (LOS), XPR, spatial correlation/geometry
- [x] Number of channel realizations (fading seeds)
- [ ] Spatial consistency / mobility tracks, O2I penetration

### F. Receiver
- [x] Channel estimation (◐ practical DM-RS error model)
- [x] Equalizer (MMSE MIMO)
- [x] Soft demapper (max-log LLR)
- [x] Timing / frequency synchronisation offsets (waveform path; SE loop ideal)
- [ ] Noise-variance estimation error

### G. Impairments (rigorous vs cartoon)
- [x] SNR definition stated (Es/N0 per RE)
- [x] AWGN
- [x] Carrier frequency offset (CFO) — waveform path (`ofdm.py`)
- [x] Timing offset — waveform path (`ofdm.py`)
- [ ] Phase noise (FR2)
- [ ] PA nonlinearity / EVM / clipping
- [ ] IQ imbalance, ADC quantization
- [ ] Inter-cell interference

### H. HARQ / link adaptation / procedures
- [x] HARQ: max retransmissions, combining (◐ chase; IR = future)
- [x] CSI feedback: RI / PMI / CQI with feedback delay
- [ ] CSI quantization (subband, codebook granularity)
- [x] Outer-loop link adaptation (OLLA)
- [x] BLER target

### I. Simulation control
- [x] SNR range / points
- [x] Blocks / slots per SNR point (MC depth)
- [x] Random seed
- [x] Metrics: BLER, throughput, spectral efficiency

---

## 3. Priority gaps to reference-grade

1. ~~Time-domain OFDM waveform (IFFT/CP/FFT) with CFO and timing offset~~ —
   **done** in `nrdlsim/ofdm.py`; next: fold the waveform path into the main
   SE loop so BLER curves include CFO/timing, not just EVM.
2. **RF impairments**: phase noise (FR2), PA nonlinearity/EVM.
3. **IR-HARQ** with a real soft (LLR) combining buffer and RV sequence.
4. **CSI quantization** (Type-I/II codebook PMI, subband CQI).
5. **Interference** modelling (inter-cell / multi-user).

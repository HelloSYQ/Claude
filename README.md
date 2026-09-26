# NR Downlink Link-Level Simulator (`nrdlsim`)

A modular **5G NR (New Radio) downlink link-level simulator** written in Python.
It models the PDSCH transmit/receive chain end to end and outputs the achieved
**spectral efficiency** (bits/s/Hz) versus SNR, together with BLER, throughput,
and the adapted MCS/rank/CQI statistics.

The implementation follows the relevant 3GPP specifications:

| Area | Specification |
|------|---------------|
| Numerology, resource grid, DM-RS, modulation | TS 38.211 |
| DL-SCH coding (CRC, segmentation, LDPC, rate matching) | TS 38.212 |
| MCS/TBS, CSI (CQI/PMI/RI), link adaptation | TS 38.214 |
| Channel models (TDL-A…E) | TR 38.901 |

---

## Modules

The simulator is decomposed into the modules the task asked for, each a
self-contained file under `nrdlsim/`:

| Module | Responsibility | Spec |
|--------|----------------|------|
| `config.py` | Carrier / PDSCH / antenna / channel / HARQ configuration, numerology (SCS = 15·2^μ kHz), slot timing, bandwidth | TS 38.211 §4 |
| `mcs_tables.py` | **MCS scheduling** tables 1/2/3 and CQI tables 1/2; CQI→MCS mapping | TS 38.214 §5.1.3.1, §5.2.2.1 |
| `tbs.py` | Transport block size (≤3824-bit quantisation table + >3824 formula) | TS 38.214 §5.1.3.2 |
| `resource_grid.py` | **Resource grid** + **DM-RS** RE accounting, data-RE / overhead computation | TS 38.211 §7.3.1.6, §7.4.1.1 |
| `modulation.py` | QPSK/16/64/256-QAM Gray mapping and max-log **soft (LLR) demapping** | TS 38.211 §5.1 |
| `ofdm.py` | **Time-domain CP-OFDM waveform** (IFFT + cyclic prefix / FFT), RF impairments (**CFO, timing offset**), multipath, EVM measurement | TS 38.211 §5.3 |
| `channel_models.py` | **NR channel generation**: TDL-A…E (Kronecker-correlated MIMO) and CDL-A…E (per-cluster AoD/AoA/ZoD/ZoA, intra-cluster rays, **dual-polarized UPA panels** with cross-pol XPR coupling, **directional element gain pattern**, Ricean LOS for CDL-D/E); configurable delay spread & Doppler | TR 38.901 §7.3, §7.5, §7.7 |
| `layer_mapping.py` | **Layer mapping** (codeword→layer, 1–8 layers, dual-codeword for rank ≥5) and SVD/codebook **precoding** | TS 38.211 §7.3.1.3 |
| `ldpc.py` | **DL-SCH coding**: CRC (24A/24B/16), code-block segmentation, base-graph (BG1/BG2) selection, QC-LDPC encode, normalised min-sum decode, rate matching | TS 38.212 §5, §7.2 |
| `link_abstraction.py` | BICM capacity, MIESM effective-SINR compression, NR-LDPC BLER model | — |
| `csi.py` | **CSI feedback**: rank (RI), precoder (PMI), CQI selection with BLER target and configurable feedback **delay** | TS 38.214 §5.2 |
| `scheduler.py` | **Resource + MCS scheduling** with outer-loop link adaptation (OLLA) | TS 38.214 |
| `receiver.py` | DM-RS channel estimation, **MMSE MIMO equalisation**, per-RE post-equaliser SINR | — |
| `link_simulator.py` | Per-slot chain integration, HARQ (chase combining), **spectral-efficiency** accounting | — |

### End-to-end per-slot chain

```
 TR 38.901 TDL channel ─▶ DM-RS channel estimation
        └─▶ CSI feedback (RI/PMI/CQI, delayed N slots)
                └─▶ scheduler (resource alloc + MCS link adaptation, OLLA)
                        └─▶ TBS / DL-SCH LDPC coding
                                └─▶ layer mapping + SVD precoding
                                        └─▶ MIMO transmit + AWGN
                                                └─▶ MMSE equalisation
                                                        └─▶ demap / decode
                                                                └─▶ HARQ
                                                                        └─▶ SE
```

---

## Two FEC / performance modes

Because decoding every code block over a full SNR sweep is expensive, the
simulator offers two paths, selected by `fec_mode`:

* **`miesm` (default)** — a standard *link-to-system abstraction*. Each
  resource element's post-equaliser SINR is mapped to its BICM mutual
  information, the values are compressed to an effective SINR (MIESM), and an
  NR-LDPC BLER waterfall (calibrated to capacity + coding gap, sharpening with
  block length) decides block success. This produces smooth spectral-efficiency
  curves quickly.

* **`ldpc`** — a **bit-true** QC-LDPC path: real CRC, encoding, QAM, per-RE
  channel + noise, MMSE equalisation, LLR demapping and min-sum decoding of a
  representative code block. Slower; intended for validation on small
  configurations. The self-tests confirm the decoder achieves real coding gain
  (100 % decode at +6 dB, 0 % at −6 dB on QPSK).

---

## Guided tour (Jupyter notebook)

For a comprehensive, runnable walkthrough of **every module** — with a live demo
and plot for each (constellations, channel PDP/frequency response, CDL angular
map and antenna pattern, BICM capacity, OFDM CFO/timing, a full SE curve, and
the calibration) — open:

**[`notebooks/NR_Downlink_Simulator_Guide.ipynb`](notebooks/NR_Downlink_Simulator_Guide.ipynb)**

It renders with outputs directly on GitHub. To re-execute it locally:

```bash
jupyter nbconvert --to notebook --execute --inplace \
    notebooks/NR_Downlink_Simulator_Guide.ipynb
# or regenerate from source: python notebooks/build_guide.py
```

## Installation

```bash
python3 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
```

`nrdlsim` is also an installable package (`pyproject.toml`; runtime
dependency: NumPy only). Other projects can pin it straight from git:

```bash
pip install "nrdlsim @ git+https://github.com/HelloSYQ/Claude.git@<commit>"
```

## Usage

```bash
# Default: 4x2 MIMO, TDL-C, 256QAM, link adaptation + HARQ, SE curve
python run_simulation.py --plot

# AWGN SISO baseline
python run_simulation.py --model AWGN --ntx 1 --nrx 1

# CDL clustered-delay-line channel (geometric MIMO)
python run_simulation.py --model CDL-C --ds 300 --plot
python run_simulation.py --model CDL-D --doppler 50    # LOS profile

# CDL with a dual-polarized UPA: 8 tx ports = 2x2 positions x 2 pol, rank 4
python run_simulation.py --model CDL-C --ntx 8 --nrx 4 \
    --tx-pol 2 --rx-pol 2 --tx-layout 2x2 --rx-layout 1x2 --layers 4

# ...with the 38.901 directional element pattern and 8 deg downtilt
python run_simulation.py --model CDL-C --ntx 8 --nrx 4 \
    --tx-pol 2 --rx-pol 2 --tx-layout 2x2 --rx-layout 1x2 --layers 4 \
    --tx-pattern 38.901 --downtilt 8 --element-gain 8

# Fixed MCS (no link adaptation), higher Doppler
python run_simulation.py --fixed-mcs --mcs 20 --doppler 300

# Bit-true LDPC validation on a small config
python run_simulation.py --fec ldpc --snr 0 12 3 --slots 30 --ntx 2 --nrx 2

# Compare channel models and MIMO orders (writes results/compare_se.png)
python examples/compare_configs.py

# Time-domain OFDM waveform: EVM vs CFO and timing offset
python examples/waveform_demo.py

# CDL power-delay-profile verification vs TR 38.901
# (writes results/cdl_pdp_verification.png, ~1 min)
python examples/verify_cdl_pdp.py
```

### Key CLI options

| Option | Meaning | Default |
|--------|---------|---------|
| `--mu` | numerology (SCS = 15·2^μ kHz) | 1 (30 kHz) |
| `--rb` | resource blocks (bandwidth) | 51 |
| `--ntx / --nrx` | gNB / UE antennas | 4 / 2 |
| `--model` | `TDL-A…E`, `CDL-A…E` or `AWGN` | TDL-C |
| `--ds / --doppler` | delay spread (ns) / max Doppler (Hz) | 100 / 100 |
| `--mcs-table` | 1 (64QAM), 2 (256QAM), 3 (low-SE), 4 (256QAM + 1024QAM, illustrative) | 2 |
| `--tx-pol / --rx-pol` | CDL polarizations per position (1, or 2 for cross-polar ±45°) | 1 |
| `--tx-layout / --rx-layout` | CDL panel `VxH` position grid (e.g. `2x2`); UPA when V>1 | auto (1×N) |
| `--spacing-v / --spacing-h` | CDL element spacing (wavelengths) | 0.5 |
| `--tx-pattern / --rx-pattern` | CDL element pattern: `omni` or `38.901` directional | omni |
| `--downtilt / --boresight-az` | CDL tx mechanical downtilt / boresight azimuth (deg) | 0 |
| `--element-gain` | CDL max element gain G_E,max (dBi) | 8.0 |
| `--snr` | START STOP STEP (dB) | −5 30 2.5 |
| `--slots` | slots simulated per SNR point | 200 |
| `--csi-delay` | CSI report delay (slots) | 4 |
| `--fec` | `miesm` or `ldpc` | miesm |

---

## Example output

```
 SNR[dB]   SE[b/s/Hz]   Tput[Mbps]  BLER1st  resBLER   avgMCS  avgRank   avgCQI
   -5.00       0.3213        5.900    0.135    0.006     2.02     1.02     3.38
    0.00       0.7605       13.963    0.132    0.006     4.44     1.02     4.96
   10.00       2.3761       43.626    0.152    0.000     7.15     2.00     6.73
   20.00       5.0943       93.531    0.156    0.000    15.61     2.00    11.52
   30.00       8.2198      150.916    0.126    0.000    23.82     2.00    14.63
```

Observed behaviour (all physically consistent):

* SE rises with SNR and saturates near the rank·log₂(M)·R ceiling.
* AWGN > TDL fading; SE scales with MIMO rank (1×1 ≈ 4, 2×2 ≈ 8, 4×4 ≈ 16 b/s/Hz).
* Rank adaptation uses rank 1 at low SNR and switches to rank 2 as SNR rises.
* `BLER1st` is the first-transmission BLER, which OLLA drives to the 0.1
  target. In a 200-slot run it still includes OLLA's settling period (≈0.13);
  over 3000 slots it converges to ≈0.10. `resBLER` is the fraction of transport
  blocks still lost after all HARQ retransmissions.

The **spectral efficiency** is computed over the allocated bandwidth:

```
SE = delivered_information_bits / (num_slots · slot_duration) / (num_rb · 12 · SCS)
```

### Link adaptation in the library path

1. The UE evaluates every rank with the power split the gNB will use
   (`W/√rank`), and reports the highest CQI whose effective SINR meets the
   BLER target on the CSI reference resource.
2. The gNB maps the CQI back to that SINR, subtracts the OLLA offset (dB), and
   picks the highest MCS that meets the target at that SINR.
3. OLLA is updated only on first transmissions: +0.5 dB after a NACK and
   −0.5·p/(1−p) dB after an ACK. It can move in either direction.
4. A HARQ retransmission keeps the transport block's rank, MCS and TBS, and is
   chase-combined with the earlier attempts.

## Calibration against 3GPP / 5G-IA

`examples/calibration_5gia.py` cross-checks the simulator (SNR at 70 %
throughput, 3GPP RAN4 FR1/FDD) against **two independent published references**:
the **5G-IA link-level calibration** (ideal, 8-company average) and the **3GPP
TS 38.104** PUSCH minimum performance requirements (FRCs G-FR1-A3/A4/A5, which
map exactly onto MCS 2/16/20). The simulator tracks case-to-case behaviour to
within **~0.65 dB** (bias-removed) across QPSK/16QAM/64QAM, TDL-A/B/C, SIMO and
2×2 MIMO, and two bandwidths — sitting ~1.3 dB below the ideal calibration
(MIESM coding-gap optimism) and ~3.4 dB below the conformance requirement (that
gap = coding-gap optimism + the ~2 dB implementation margin the requirement
builds in, exactly as an ideal simulator should). See
[`docs/calibration.md`](docs/calibration.md) for the full table and analysis.
This exercise also surfaced and fixed a transmit-power normalization bug (each
layer had received full power; now total power is split across layers,
SNR = total Es/N0).

## Tests

```bash
python tests/test_modules.py
python tests/test_link_adaptation.py
python tests/test_cdl_pdp.py
python tests/test_antenna_orientation.py
```

`test_modules.py` validates constellation energy, noiseless modulation
round-trip, MCS/TBS, TDL fading power normalisation, monotone BICM capacity,
and genuine LDPC coding gain.

`test_link_adaptation.py` holds regression tests for the link-adaptation path.
They check that:

- the CSI SINR equals the SINR of what is actually transmitted
- the CQI/MCS thresholds invert the BLER waterfall
- OLLA moves in both directions and has zero drift at the target
- first-transmission BLER converges to the target
- the config is not mutated, and serial and parallel sweeps give identical results
- SE is computed over the allocated bandwidth
- TBS quantisation follows the spec (`floor`, round-half-up)
- MCS table 4 runs under link adaptation
- HARQ accounting is correct (initial vs residual BLER)

`test_cdl_pdp.py` checks that the CDL generator reproduces the TR 38.901 power
delay profile. It checks that:

- the tables' delays, powers and LOS power equal the TR 38.901 values shipped
  with NVIDIA Sionna. This check is skipped unless Sionna's data files are
  present (`pip install --no-deps sionna`).
- each table is normalised to unit RMS delay spread, and the CDL-D/E K-factors
  are 13.3 / 22.0 dB
- every cluster's power, recovered from the generated H(f) by least squares on
  the table delays, is within 0.5 dB of the table. The fit leaves no residual,
  so all the energy sits exactly at the table delays.
- the realised RMS delay spread equals the configured DS
- E[H(f)H\*(f+Δf)] equals the Fourier transform of the table PDP
- the Ricean K-factor recovered from the generated channel matches the table

### CDL PDP verification

`examples/verify_cdl_pdp.py` runs the same checks with 1000 channel draws
(each with its own random ray phases) and plots them in
`results/cdl_pdp_verification.png`. It also checks that the error falls as
1/√N (no bias), and that the directional TR 38.901 element reshapes the PDP
exactly by each cluster's mean element gain.

Result at DS = 100 ns: cluster powers match the table to 0.05–0.14 dB rms
(maximum 0.29 dB). The realised delay spread is within 1% of the configured
value at 30, 100 and 300 ns. The recovered K-factors are 13.37 dB (CDL-D) and
22.07 dB (CDL-E), against 13.3 and 22.0 dB in the table.

---

## Time-domain waveform path

`nrdlsim/ofdm.py` provides a true sample-level CP-OFDM chain — resource grid →
IFFT + cyclic prefix → RF impairments + multipath + AWGN → CP removal + FFT →
one-tap equalisation — with configurable **carrier frequency offset** and
**symbol-timing offset**. It captures effects the frequency-domain SE loop
abstracts away, and is validated against theory (`examples/waveform_demo.py`):

* loopback reconstruction is exact (error ~1e-15);
* with delay spread < CP, time-domain multipath equals the per-subcarrier
  frequency-domain multiply (the CP orthogonality property);
* a CFO drives the effective SINR down to the analytic ICI ceiling
  `3/(π·ε)²` (ε = f_CFO/SCS) to within ~0.1 dB;
* a timing offset inside the CP is fully recoverable; beyond it, ISI collapses
  the SINR.

The main spectral-efficiency sweep runs in the frequency domain (exact when the
CP covers the delay spread and sync is ideal); the waveform module is the
sample-level path for studying synchronisation and CP-overrun.

## Notes and scope

* The MIMO TDL construction is the **correlation-based** extension of the
  SISO TDL models (Kronecker spatial correlation on i.i.d. per-tap fading),
  matching how TDL is applied to multi-antenna links.
* The **CDL** models synthesise the MIMO channel geometrically from the cluster
  angles, so they carry the true angular/spatial structure (beamforming gain,
  spatial correlation) rather than a correlation surrogate. Antennas are modelled
  as **uniform planar array (UPA) panels** (TR 38.901 §7.3): the panel is a
  `rows × cols` grid of element positions, each carrying one or two
  (cross-polar ±45°) polarizations, with configurable element spacing. The
  per-ray channel uses the full **dual-polarized coefficient** of eq. 7.5-22 —
  a 2×2 polarization coupling matrix with the model's XPR (cross-polar ratio)
  and independent random phases — plus location phases from the element
  positions. CDL-D/E add the deterministic Ricean specular LOS path (eq. 7.5-29).
  A scalar normalisation makes the average per-port power unity so the SNR sweep
  is comparable across array/polarization configurations.
* Each element optionally carries the **3GPP directional radiation pattern**
  (TR 38.901 Table 7.3-1): combined vertical/horizontal cuts with a configurable
  3 dB beamwidth, front-to-back ratio, maximum gain G_E,max, boresight azimuth
  and mechanical downtilt. The pattern scales each ray's field by √(gain) at its
  departure/arrival angle, angularly filtering the multipath (fewer effective
  clusters, altered spatial correlation and rank). The unit per-port-power
  normalisation accounts for the pattern analytically, so enabling it reshapes
  the channel's spatial structure without conflating it with a raw SNR offset.
  Set `--tx-pattern 38.901` (default `omni` = isotropic 0 dBi).
* **Panel orientation** follows TR 38.901 §7.1.3. Each panel has a bearing α,
  a mechanical downtilt β (positive points below the horizon) and a slant γ
  (roll about the boresight). The CDL parameters are `boresight_az_deg`,
  `downtilt_deg` and `slant_deg` for the gNB, plus the `rx_*` equivalents for
  the UE.
  - The rotation R = Rz(α)Ry(β)Rx(γ) is applied to the **element positions**,
    the **radiation pattern** and the **polarization** together. A tilted or
    rolled handset therefore also rotates its array and its polarization,
    through the ψ angle of eq. 7.1-15.
  - The UE element can differ from the gNB element (`rx_element_max_gain_dbi`,
    `rx_element_hpbw_deg`, `rx_element_front_back_db`).
  - `tests/test_antenna_orientation.py` checks the implementation against the
    spec's closed-form expressions (eqs. 7.1-7, 7.1-8, 7.1-15).
* Channel estimation, PMI selection and CQI use ideal-CSI SVD beamforming as a
  practical proxy for the Type-I codebook; DM-RS estimation error is modelled.
* The `miesm` link abstraction is the standard methodology for producing SE
  curves; the `ldpc` mode provides a bit-true reference. Neither embeds the full
  38.212 base-graph shift tables — the LDPC code uses the NR base-graph
  *architecture* (systematic part + dual-diagonal accumulator core, lifted by
  circular shifts) rather than the exact tabulated coefficients.

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

## Installation

```bash
python3 -m venv myvenv
source myvenv/bin/activate
pip install -r requirements.txt
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
```

### Key CLI options

| Option | Meaning | Default |
|--------|---------|---------|
| `--mu` | numerology (SCS = 15·2^μ kHz) | 1 (30 kHz) |
| `--rb` | resource blocks (bandwidth) | 51 |
| `--ntx / --nrx` | gNB / UE antennas | 4 / 2 |
| `--model` | `TDL-A…E`, `CDL-A…E` or `AWGN` | TDL-C |
| `--ds / --doppler` | delay spread (ns) / max Doppler (Hz) | 100 / 100 |
| `--mcs-table` | 1 (64QAM), 2 (256QAM), 3 (low-SE) | 2 |
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
 SNR[dB]   SE[b/s/Hz]   Tput[Mbps]     BLER   avgMCS  avgRank   avgCQI
   -5.00       0.36          6.7      0.100     1.35     1.97     2.80
    0.00       0.51          9.4      0.117     1.93     2.00     4.27
   10.00       3.15         57.8      0.117     9.72     2.00     8.12
   20.00       5.59        102.6      0.100    16.40     2.00    11.33
   30.00       8.46        155.3      0.083    23.47     2.00    13.47
```

Observed behaviour (all physically consistent):

* SE rises with SNR and saturates near the rank·log₂(M)·R ceiling.
* AWGN > TDL fading; SE scales with MIMO rank (1×1 ≈ 4, 2×2 ≈ 8, 4×4 ≈ 16 b/s/Hz).
* Residual BLER is held near the 0.1 target by the OLLA link adaptation.

The **spectral efficiency** is computed as

```
SE = delivered_information_bits / (num_slots · slot_duration) / occupied_bandwidth
```

## Tests

```bash
python tests/test_modules.py
```

Validates constellation energy, noiseless modulation round-trip, MCS/TBS,
TDL fading power normalisation, monotone BICM capacity, and genuine LDPC
coding gain.

---

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
* Channel estimation, PMI selection and CQI use ideal-CSI SVD beamforming as a
  practical proxy for the Type-I codebook; DM-RS estimation error is modelled.
* The `miesm` link abstraction is the standard methodology for producing SE
  curves; the `ldpc` mode provides a bit-true reference. Neither embeds the full
  38.212 base-graph shift tables — the LDPC code uses the NR base-graph
  *architecture* (systematic part + dual-diagonal accumulator core, lifted by
  circular shifts) rather than the exact tabulated coefficients.

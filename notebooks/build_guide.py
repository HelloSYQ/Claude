"""Generate the comprehensive guided-tour notebook for nrdlsim."""
import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []


def md(text):
    cells.append(nbf.v4.new_markdown_cell(text.strip("\n")))


def co(text):
    cells.append(nbf.v4.new_code_cell(text.strip("\n")))


# ---------------------------------------------------------------- title
md(r"""
# NR Downlink Link-Level Simulator — Guided Tour

A complete, modular **5G NR (New Radio) downlink (PDSCH) link-level simulator**.
This notebook walks through **every module**, explains what it does and which
3GPP specification it follows, and runs a live demo of each one — ending with a
full spectral-efficiency curve and the calibration against published 3GPP /
5G-IA results.

| Spec | Covers |
|------|--------|
| **TS 38.211** | numerology, resource grid, DM-RS, modulation, layer mapping |
| **TS 38.212** | DL-SCH coding: CRC, segmentation, LDPC, rate matching |
| **TS 38.214** | MCS/TBS, CSI (CQI/PMI/RI), link adaptation |
| **TR 38.901** | channel models (TDL, CDL), antenna panels & patterns |

### The per-slot processing chain

```
 channel (TR 38.901 TDL/CDL)  ->  DM-RS channel estimation
   ->  CSI feedback (RI/PMI/CQI, delayed)
       ->  scheduler (resource + MCS link adaptation, OLLA)
           ->  TBS / DL-SCH LDPC coding
               ->  layer mapping + precoding
                   ->  MIMO transmit (+ optional OFDM waveform) + AWGN
                       ->  MMSE equalisation  ->  demap / decode
                           ->  HARQ  ->  throughput / spectral efficiency
```
""")

# ---------------------------------------------------------------- setup
md(r"""
## 0 · Environment setup

Locate the package, enable inline plots, and import the pieces we will use.
""")
co(r"""
import os, sys
# find the repo root that contains the nrdlsim package and work from there
_p = os.getcwd()
while _p != os.path.dirname(_p) and not os.path.isdir(os.path.join(_p, "nrdlsim")):
    _p = os.path.dirname(_p)
sys.path.insert(0, _p); os.chdir(_p)

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams["figure.figsize"] = (9, 4)
plt.rcParams["axes.grid"] = True

import nrdlsim
print("nrdlsim version", nrdlsim.__version__, "| working dir:", os.getcwd())
""")

# ---------------------------------------------------------------- config
md(r"""
## 1 · `config.py` — numerology & configuration  *(TS 38.211 §4)*

Everything starts from configuration dataclasses. `CarrierConfig` fixes the
**numerology** μ (subcarrier spacing = 15·2^μ kHz), the bandwidth in resource
blocks, and derives slot timing. The other dataclasses configure the PDSCH
transmission, the antenna array, the channel and HARQ.
""")
co(r"""
from nrdlsim.config import CarrierConfig, PDSCHConfig, AntennaConfig, ChannelConfig

carrier = CarrierConfig(mu=1, n_size_grid=106)   # 30 kHz SCS, 106 PRB
print(f"numerology mu           = {carrier.mu}")
print(f"subcarrier spacing      = {carrier.scs_khz} kHz")
print(f"resource blocks         = {carrier.n_size_grid}  ({carrier.n_subcarriers} subcarriers)")
print(f"occupied bandwidth      = {carrier.occupied_bandwidth_hz/1e6:.2f} MHz")
print(f"slots per subframe (1ms)= {carrier.slots_per_subframe}")
print(f"slot duration           = {carrier.slot_duration_s*1e6:.1f} us")
print(f"OFDM symbols per slot   = {carrier.symbols_per_slot}")
""")

# ---------------------------------------------------------------- mcs/tbs
md(r"""
## 2 · `mcs_tables.py` + `tbs.py` — MCS, CQI and transport block size  *(TS 38.214 §5.1.3)*

The **MCS index** selects a modulation order Qm and a target code rate R
(Tables 5.1.3.1-1/2/3). The **CQI tables** (5.2.2.1-2/3) are used by CSI
feedback. `tbs.py` implements the full **transport block size** procedure
(§5.1.3.2), including the ≤3824-bit quantisation table and the >3824-bit
formula.
""")
co(r"""
from nrdlsim import mcs_tables, tbs as tbs_mod

# spectral efficiency Qm*R across MCS table 2 (up to 256QAM)
idx = range(mcs_tables.num_mcs(2))
eff = [mcs_tables.get_mcs(i, 2).spectral_efficiency for i in idx]
qm  = [mcs_tables.get_mcs(i, 2).modulation_order for i in idx]
colors = {2:"#2563eb",4:"#16a34a",6:"#f59e0b",8:"#dc2626"}
plt.figure()
plt.bar(list(idx), eff, color=[colors[q] for q in qm])
plt.xlabel("MCS index (table 2)"); plt.ylabel("Qm·R  (bits/RE)")
plt.title("MCS spectral efficiency — colour = modulation (QPSK/16/64/256QAM)")
plt.show()

info = mcs_tables.get_mcs(16, 2)
print(f"MCS 16 (table 2): {info.modulation_name}, R={info.target_code_rate:.3f}")
n_re = tbs_mod.re_per_rb(13, 24, 0)              # 13 symbols, 24 DMRS RE, no overhead
tb = tbs_mod.compute_tbs(n_re, 106, info.modulation_order, info.target_code_rate, 2)
print(f"TBS for 106 PRB, 2 layers = {tb} bits ({tb/8/1e3:.1f} kB) per slot")
""")

# ---------------------------------------------------------------- grid
md(r"""
## 3 · `resource_grid.py` — resource grid & DM-RS overhead  *(TS 38.211 §7.4.1)*

The PDSCH does not fill the whole slot: **DM-RS** (demodulation reference
signals) occupy some resource elements for channel estimation. This module
counts the DM-RS overhead and the remaining data REs — which directly sets the
achievable throughput.
""")
co(r"""
from nrdlsim import resource_grid as rg
from nrdlsim.config import DMRSConfig

pdsch = PDSCHConfig(num_rb=106, num_symbols=13, dmrs=DMRSConfig(additional_positions=1))
print("DM-RS OFDM symbol positions in slot:", rg.dmrs_symbol_positions(pdsch))
print(f"DM-RS REs per PRB      : {rg.dmrs_re_per_rb(pdsch)}")
print(f"data REs per PRB       : {rg.data_re_per_rb(pdsch)}")
print(f"DM-RS overhead fraction: {rg.dmrs_overhead_fraction(pdsch)*100:.1f}%")
print(f"total data REs (slot)  : {rg.num_data_re(pdsch)}")
""")

# ---------------------------------------------------------------- modulation
md(r"""
## 4 · `modulation.py` — QAM mapping & soft demapping  *(TS 38.211 §5.1)*

Gray-mapped QPSK / 16 / 64 / 256-QAM constellations (unit average energy) and a
**max-log-MAP soft demapper** that returns per-bit LLRs. The constellations are
built exactly from the TS 38.211 bit-to-symbol expressions.
""")
co(r"""
from nrdlsim.modulation import Modulator, _build_constellation

fig, axes = plt.subplots(1, 4, figsize=(14, 3.4))
for ax, (name, q) in zip(axes, [("QPSK",2),("16QAM",4),("64QAM",6),("256QAM",8)]):
    c = _build_constellation(q)
    ax.scatter(c.real, c.imag, s=12, color="#2563eb")
    ax.set_title(f"{name} (Qm={q})"); ax.set_aspect("equal"); ax.grid(True, alpha=0.3)
plt.suptitle("NR Gray-mapped constellations (unit average energy)")
plt.show()

# soft demap: LLR sign recovers bits at high SNR
mod = Modulator(6)
bits = np.random.default_rng(0).integers(0, 2, 6*1000)
sym = mod.modulate(bits)
noise = 0.01
rx = sym + np.sqrt(noise/2)*(np.random.randn(sym.size)+1j*np.random.randn(sym.size))
llr = mod.demodulate_llr(rx, noise)
print("64QAM soft-demap bit error rate at high SNR:", np.mean((llr<0)!=bits))
""")

# ---------------------------------------------------------------- TDL
md(r"""
## 5 · `channel_models.py` — TDL channel  *(TR 38.901 §7.7.2)*

**Tapped Delay Line** models TDL-A…E: a set of taps (delay, power) with
per-tap Rayleigh/Rician fading (sum-of-sinusoids Doppler) and a Kronecker
antenna-correlation MIMO extension. Below: the power-delay profile and the
resulting frequency-selective response.
""")
co(r"""
from nrdlsim.channel_models import TDLChannel

ch = TDLChannel("TDL-C", delay_spread_ns=300, max_doppler_hz=100,
                n_tx=2, n_rx=2, rng=np.random.default_rng(1))
fig, (a1, a2) = plt.subplots(1, 2, figsize=(13, 4))
a1.stem(ch.delays_s*1e9, 10*np.log10(ch.powers))
a1.set_xlabel("delay (ns)"); a1.set_ylabel("tap power (dB)")
a1.set_title("TDL-C power-delay profile (DS=300 ns)")

freqs = (np.arange(106)-53)*12*30e3
H = ch.frequency_response(freqs, 0.0)
a2.plot(freqs/1e6, 20*np.log10(np.abs(H[:,0,0])))
a2.set_xlabel("baseband frequency (MHz)"); a2.set_ylabel("|H(f)| (dB)")
a2.set_title("Frequency-selective fading across the band")
plt.show()
""")

# ---------------------------------------------------------------- CDL
md(r"""
## 5b · `channel_models.py` — CDL channel, dual-pol UPA & antenna pattern  *(TR 38.901 §7.3/7.5/7.7.1)*

**Clustered Delay Line** models specify per-cluster **angles**
(AoD/AoA/ZoD/ZoA). The MIMO channel is synthesised geometrically from
**uniform planar array** steering vectors with the full **dual-polarized**
coefficient (2×2 cross-pol coupling + XPR), and each element can carry the
**3GPP directional radiation pattern** (Table 7.3-1).
""")
co(r"""
from nrdlsim.channel_models import CDLChannel, element_power_gain, _CDL

# cluster angular map (AoD vs AoA) for CDL-C, marker size ~ cluster power
cl = np.array(_CDL["CDL-C"]["clusters"])
plt.figure(figsize=(6,5))
plt.scatter(cl[:,2], cl[:,3], s=(cl[:,1]-cl[:,1].min()+2)*12, c="#16a34a", alpha=0.7)
plt.xlabel("AoD (deg)"); plt.ylabel("AoA (deg)")
plt.title("CDL-C clusters — departure vs arrival azimuth"); plt.show()

# 3GPP directional element pattern (Table 7.3-1), azimuth cut
az = np.linspace(-180, 180, 361)
g = 10*np.log10(element_power_gain(az, 90.0))
plt.figure(figsize=(7,3.6))
plt.plot(az, g, color="#dc2626")
plt.axhline(8, ls=":", color="gray"); plt.axhline(5, ls=":", color="gray")
plt.xlabel("azimuth (deg)"); plt.ylabel("gain (dBi)")
plt.title("Directional element pattern: 8 dBi boresight, -3 dB at ±32.5°")
plt.show()

# dual-pol UPA channel: 8 tx ports = 2x2 positions x 2 pol
ch = CDLChannel("CDL-C", 100, 100, n_tx=8, n_rx=4, tx_pol=2, rx_pol=2,
                tx_layout=(2,2), tx_pattern="38.901", downtilt_deg=8,
                rng=np.random.default_rng(0))
H = ch.frequency_response(freqs, 0.0)
print("dual-pol UPA channel H shape [freq, rx, tx] =", H.shape)
""")

# ---------------------------------------------------------------- layer mapping
md(r"""
## 6 · `layer_mapping.py` — layer mapping & precoding  *(TS 38.211 §7.3.1)*

Codeword symbols are distributed across transmission layers (Table 7.3.1.3-1),
then mapped to antenna ports by a precoder. `svd_precoder` gives the
closed-loop eigen-beamforming precoder; open-loop uses an identity/DFT mapping.
""")
co(r"""
from nrdlsim.layer_mapping import layer_map, svd_precoder, num_codewords

cw = [np.arange(12)]                       # one codeword, 12 symbols
layers = layer_map(cw, num_layers=2)
print("codewords for rank 2:", num_codewords(2), "| layer array shape:", layers.shape)

W = svd_precoder(H, num_layers=2)          # per-subcarrier precoder
print("SVD precoder W shape [freq, n_tx, layers] =", W.shape)
print("columns orthonormal? W^H W ≈ I:",
      np.allclose(W[0].conj().T @ W[0], np.eye(2), atol=1e-6))
""")

# ---------------------------------------------------------------- ldpc
md(r"""
## 7 · `ldpc.py` — DL-SCH coding chain  *(TS 38.212 §5, §7.2)*

The bit-true FEC: **CRC** attachment (24A/24B/16), code-block **segmentation**,
**base-graph** selection (BG1/BG2), a **QC-LDPC** encoder (NR base-graph
architecture: systematic part + dual-diagonal accumulator core), and a
normalised **min-sum** decoder. Here it demonstrates genuine coding gain.
""")
co(r"""
from nrdlsim import ldpc as L
from nrdlsim.modulation import Modulator

code = L.build_code(bg=2, z=32)
print(f"QC-LDPC: BG2, Z=32 -> K={code.k_bits} info bits, N={code.n_bits} coded "
      f"(rate {code.k_bits/code.n_bits:.2f})")
mod = Modulator(2)
rng = np.random.default_rng(3)
def decode_rate(snr_db, trials=30):
    ok = 0; noise = np.sqrt(1/(2*10**(snr_db/10)))
    for _ in range(trials):
        info = rng.integers(0,2,code.k_bits).astype(np.int8)
        cw = L.encode(code, info)
        pad = (-cw.size) % 2
        tx = mod.modulate(np.concatenate([cw, np.zeros(pad, np.int8)]))
        rx = tx + noise*(rng.standard_normal(tx.size)+1j*rng.standard_normal(tx.size))
        llr = mod.demodulate_llr(rx, 2*noise**2)[:code.n_bits]
        hard, good = L.decode(code, llr, max_iter=25)
        ok += good and np.array_equal(hard[:code.k_bits], info)
    return ok/trials
print("decode success @ -6 dB:", decode_rate(-6.0))
print("decode success @ +6 dB:", decode_rate(+6.0), "  <- coding gain")
""")

# ---------------------------------------------------------------- link abstraction
md(r"""
## 8 · `link_abstraction.py` — BICM capacity & MIESM

For fast full-curve sweeps the coded BLER is predicted from the per-RE SINRs.
The chain is: map each SINR to its **bit-interleaved coded-modulation (BICM)
mutual information**, compress across REs to an **effective SINR** (MIESM), then
apply an NR-LDPC BLER waterfall. The BICM capacity curves are shown below.
""")
co(r"""
from nrdlsim.link_abstraction import bicm_capacity
snr = np.linspace(-10, 35, 60)
plt.figure()
for name, q in [("QPSK",2),("16QAM",4),("64QAM",6),("256QAM",8)]:
    plt.plot(snr, bicm_capacity(snr, q), label=name)
plt.plot(snr, np.log2(1+10**(snr/10)), "k--", lw=1, label="Shannon")
plt.xlabel("SNR (dB)"); plt.ylabel("BICM capacity (bits/symbol)")
plt.title("BICM mutual information per modulation (saturates at Qm)")
plt.legend(); plt.show()
""")

# ---------------------------------------------------------------- csi
md(r"""
## 9 · `csi.py` — CSI feedback: RI / PMI / CQI  *(TS 38.214 §5.2)*

From the estimated channel the UE selects the **rank (RI)** that maximises
throughput, the **precoder (PMI)** (SVD proxy for the codebook), and the
highest **CQI** meeting the BLER target. Reports are delayed by a configurable
number of slots to model the feedback loop.
""")
co(r"""
from nrdlsim.csi import compute_csi
noise_var = 10**(-15/10)                    # 15 dB SNR
report = compute_csi(H, noise_var, max_rank=4, mcs_table=2, target_bler=0.1)
print(f"selected rank (RI)   : {report.rank}")
print(f"reported CQI         : {report.cqi}")
print(f"effective SINR       : {report.eff_sinr_db:.1f} dB")
print(f"precoder shape (PMI) : {report.precoder.shape}")
""")

# ---------------------------------------------------------------- scheduler
md(r"""
## 10 · `scheduler.py` — MCS scheduling & OLLA  *(TS 38.214)*

The scheduler maps the reported CQI to an MCS and applies an **outer-loop link
adaptation (OLLA)** offset driven by HARQ ACK/NACK, so the operating BLER
converges to the target. Below, OLLA reacts to a NACK burst then recovers.
""")
co(r"""
from nrdlsim.scheduler import Scheduler
sched = Scheduler(total_rb=106, mcs_table=2, target_bler=0.1)
offsets = []
acks = [True]*15 + [False]*4 + [True]*20   # a NACK burst in the middle
for a in acks:
    sched.update_olla(a); offsets.append(sched.olla_offset)
plt.figure(figsize=(8,3.2))
plt.plot(offsets, marker="o", ms=3)
plt.xlabel("HARQ feedback #"); plt.ylabel("OLLA back-off (eff units)")
plt.title("OLLA backs off on NACKs, relaxes on ACKs"); plt.show()
""")

# ---------------------------------------------------------------- receiver
md(r"""
## 11 · `receiver.py` — MMSE MIMO equalisation

The receiver estimates the channel from DM-RS and applies an **MMSE** equaliser,
returning per-layer post-equaliser SINRs that feed the LLRs / link abstraction.
`batch_mmse_sinr` processes the whole allocation in a few batched matrix ops.
""")
co(r"""
from nrdlsim.receiver import batch_mmse_sinr
W2 = svd_precoder(H, 2)
H_eff = H @ W2                              # [freq, n_rx, layers]
sinr = batch_mmse_sinr(H_eff, noise_var)   # [freq, layers]
print("per-layer post-MMSE SINR (dB): layer0 = %.1f, layer1 = %.1f (mean over band)"
      % (10*np.log10(sinr[:,0].mean()), 10*np.log10(sinr[:,1].mean())))
""")

# ---------------------------------------------------------------- ofdm
md(r"""
## 12 · `ofdm.py` — time-domain OFDM waveform  *(TS 38.211 §5.3)*

The sample-level path: grid → IFFT + cyclic prefix → RF impairments (**CFO**,
**timing offset**) + multipath → CP removal + FFT. It captures effects the
frequency-domain path abstracts away: a CFO drives the effective SINR down to
the analytic inter-carrier-interference ceiling, and a timing error is harmless
inside the CP but ISI-limited beyond it.
""")
co(r"""
from nrdlsim.ofdm import OFDMModulator, waveform_evm, cfo_ici_sinr_db
mod = OFDMModulator(n_rb=51, scs_hz=30e3, mu=1); cp = int(mod.cp[1])
g = ((np.random.default_rng(0).integers(0,2,(mod.n_sc,mod.n_sym))*2-1)
     + 1j*(np.random.default_rng(1).integers(0,2,(mod.n_sc,mod.n_sym))*2-1))/np.sqrt(2)
h = np.zeros(20, complex); h[0]=1; h[5]=0.5; h[12]=0.3j

eps = np.linspace(0, 0.06, 13)
sinr = [waveform_evm(mod, g, 45, cfo_hz=e*30e3, h_time=h,
                     rng=np.random.default_rng(1))[1] for e in eps]
ceil = [cfo_ici_sinr_db(e*30e3, 30e3) if e>0 else np.nan for e in eps]
offs = np.arange(-cp-60, 61, 6)
sto = [waveform_evm(mod, g, 45, timing_offset=int(t), h_time=h,
                    rng=np.random.default_rng(2))[1] for t in offs]
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4))
a1.plot(eps, sinr, "o-", label="waveform"); a1.plot(eps, ceil, "--", label="analytic ICI")
a1.set_xlabel("normalised CFO ε=f/SCS"); a1.set_ylabel("effective SINR (dB)")
a1.set_title("CFO → inter-carrier interference"); a1.legend()
a2.plot(offs, sto, "s-"); a2.axvspan(-cp,0, color="#16a34a", alpha=0.1, label="within CP")
a2.set_xlabel("timing offset (samples)"); a2.set_ylabel("effective SINR (dB)")
a2.set_title(f"Timing offset vs CP ({cp} samples)"); a2.legend()
plt.show()
""")

# ---------------------------------------------------------------- link simulator
md(r"""
## 13 · `link_simulator.py` — full chain & spectral efficiency

`NRDownlinkSimulator` ties every module together, loops over slots with HARQ and
link adaptation, and outputs **spectral efficiency = throughput / bandwidth**.
Here is a full SNR sweep for a 4×2 CDL-C link (256QAM, link-adaptation on).
""")
co(r"""
from nrdlsim.config import SimConfig, HARQConfig
from nrdlsim.link_simulator import NRDownlinkSimulator

cfg = SimConfig(
    carrier=CarrierConfig(mu=1, n_size_grid=51),
    pdsch=PDSCHConfig(num_rb=51, num_layers=2, mcs_table=2),
    antenna=AntennaConfig(n_tx=4, n_rx=2),
    channel=ChannelConfig(model="TDL-C", delay_spread_ns=100, max_doppler_hz=100),
    snr_db_range=(-5, 30, 5), num_slots=60)
results = NRDownlinkSimulator(cfg).run(n_jobs=-1)   # parallel across SNR points

snrs = [r.snr_db for r in results]
se   = [r.spectral_efficiency for r in results]
fig,(a1,a2)=plt.subplots(1,2,figsize=(13,4))
a1.plot(snrs, se, "o-", color="#2563eb"); a1.set_xlabel("SNR (dB)")
a1.set_ylabel("spectral efficiency (b/s/Hz)"); a1.set_title("SE vs SNR (4x2 TDL-C, 256QAM)")
a2.plot(snrs, [r.avg_mcs for r in results], "s-", color="#16a34a", label="avg MCS")
a2.plot(snrs, [r.avg_cqi for r in results], "^-", color="#f59e0b", label="avg CQI")
a2.set_xlabel("SNR (dB)"); a2.set_title("Link adaptation follows the channel"); a2.legend()
plt.show()
print(f"peak SE = {max(se):.2f} b/s/Hz at {snrs[int(np.argmax(se))]:.0f} dB")
""")

# ---------------------------------------------------------------- calibration
md(r"""
## 14 · Validation & calibration

The simulator is validated at three levels:

* **Component self-tests** (`tests/test_modules.py`): constellation energy,
  noiseless modulation round-trip, TDL/CDL power, BICM monotonicity, genuine
  LDPC coding gain, dual-pol/UPA power, antenna pattern, OFDM loopback/CP/CFO.
* **Analytical calibration** (`tests/validate_ber.py`): uncoded QAM BER matches
  the closed-form AWGN theory.
* **3GPP / 5G-IA calibration** (`examples/calibration_5gia.py`,
  `docs/calibration.md`): SNR at 70 % throughput vs the **5G-IA** ideal
  calibration and the **3GPP TS 38.104** requirements. The simulator tracks
  case-to-case behaviour to **~0.65 dB** (bias-removed) across QPSK/16QAM/64QAM,
  TDL-A/B/C and SIMO/2×2 MIMO.

| | vs 5G-IA (ideal) | vs 3GPP TS 38.104 |
|---|---|---|
| mean Δ | −1.33 dB | −3.43 dB |
| bias-removed spread | 0.64 dB | 0.69 dB |
""")

# ---------------------------------------------------------------- perf + summary
md(r"""
## 15 · Performance & the module map

The simulator is vectorised and parallel: batched per-RB linear algebra, a
disk-cached BICM table, `run(n_jobs=-1)` / `--jobs` for parallel sweeps, and a
fast path that skips the CSI search under fixed-MCS. The 9-case calibration
runs in ~2 s.

### Module map

| Module | Role | Spec |
|--------|------|------|
| `config` | numerology, PDSCH/antenna/channel/HARQ configuration | 38.211 §4 |
| `mcs_tables`, `tbs` | MCS/CQI tables, transport block size | 38.214 §5.1.3 |
| `resource_grid` | resource grid + DM-RS overhead | 38.211 §7.4.1 |
| `modulation` | QAM mapping + soft LLR demapping | 38.211 §5.1 |
| `channel_models` | TDL + CDL, dual-pol UPA, element pattern | 38.901 §7.3/7.5/7.7 |
| `layer_mapping` | codeword→layer mapping + precoding | 38.211 §7.3.1 |
| `ldpc` | DL-SCH LDPC coding chain | 38.212 §5, §7.2 |
| `link_abstraction` | BICM capacity, MIESM, LDPC BLER | — |
| `csi` | CSI feedback RI/PMI/CQI + delay | 38.214 §5.2 |
| `scheduler` | MCS scheduling + OLLA | 38.214 |
| `receiver` | DM-RS estimation + MMSE equalisation | — |
| `ofdm` | time-domain CP-OFDM + CFO/timing | 38.211 §5.3 |
| `link_simulator` | full chain, HARQ, spectral efficiency | — |

**Run it from the CLI:**
```bash
python run_simulation.py --model CDL-C --snr -5 30 1 --slots 500 --jobs -1 --plot
python examples/calibration_5gia.py       # 3GPP / 5G-IA calibration
python tests/test_modules.py              # component self-tests
```
""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = "notebooks/NR_Downlink_Simulator_Guide.ipynb"
import os as _os
_os.makedirs("notebooks", exist_ok=True)
with open(out, "w") as f:
    nbf.write(nb, f)
print("wrote", out, "with", len(cells), "cells")

"""Generate the beginner series "NR link-level simulation from zero".

    python notebooks/learn/build_learn.py            # write the notebooks
    python notebooks/learn/build_learn.py --execute  # ... and run them, storing outputs

Unlike the two reference notebooks one directory up, these are stored *with*
their outputs, so a newcomer can read them on GitHub without running
anything.  ``--execute`` needs nbclient and ipykernel (``pip install
.[notebooks]``); run it where ipywidgets is *not* installed, so the stored
outputs show the slider cells' plain-text fallback rather than a widget that
GitHub can't display.  Cell ids come from each cell's text, so regenerating
changes only the cells that were edited.
"""
import argparse
import hashlib
import os

import nbformat as nbf

HERE = os.path.dirname(os.path.abspath(__file__))
COLAB = "https://colab.research.google.com/github/HelloSYQ/Claude/blob/HEAD/notebooks/learn/"
BADGE = "https://colab.research.google.com/assets/colab-badge.svg"

FILES = {0: "00_what_a_link_level_simulator_does.ipynb",
         1: "01_bits_symbols_and_noise.ipynb",
         2: "02_more_bits_per_symbol.ipynb"}


class Notebook:
    def __init__(self):
        self.cells = []

    def _add(self, new_cell, text):
        src = text.strip("\n")
        cid = hashlib.sha1(src.encode()).hexdigest()[:8]
        while any(c.id == cid for c in self.cells):
            cid = hashlib.sha1(cid.encode()).hexdigest()[:8]
        self.cells.append(new_cell(src, id=cid))

    def md(self, text):
        self._add(nbf.v4.new_markdown_cell, text)

    def co(self, text):
        self._add(nbf.v4.new_code_cell, text)


def header(nb, number, title, sentence, learn, needs, minutes):
    nb.md(f"""
# {number} · {title}

**NR link-level simulation from zero**, notebook {number} · [All notebooks](README.md) · [Glossary](GLOSSARY.md) · [![Open in Colab]({BADGE})]({COLAB}{FILES[number]})

**In one sentence:** {sentence}

**You will learn**
{learn}

**You need** {needs} **Time:** about {minutes} minutes.

**How to use it.** Run the cells from top to bottom with Shift+Enter. The outputs are stored, so you can also just read.
""")
    nb.co(SETUP)


SETUP = r'''
# Setup: make the nrdlsim package importable, then import the basics.
# (On Google Colab this installs nrdlsim from GitHub the first time.)
import os, sys, subprocess, importlib.util
if importlib.util.find_spec("nrdlsim") is None:
    root = os.getcwd()
    while root != os.path.dirname(root) and not os.path.isdir(os.path.join(root, "nrdlsim")):
        root = os.path.dirname(root)
    if os.path.isdir(os.path.join(root, "nrdlsim")):
        sys.path.insert(0, root)              # running inside the repository
    else:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "git+https://github.com/HelloSYQ/Claude"], check=True)

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.rcParams.update({"figure.dpi": 80, "axes.grid": True, "grid.alpha": 0.3})

def check(question, answer, expected, tol):
    """Say whether an answer is within tol of the expected value."""
    if answer is None:
        print(f"{question}: put your answer in place of None and run the cell again")
    elif abs(answer - expected) <= tol:
        print(f"{question}: yes, {answer} is right")
    else:
        print(f"{question}: not quite, try again (or open the answer below)")
'''


# =====================================================================
# 0 · What does a link-level simulator tell you?
# =====================================================================
def notebook_0():
    nb = Notebook()
    header(nb, 0, "What does a link-level simulator tell you?",
           "a link-level simulator answers *\"how fast can a base station send data "
           "to one phone, at a given signal quality?\"* by imitating, step by step, "
           "what the real system does.",
           "- what a radio *link* is, and what this simulator models;\n"
           "- decibels, the unit everyone in wireless uses;\n"
           "- how to run the simulator and read its main result, the "
           "spectral-efficiency curve.",
           "Python and a little numpy; no wireless knowledge.", 15)

    nb.md(r"""
## 1 · The question

When your phone downloads a video, the data comes from a **base station**: the antennas on a mast that serve your area. In 5G the base station is called a *gNB* and the phone a *UE* (user equipment). Data travelling from the base station to the phone is the **downlink**.

The base station sends the data as radio waves. On the way they get weaker, bounce off buildings and arrive as several echoes, and at the phone they are mixed with **noise**, random disturbances that every receiver picks up. The phone has to work out which bits were sent.

One transmitter and one receiver form a **link**. A **link-level simulator** (LLS) models one link in detail: the bits, the coding, the radio waves, the echoes, the noise, the receiver, and the base station's decisions about how fast to send. It answers one question:

> For a given signal quality, how many bits per second arrive correctly?

A *system-level* simulator makes the opposite trade-off: many base stations and phones, each link simplified. Its simplified links are calibrated with link-level results.
""")
    nb.co(r"""
# This cell only draws the picture; you can skip its code.
fig, ax = plt.subplots(figsize=(9, 2.6))
ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 3)
ax.add_patch(plt.Rectangle((0.2, 0.8), 2.0, 1.4, fc="#dbeafe", ec="#2563eb"))
ax.text(1.2, 1.5, "base station\n(gNB)", ha="center", va="center")
ax.add_patch(plt.Rectangle((7.8, 0.8), 2.0, 1.4, fc="#dcfce7", ec="#16a34a"))
ax.text(8.8, 1.5, "phone\n(UE)", ha="center", va="center")
for start, end, style in ((1.85, 1.75, "-"), (1.5, 1.5, "--"), (1.15, 1.25, ":")):
    ax.annotate("", xy=(7.75, end), xytext=(2.25, start),
                arrowprops=dict(arrowstyle="->", linestyle=style, color="#6b7280"))
ax.text(5, 2.15, "radio waves: weakened, echoed", ha="center", color="#374151")
ax.text(7.2, 0.5, "+ noise", ha="center", color="#dc2626")
ax.set_title("A downlink: one base station, one phone")
plt.show()
""")

    nb.md(r"""
## 2 · Decibels in five minutes

Wireless engineers compare powers as **ratios**, and the ratios span enormous ranges: the signal that reaches a phone can be a billion billion times weaker than what the base station transmits. So ratios are written in **decibels (dB)**:

$$\text{ratio in dB} = 10 \cdot \log_{10}(\text{ratio})$$

A few values are worth remembering:

| ratio | in dB |
|---|---|
| 1 (equal) | 0 dB |
| 2 | about 3 dB |
| 10 | 10 dB |
| 100 | 20 dB |
| 1/10 | −10 dB |

Multiplying ratios means adding decibels: 200 = 2 × 100 is 3 + 20 = 23 dB.

The ratio that matters most here is the **signal-to-noise ratio (SNR)**: the signal power divided by the noise power. An SNR of 20 dB means the signal is 100 times stronger than the noise; at 0 dB they are equally strong; at −5 dB the noise is about three times stronger than the signal, and, as you'll see, data still gets through.
""")
    nb.co(r"""
def to_db(ratio):
    return 10 * np.log10(ratio)

def from_db(db):
    return 10 ** (db / 10)

for db in (-10, 0, 3, 6, 10, 20, 30):
    print(f"{db:4d} dB  ->  ratio {from_db(db):8.2f}")
print("\nthe ratio 200 in dB:", round(to_db(200), 1))
""")
    nb.md(r"""
**Check yourself.** The noise power doubles while the signal stays the same. By how many dB does the SNR drop? Replace `None` with your answer and run the cell.
""")
    nb.co(r"""
answer = None
check("SNR drop when the noise doubles (dB)", answer, expected=3, tol=0.1)
""")
    nb.md(r"""
<details><summary>Answer</summary>

About 3 dB. The SNR halves, and a factor of 2 is 10·log₁₀(2) ≈ 3.01 dB.

</details>
""")

    nb.md(r"""
## 3 · Run the simulator once

Let's treat the simulator as a black box and look at what comes out. The set-up used throughout this series:

- a **20 MHz** slice of spectrum at **3.5 GHz**, a typical 5G band. It is divided into 51 *resource blocks* (notebook 6);
- a base station with **4 antennas** and a phone with **2**;
- **TDL-C**, a standard 3GPP model of an urban radio channel, with echoes spread over about 100 ns, and the phone carried at walking speed (3 km/h);
- SNRs from −5 dB to 40 dB.

At each SNR the simulator sends data for 400 *slots* of 0.5 ms each, 0.2 s in all. In every slot it goes through all the steps a real system does, and it counts the bits that arrive correctly.

**Predict first:** as the SNR goes up, does the data rate keep growing forever, or does it level off?
""")
    nb.co(r"""
from nrdlsim.config import SimConfig, CarrierConfig, PDSCHConfig, AntennaConfig, ChannelConfig
from nrdlsim.link_simulator import NRDownlinkSimulator

config = SimConfig(
    carrier=CarrierConfig(mu=1, n_size_grid=51),          # 30 kHz subcarriers, 51 resource blocks
    pdsch=PDSCHConfig(num_rb=51, num_layers=2, mcs_table=2),  # use all 51 blocks, up to 2 streams
    antenna=AntennaConfig(n_tx=4, n_rx=2),                # 4 antennas at the base station, 2 at the phone
    channel=ChannelConfig(model="TDL-C", delay_spread_ns=100, ue_speed_kmh=3.0),
    snr_db_range=(-5, 40, 2.5),                           # -5 dB to 40 dB in 2.5 dB steps
    num_slots=400)                                        # 400 slots of 0.5 ms per SNR

results = NRDownlinkSimulator(config).run()
snr_db = np.array([r.snr_db for r in results])
se = np.array([r.spectral_efficiency for r in results])
print(f"simulated {len(results)} SNR points")
""")

    nb.md(r"""
## 4 · Reading the curve

The simulator reports the **spectral efficiency (SE)**: the bits per second carried by each hertz of spectrum, in bit/s/Hz. It is the data rate divided by the bandwidth, so it doesn't depend on how much spectrum you have. Multiply it by the bandwidth to get the data rate: on our 18.4 MHz of occupied spectrum, 1 bit/s/Hz is 18.4 Mbit/s.
""")
    nb.co(r"""
bandwidth_hz = 51 * 12 * 30e3        # 51 resource blocks x 12 subcarriers x 30 kHz

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(snr_db, se, "o-", color="#2563eb")
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("spectral efficiency (bit/s/Hz)")
ax.set_title("Downlink spectral efficiency: 4 antennas to 2 antennas, TDL-C")
right = ax.secondary_yaxis("right", functions=(lambda s: s * bandwidth_hz / 1e6,
                                               lambda r: r * 1e6 / bandwidth_hz))
right.set_ylabel("data rate (Mbit/s)")
plt.show()

for target in (0, 10, 20, 30, 40):
    k = int(np.argmin(np.abs(snr_db - target)))
    print(f"SNR {target:2d} dB: {se[k]:5.2f} bit/s/Hz = {se[k] * bandwidth_hz / 1e6:4.0f} Mbit/s")
""")
    nb.md(r"""
**What you see**

1. **Even at −5 dB, data gets through**, slowly. The noise is stronger than the signal, but the base station sends few bits per symbol with plenty of redundancy, and the antennas add gain, so the phone can still decode them (notebooks 2–4 and 7).
2. **More SNR, more data.** A cleaner signal lets the base station pack more bits into each symbol (notebook 2) and send two streams at once to the phone's two antennas (notebook 7).
3. **The curve levels off.** There is a fastest setting, and no amount of SNR goes beyond it. The next cell works out where it is.
4. **The curve isn't perfectly smooth.** Fading is random and each point is only 0.2 s of transmission. More slots make it smoother.

**Where does the top speed come from?** The fastest setting combines the densest constellation in the table (256QAM, 8 bits per symbol, notebook 2), the highest code rate (0.93, so only 7 % redundancy, notebook 3) and two streams (notebook 7). Part of the time and frequency also carries reference signals and control information instead of data (notebook 6).
""")
    nb.co(r"""
from nrdlsim import mcs_tables, tbs as tbs_mod, resource_grid

top = mcs_tables.get_mcs(mcs_tables.num_mcs(2) - 1, 2)      # the fastest setting in the table
data_res = resource_grid.data_re_per_rb(config.pdsch)        # data-carrying resource elements per block
top_bits = tbs_mod.compute_tbs(data_res, 51, top.modulation_order, top.target_code_rate, 2)
ceiling = top_bits / 0.5e-3 / bandwidth_hz                   # bits per 0.5 ms slot -> bit/s/Hz
print(f"fastest setting: {top.modulation_name}, code rate {top.target_code_rate:.2f}, 2 streams")
print(f"-> at most {top_bits} bits per slot = {ceiling:.2f} bit/s/Hz = "
      f"{ceiling * bandwidth_hz / 1e6:.0f} Mbit/s")

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(snr_db, se, "o-", color="#2563eb", label="simulated")
ax.axhline(ceiling, ls="--", color="#dc2626", label="top speed of this system")
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("spectral efficiency (bit/s/Hz)"); ax.legend()
plt.show()
""")

    nb.md(r"""
## 5 · What's inside the box

In every 0.5 ms slot the simulator runs the same chain of steps. In plain words, with the notebook that explains each one:

| Step | What happens | Notebook |
|---|---|---|
| Bits to symbols | groups of bits become points of a *constellation* (arrows) | 1, 2 |
| Error-correcting code | redundancy is added so the phone can repair errors | 3 |
| Choosing the speed | the phone reports how well it hears; the base station picks the bits per symbol and the redundancy | 4 |
| The radio channel | the waves are weakened, echoed and faded on the way | 5 |
| The resource grid (OFDM) | the band is cut into many narrow subcarriers and short symbols; some carry reference signals | 6 |
| Several antennas | beamforming and parallel streams (*layers*) | 7 |
| Receiver and error prediction | the phone undoes the channel; the simulator predicts which blocks fail | 8 |
| The slot loop | feedback, retransmissions, counting the bits that got through | 9 |

Notebook 9 opens the box completely; by then every step will be familiar.
""")

    nb.md(r"""
## 6 · Try it yourself

**Predict first:** the phone now has **one** antenna instead of two. What happens to the curve at low SNR, and at high SNR?
""")
    nb.co(r"""
config_1rx = SimConfig(
    carrier=CarrierConfig(mu=1, n_size_grid=51),
    pdsch=PDSCHConfig(num_rb=51, num_layers=2, mcs_table=2),
    antenna=AntennaConfig(n_tx=4, n_rx=1),                # the only change: 1 antenna at the phone
    channel=ChannelConfig(model="TDL-C", delay_spread_ns=100, ue_speed_kmh=3.0),
    snr_db_range=(-5, 40, 2.5), num_slots=400)
se_1rx = np.array([r.spectral_efficiency for r in NRDownlinkSimulator(config_1rx).run()])

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(snr_db, se, "o-", color="#2563eb", label="phone with 2 antennas")
ax.plot(snr_db, se_1rx, "s-", color="#f59e0b", label="phone with 1 antenna")
ax.set_xlabel("SNR (dB)"); ax.set_ylabel("spectral efficiency (bit/s/Hz)"); ax.legend()
plt.show()
print(f"at -5 dB: {se[0]:.2f} vs {se_1rx[0]:.2f} bit/s/Hz")
print(f"at {snr_db[-1]:.0f} dB: {se[-1]:.2f} vs {se_1rx[-1]:.2f} bit/s/Hz")
""")
    nb.md(r"""
**What happened**

- **At high SNR the top speed halves.** A phone with one antenna can only receive one stream, so the base station sends one stream instead of two (notebook 7).
- **At low SNR the one-antenna phone is slower too.** Two antennas collect twice the signal energy, and they rarely fade at the same moment, so a second antenna also makes a weak signal more usable.

More to try: set `ue_speed_kmh` to 60 (a car), `model` to `"TDL-A"`, or `n_tx` to 8. Predict first, then run.
""")

    nb.md(r"""
## Where in the code

- `nrdlsim/config.py`: the configuration objects used above.
- `nrdlsim/link_simulator.py`: `NRDownlinkSimulator`, the slot loop (notebook 9).
- `run_simulation.py`: the same simulation from the command line, for example `python run_simulation.py --plot`.

## Recap

- A **link** is one transmitter and one receiver; a **link-level simulator** models one link in detail.
- **Decibels** turn ratios into sums: +3 dB ≈ ×2, +10 dB = ×10.
- The **SNR** compares the signal power with the noise power.
- The main result is **spectral efficiency vs SNR**. It grows with the SNR and levels off at the system's top speed.

**Words you met:** base station (gNB), phone (UE), downlink, link, noise, decibel, SNR, spectral efficiency, slot. All of them are in the [glossary](GLOSSARY.md).

**Next:** [notebook 1](01_bits_symbols_and_noise.ipynb), how bits become a radio signal and what noise does to it.
""")
    return nb


# =====================================================================
# 1 · Bits, symbols and noise
# =====================================================================
def notebook_1():
    nb = Notebook()
    header(nb, 1, "Bits, symbols and noise",
           "the transmitter turns groups of bits into *symbols* (arrows with a "
           "length and an angle), noise nudges every arrow at random, and the "
           "receiver guesses which arrow was sent; sometimes it guesses wrong.",
           "- why a radio signal is described by complex numbers, and why that is "
           "less scary than it sounds;\n"
           "- how two bits become one QPSK symbol, by hand and with `nrdlsim`;\n"
           "- what noise does to symbols, and how to count the bit errors it causes;\n"
           "- how the receiver can say how *sure* it is about each bit.",
           "notebook 0 (decibels, SNR).", 20)

    nb.md(r"""
## 1 · A radio wave is an arrow

A transmitter sends a sine wave at its carrier frequency, 3.5 GHz in our set-up. To carry information it changes two things about the wave:

- its **amplitude**: how big it is;
- its **phase**: how far along its cycle it is, a shift in time.

Two numbers describe the wave at each moment, and engineers store them as one **complex number**: an arrow whose length is the amplitude and whose angle is the phase. Three facts about complex numbers are all you need here:

1. `a + 1j*b` is the arrow to the point (a, b); in Python, `1j` is the imaginary unit.
2. `abs(z)` is the arrow's length and `np.angle(z)` its angle.
3. Multiplying two complex numbers multiplies their lengths and adds their angles, so multiplication **scales and rotates** an arrow. That is exactly what the radio channel does to a signal (notebook 5).

Below are two waves, and the same two waves as arrows.
""")
    nb.co(r"""
t = np.linspace(0, 2, 400)                       # time, in periods of the carrier
waves = {"amplitude 1, phase 0°": 1.0 * np.exp(1j * 0),
         "amplitude 0.5, phase 90°": 0.5 * np.exp(1j * np.pi / 2)}

fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.4), gridspec_kw={"width_ratios": [2, 1]})
for (name, z), color in zip(waves.items(), ("#2563eb", "#dc2626")):
    a1.plot(t, abs(z) * np.cos(2 * np.pi * t + np.angle(z)), color=color, label=name)
    a2.annotate("", xy=(z.real, z.imag), xytext=(0, 0),
                arrowprops=dict(arrowstyle="->", color=color, lw=2))
a1.set_xlabel("time (carrier periods)"); a1.set_ylabel("wave")
a1.legend(fontsize=8, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 1.0))
a2.set_xlim(-1.2, 1.2); a2.set_ylim(-1.2, 1.2); a2.set_aspect("equal")
a2.set_xlabel("real part"); a2.set_ylabel("imaginary part"); a2.set_title("the same waves as arrows")
plt.show()

z = 0.5 * np.exp(1j * np.pi / 2)
print("red arrow: length", abs(z), "| angle", np.degrees(np.angle(z)), "degrees")
""")
    nb.md(r"""
**What you see:** the red wave is half as tall as the blue one and shifted by a quarter of a period; its arrow is half as long and turned by 90°. From here on we only draw arrows.
""")

    nb.md(r"""
## 2 · Bits to symbols, by hand: QPSK

The transmitter sends one arrow at a time, each for a short moment called a **symbol**. If it can choose among 4 arrows, each symbol carries 2 bits, because 4 = 2².

**QPSK** (quadrature phase-shift keying) uses 4 arrows of the same length, one in each quarter of the plane:

- the first bit picks the sign of the horizontal (real) part: 0 gives +, 1 gives −;
- the second bit picks the sign of the vertical (imaginary) part, the same way;
- dividing by √2 makes every arrow 1 long, so the average **symbol energy** is 1.

The set of allowed arrows is the **constellation**. Neighbouring points differ in exactly one bit (*Gray mapping*), so a small mistake costs one bit, not two.
""")
    nb.co(r"""
def qpsk(bits):
    pairs = np.asarray(bits).reshape(-1, 2)           # two bits per symbol
    return ((1 - 2 * pairs[:, 0]) + 1j * (1 - 2 * pairs[:, 1])) / np.sqrt(2)

bits = np.array([0, 0,  0, 1,  1, 0,  1, 1])
symbols = qpsk(bits)
print(np.round(symbols, 3))

fig, ax = plt.subplots(figsize=(4, 4))
ax.scatter(symbols.real, symbols.imag, s=80, color="#2563eb")
for s, pair in zip(symbols, bits.reshape(-1, 2)):
    ax.annotate("".join(map(str, pair)), (s.real, s.imag), textcoords="offset points", xytext=(8, 8))
ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
ax.set_xlim(-1.2, 1.2); ax.set_ylim(-1.2, 1.2); ax.set_aspect("equal")
ax.set_title("QPSK: 2 bits per symbol")
plt.show()
""")
    nb.md(r"""
The simulator's modulator, `nrdlsim.modulation.Modulator`, follows TS 38.211, the 5G specification of the physical layer. For QPSK it produces exactly the same arrows:
""")
    nb.co(r"""
from nrdlsim.modulation import Modulator

rng = np.random.default_rng(1)
many_bits = rng.integers(0, 2, 2000)
print("same as nrdlsim:", np.allclose(qpsk(many_bits), Modulator(2).modulate(many_bits)))
""")

    nb.md(r"""
## 3 · Noise

Every receiver adds **noise**: tiny random disturbances, caused for example by the thermal motion of electrons in its electronics. The standard model is **additive white Gaussian noise (AWGN)**: each received symbol is the sent symbol plus a random arrow whose two parts are independent Gaussian ("bell curve") numbers.

The noise strength is its power, N₀. Our symbols have power 1, so

$$\text{SNR} = \frac{1}{N_0} \qquad\Longleftrightarrow\qquad N_0 = 10^{-\text{SNR}_\text{dB}/10}$$

where SNR is the ratio and SNR<sub>dB</sub> the same thing in decibels. Each part of the noise, real and imaginary, gets half of N₀.
""")
    nb.co(r"""
def add_noise(symbols, snr_db, rng):
    n0 = 10 ** (-snr_db / 10)                        # noise power, for symbols of power 1
    noise = np.sqrt(n0 / 2) * (rng.standard_normal(symbols.size)
                               + 1j * rng.standard_normal(symbols.size))
    return symbols + noise

sent = qpsk(rng.integers(0, 2, 2 * 1000))            # 1000 random QPSK symbols
""")
    nb.md(r"""
**Predict first:** below are 1000 received symbols at 20, 10 and 0 dB. At which SNR will some of them land in the wrong quarter of the plane?
""")
    nb.co(r"""
fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, snr in zip(axes, (20, 10, 0)):
    received = add_noise(sent, snr, rng)
    ax.scatter(received.real, received.imag, s=3, alpha=0.5, color="#2563eb")
    ax.scatter(sent.real, sent.imag, s=60, color="#dc2626", marker="x")
    ax.axhline(0, color="gray", lw=0.8); ax.axvline(0, color="gray", lw=0.8)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2.5, 2.5); ax.set_aspect("equal")
    ax.set_title(f"SNR {snr} dB")
plt.suptitle("Received QPSK symbols (blue) around the arrows that were sent (red)")
plt.show()
""")
    nb.md(r"""
**What you see:** at 20 dB each arrow arrives as a tight cloud. At 10 dB the clouds spread but stay in their quarter. At 0 dB the noise is as strong as the signal: the clouds reach across the axes, and some symbols land in the wrong quarter.

**Play with it.** If you run the notebook yourself, the next cell gives you an SNR slider. It needs the `ipywidgets` package, which Colab has.
""")
    nb.co(r"""
def show_received(snr_db=10.0):
    received = add_noise(sent, snr_db, np.random.default_rng(0))
    plt.figure(figsize=(4, 4))
    plt.scatter(received.real, received.imag, s=3, alpha=0.5)
    plt.axhline(0, color="gray", lw=0.8); plt.axvline(0, color="gray", lw=0.8)
    plt.xlim(-2.5, 2.5); plt.ylim(-2.5, 2.5); plt.gca().set_aspect("equal")
    plt.title(f"SNR {snr_db:.0f} dB"); plt.show()

try:
    import ipywidgets
    ipywidgets.interact(show_received,
                        snr_db=ipywidgets.FloatSlider(value=10, min=-5, max=30, step=1))
except ImportError:
    print("ipywidgets is not installed here, so there is no slider.\n"
          "Try other SNRs with, for example, show_received(snr_db=5).")
""")

    nb.md(r"""
## 4 · Deciding, and counting errors

The receiver doesn't know which arrow was sent. For QPSK its best guess is the nearest arrow, which comes down to signs: a positive real part means the first bit was 0, a negative one means 1, and the same for the imaginary part and the second bit.

A **bit error** is a bit decided wrongly. The **bit error rate (BER)** is the fraction of bits decided wrongly.

For QPSK over AWGN, theory gives the BER exactly:

$$\text{BER} = Q\left(\sqrt{\text{SNR}}\right)$$

with the SNR as a ratio, not in dB. Q(x) is the chance that a standard Gaussian number exceeds x: here, the chance that the noise pushes a symbol across an axis. Let's measure the BER and compare.
""")
    nb.co(r"""
from math import erfc, sqrt

def decide_qpsk(received):
    bits = np.empty((received.size, 2), dtype=int)
    bits[:, 0] = received.real < 0                   # first bit: sign of the real part
    bits[:, 1] = received.imag < 0                   # second bit: sign of the imaginary part
    return bits.reshape(-1)

def q_function(x):
    return 0.5 * erfc(x / sqrt(2))

snrs = np.arange(0, 12)
n_bits = 400_000
tx_bits = rng.integers(0, 2, n_bits)
tx = qpsk(tx_bits)
ber_measured = [np.mean(decide_qpsk(add_noise(tx, s, rng)) != tx_bits) for s in snrs]
ber_theory = [q_function(sqrt(10 ** (s / 10))) for s in snrs]

plt.figure(figsize=(7, 4))
plt.semilogy(snrs, ber_theory, "-", color="#111827", label="theory: Q(√SNR)")
plt.semilogy(snrs, ber_measured, "o", color="#2563eb", label=f"measured, {n_bits:,} bits per point")
plt.xlabel("SNR (dB)"); plt.ylabel("bit error rate (log scale)"); plt.legend()
plt.title("QPSK over AWGN")
plt.show()
for s in (0, 6, 10):
    print(f"SNR {s:2d} dB: about 1 bit in {1 / ber_theory[s]:5,.0f} is wrong")
""")
    nb.md(r"""
**What you see**

- The measured points sit on the theory curve.
- The vertical axis is logarithmic: each grid line down means 10 times fewer errors. The error rate falls steeply, from about one bit in six at 0 dB to about one in 1300 at 10 dB.
- Even one error in a thousand bits is useless for data. An internet packet of 1500 bytes holds 12 000 bits, so it would almost never arrive intact. Real systems therefore add redundancy, an *error-correcting code*, that lets the receiver repair errors (notebook 3).
""")

    nb.md(r"""
## 5 · How sure is the receiver?

A point far from the axes was almost certainly sent from that quarter; a point close to an axis could easily have come from the other side. Error-correcting decoders work much better if they know this. So instead of a hard 0 or 1, the receiver passes on a **log-likelihood ratio (LLR)** for each bit:

- positive means "probably 0", negative means "probably 1";
- the larger its size, the surer the receiver is.

Formally, LLR = log P(bit = 0 | what was received) − log P(bit = 1 | what was received). `Modulator.demodulate_llr` computes it for any constellation, from the received symbols and the noise power.
""")
    nb.co(r"""
snr = 3
n0 = 10 ** (-snr / 10)
bits3 = rng.integers(0, 2, 40_000)
received3 = add_noise(qpsk(bits3), snr, rng)
llr = Modulator(2).demodulate_llr(received3, n0)

plt.figure(figsize=(7, 3.6))
bins = np.linspace(-15, 15, 61)
plt.hist(llr[bits3 == 0], bins, alpha=0.6, color="#2563eb", label="bits sent as 0")
plt.hist(llr[bits3 == 1], bins, alpha=0.6, color="#dc2626", label="bits sent as 1")
plt.axvline(0, color="black", lw=1)
plt.xlabel("LLR"); plt.ylabel("number of bits"); plt.title(f"LLRs at an SNR of {snr} dB")
plt.legend()
plt.show()
print("LLR signs give the same decisions as section 4:",
      np.array_equal((llr < 0).astype(int), decide_qpsk(received3)))
""")
    nb.md(r"""
**What you see:** two humps. Bits sent as 0 mostly get positive LLRs, bits sent as 1 negative ones. Where the humps overlap, around 0, the receiver is unsure, and that is where its errors are. A decoder (notebook 3) trusts a bit with a large LLR and is ready to flip a bit whose LLR is close to 0.
""")

    nb.md(r"""
## 6 · Check yourself

**1.** About what SNR, in dB, makes 1 bit in 1000 wrong? Use the plot, or try values in `q_function(sqrt(10 ** (snr / 10)))`.
""")
    nb.co(r"""
answer = None
check("SNR for a BER of 1/1000 (dB)", answer, expected=9.8, tol=0.5)
""")
    nb.md(r"""
<details><summary>Answer</summary>

About 9.8 dB: `q_function(sqrt(10 ** (9.8 / 10)))` is 0.001.

</details>

**2.** At 10 dB, how many of the 12 000 bits of a 1500-byte packet do you expect to be wrong?
""")
    nb.co(r"""
answer = None
check("wrong bits in 12 000 at 10 dB", answer, expected=12_000 * q_function(sqrt(10)), tol=1.5)
""")
    nb.md(r"""
<details><summary>Answer</summary>

About 9: 12 000 × Q(√10) ≈ 12 000 × 0.00078 ≈ 9.4.

</details>

**3.** Swap the roles of the two bits in `qpsk`, so the first bit sets the imaginary part. Does the BER change? Why?

<details><summary>Answer</summary>

No. The noise is equally strong in every direction, so it doesn't matter which bit goes on which axis.

</details>
""")

    nb.md(r"""
## Where in the code

- `nrdlsim/modulation.py`: `Modulator.modulate` (bits to symbols) and `Modulator.demodulate_llr` (received symbols to LLRs), for QPSK up to 1024QAM.
- `tests/validate_ber.py` repeats section 4's comparison with theory for every constellation.

## Going further

- 3GPP TS 38.211, clause 5.1 (*Modulation mapper*): the exact mappings 5G uses, QPSK to 1024QAM.
- The QPSK chapter of any digital-communications textbook, for example J. G. Proakis, *Digital Communications*.

## Recap

- A radio wave's amplitude and phase form a complex number: an arrow.
- A **symbol** is one arrow. **QPSK** has 4 arrows, so it carries 2 bits per symbol.
- **AWGN** adds a random arrow to every symbol; the **SNR** sets its size.
- The receiver picks the nearest arrow. The **BER** falls steeply with the SNR but never reaches zero.
- **LLRs** tell the decoder how sure each bit is.

**Words you met:** amplitude, phase, complex number, symbol, constellation, QPSK, Gray mapping, AWGN, N₀, bit error rate, LLR. All are in the [glossary](GLOSSARY.md).

**Next:** [notebook 2](02_more_bits_per_symbol.ipynb), packing more bits into each symbol, and the price in SNR.
""")
    return nb


# =====================================================================
# 2 · More bits per symbol, and the price you pay
# =====================================================================
def notebook_2():
    nb = Notebook()
    header(nb, 2, "More bits per symbol, and the price you pay",
           "more bits per symbol needs more arrows, more arrows sit closer "
           "together, and arrows that are closer together need a cleaner signal, "
           "so the best choice depends on the SNR.",
           "- the QAM constellations 5G uses, from QPSK (2 bits per symbol) to "
           "1024QAM (10);\n"
           "- how much more SNR each step costs;\n"
           "- Shannon's limit: the most bits per symbol any system can deliver at a "
           "given SNR;\n"
           "- why the base station changes the constellation as the signal quality "
           "changes.",
           "notebooks 0 and 1.", 20)

    nb.md(r"""
## 1 · Bigger constellations

With 2^q arrows, each symbol carries q bits. 5G uses square constellations called **QAM** (quadrature amplitude modulation), in which both the length and the angle of the arrow carry information:

| constellation | arrows | bits per symbol |
|---|---|---|
| QPSK | 4 | 2 |
| 16QAM | 16 | 4 |
| 64QAM | 64 | 6 |
| 256QAM | 256 | 8 |
| 1024QAM | 1024 | 10 |

The transmit power doesn't grow with the constellation: the average symbol energy stays 1. So the more arrows there are, the closer together they sit.
""")
    nb.co(r"""
from nrdlsim.modulation import Modulator

names = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
for ax, (q, name) in zip(axes, names.items()):
    points = Modulator(q).constellation               # all 2^q arrows
    ax.scatter(points.real, points.imag, s=40 if q < 8 else 6, color="#2563eb")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal")
    ax.set_title(f"{name}: {len(points)} arrows, {q} bits")
plt.show()

for q, name in names.items():
    points = Modulator(q).constellation
    gaps = np.abs(points[:, None] - points[None, :])   # distance between every pair of arrows
    print(f"{name:7s} average energy {np.mean(abs(points) ** 2):.2f}, "
          f"closest two arrows {gaps[gaps > 0].min():.3f} apart")
""")
    nb.md(r"""
**What you see:** every constellation has an average energy of 1, but each step adds 2 bits and roughly halves the distance between neighbouring arrows. Halving a distance means a quarter of the power, about 6 dB. So to keep the same error rate, each step should need about 6 dB more SNR. Let's check.
""")

    nb.md(r"""
## 2 · The same noise on different constellations

**Predict first:** at an SNR of 18 dB, which of 16QAM, 64QAM and 256QAM can still be read without many errors?
""")
    nb.co(r"""
rng = np.random.default_rng(2)

def add_noise(symbols, snr_db, rng):                 # the same function as in notebook 1
    n0 = 10 ** (-snr_db / 10)
    return symbols + np.sqrt(n0 / 2) * (rng.standard_normal(symbols.size)
                                        + 1j * rng.standard_normal(symbols.size))

fig, axes = plt.subplots(1, 3, figsize=(12, 4))
for ax, q in zip(axes, (4, 6, 8)):
    mod = Modulator(q)
    received = add_noise(mod.modulate(rng.integers(0, 2, q * 3000)), 18, rng)
    ax.scatter(received.real, received.imag, s=2, alpha=0.4, color="#2563eb")
    ax.scatter(mod.constellation.real, mod.constellation.imag, s=8, color="#dc2626")
    ax.set_xlim(-1.4, 1.4); ax.set_ylim(-1.4, 1.4); ax.set_aspect("equal")
    ax.set_title(f"{names[q]} at 18 dB")
plt.show()
""")
    nb.md(r"""
**What you see:** at 18 dB the 16QAM clouds are well apart, the 64QAM clouds run into each other, and 256QAM is a blur in which many symbols would be misread.
""")

    nb.md(r"""
## 3 · The price in SNR

Measure each constellation's bit error rate, deciding every bit by the sign of its LLR, the hard decision of notebook 1. (This cell takes 10 to 30 seconds, depending on the computer.)
""")
    nb.co(r"""
plt.figure(figsize=(8, 4.2))
needed = {}
for q, name in names.items():
    mod = Modulator(q)
    bits = rng.integers(0, 2, q * 20_000)
    sent = mod.modulate(bits)
    snrs = np.arange(0, 3 * q + 11, 2.0)              # far enough for the BER to drop below 1e-4
    ber = np.array([np.mean((mod.demodulate_llr(add_noise(sent, s, rng), 10 ** (-s / 10)) < 0)
                            != bits) for s in snrs])
    plt.semilogy(snrs, np.where(ber > 0, ber, np.nan), "o-", ms=3, label=name)
    # SNR at which the BER crosses 1/1000 (the BER falls with SNR, so interpolate on reversed arrays)
    needed[name] = np.interp(-3, np.log10(ber[::-1] + 1e-12), snrs[::-1])
plt.axhline(1e-3, ls=":", color="gray")
plt.ylim(1e-5, 0.5); plt.xlabel("SNR (dB)"); plt.ylabel("bit error rate (log scale)")
plt.title("Bit error rate without any error-correcting code"); plt.legend()
plt.show()
for name, s in needed.items():
    print(f"{name:7s} needs about {s:4.1f} dB for 1 bit error in 1000")
""")
    nb.md(r"""
**What you see:** the curves have the same shape, shifted right by about 6 dB per step, just as the distances predicted. Each 2 extra bits per symbol cost about 6 dB, four times the SNR. Without coding, every constellation needs a lot of SNR to reach a usable error rate. Error-correcting codes (notebook 3) change that, and the next section shows how far they can go at best.
""")

    nb.md(r"""
## 4 · How many bits can each constellation really carry?

Real systems never send uncoded bits: they add redundancy (notebook 3) and correct the errors. So the useful question is: **with the best possible code, how many useful bits per symbol can each constellation deliver at a given SNR?** Information theory answers it.

- In 1948 Claude Shannon proved that over a channel with Gaussian noise no scheme can carry more than
  $$C = \log_2(1 + \text{SNR}) \ \text{bits per symbol},$$
  with the SNR as a ratio. Below that rate, errors can be made as rare as you like; above it, they can't. This is the **Shannon capacity**.
- A given constellation does worse, and it can never carry more than its own bits per symbol. 5G combines QAM with a code in the way called *bit-interleaved coded modulation* (BICM), and the limit for each constellation is its **BICM capacity**. `nrdlsim.link_abstraction.bicm_capacity` provides it.
""")
    nb.co(r"""
from nrdlsim.link_abstraction import bicm_capacity

snr_axis = np.linspace(-10, 40, 201)
shannon = np.log2(1 + 10 ** (snr_axis / 10))
all_names = {**names, 10: "1024QAM"}

plt.figure(figsize=(8, 4.5))
plt.plot(snr_axis, shannon, "k--", label="Shannon: log2(1 + SNR)")
for q, name in all_names.items():
    plt.plot(snr_axis, bicm_capacity(snr_axis, q), label=name)
plt.xlabel("SNR (dB)"); plt.ylabel("bits per symbol, with an ideal code"); plt.ylim(0, 11)
plt.title("Shannon's limit, and what each constellation can carry"); plt.legend(fontsize=8)
plt.show()
""")
    nb.md(r"""
**What you see**

1. Each constellation follows Shannon's line at first, then bends and flattens at its own number of bits.
2. At low SNR the big constellations bring nothing: at 0 dB QPSK carries as much as any of them, in fact slightly more.
3. To keep up with Shannon's line at high SNR you need bigger constellations. So the base station should use a small constellation when the signal is weak and a big one when it is strong. Choosing it slot by slot is called **link adaptation** (notebook 4).

The table below picks, at each SNR, the smallest constellation that is (almost) as good as any of them:
""")
    nb.co(r"""
print(" SNR   | Shannon | smallest constellation that is (almost) as good as any")
for s in (0, 5, 10, 15, 20, 25, 30):
    caps = {q: float(bicm_capacity(s, q)) for q in all_names}
    q_pick = min(q for q, c in caps.items() if c >= max(caps.values()) - 0.05)
    print(f"{s:3d} dB | {np.log2(1 + 10 ** (s / 10)):5.2f}   | "
          f"{all_names[q_pick]} ({caps[q_pick]:.2f} bits per symbol)")
""")
    nb.md(r"""
*Small print.* `bicm_capacity` is the curve the simulator itself uses. It was computed with the receiver's simplified ("max-log") demapper and a modest random sample, so it differs a little from the exact BICM capacity: by up to about 0.1 bit per symbol where each constellation is normally used, and by up to 0.3 bit per symbol for the large constellations at low SNR, where nobody would use them. The small sample is also why the curves wiggle a little. None of the lessons here depend on that.
""")

    nb.md(r"""
## 5 · From bits per symbol to bit/s/Hz

A channel of bandwidth B can carry about B symbols per second (notebook 6 shows how 5G does it, with many narrow subcarriers). So

$$\text{data rate} \approx B \times \text{bits per symbol}, \qquad \text{spectral efficiency} \approx \text{bits per symbol}.$$

That makes Shannon's limit a limit on notebook 0's spectral-efficiency curve. To compare like with like, simulate the simplest possible link: one antenna at each end and only noise on the way (AWGN), our running example otherwise. It takes a few seconds.
""")
    nb.co(r"""
from nrdlsim.config import SimConfig, CarrierConfig, PDSCHConfig, AntennaConfig, ChannelConfig
from nrdlsim.link_simulator import NRDownlinkSimulator
from nrdlsim import resource_grid

config = SimConfig(carrier=CarrierConfig(mu=1, n_size_grid=51),
                   pdsch=PDSCHConfig(num_rb=51, num_layers=1, mcs_table=2),
                   antenna=AntennaConfig(n_tx=1, n_rx=1),        # one antenna at each end
                   channel=ChannelConfig(model="AWGN"),          # nothing but noise
                   snr_db_range=(-5, 30, 2.5), num_slots=400)
results = NRDownlinkSimulator(config).run()
snr_sim = np.array([r.snr_db for r in results])
se_sim = np.array([r.spectral_efficiency for r in results])

best = np.max([bicm_capacity(snr_axis, q) for q in names], axis=0)   # best of QPSK...256QAM
# 12 subcarriers x 30 kHz x 0.5 ms = 180 symbols fit in one resource block per slot;
# this many of them carry data:
data_share = resource_grid.data_re_per_rb(config.pdsch) / (12 * 30e3 * 0.5e-3)

plt.figure(figsize=(8.5, 4.6))
plt.plot(snr_axis, shannon, "k--", label="Shannon's limit")
plt.plot(snr_axis, best, color="#16a34a", label="best of QPSK to 256QAM, ideal code")
plt.plot(snr_axis, best * data_share, color="#f59e0b",
         label=f"... on the {data_share:.0%} of time and frequency that carries data")
plt.plot(snr_sim, se_sim, "o-", color="#2563eb", label="the simulator: real codes, link adaptation")
plt.xlim(-5, 30); plt.ylim(0, 10)
plt.xlabel("SNR (dB)"); plt.ylabel("bit/s/Hz"); plt.legend(fontsize=8)
plt.title("One antenna at each end, noise only")
plt.show()

k = (snr_sim >= 5) & (snr_sim <= 25)
reached = se_sim[k] / (np.interp(snr_sim[k], snr_axis, best) * data_share)
print(f"between 5 and 25 dB the simulator reaches {reached.min():.0%} to "
      f"{reached.max():.0%} of the orange curve")
""")
    nb.md(r"""
**What you see**, from top to bottom:

- **Shannon's line** (black) is the absolute limit.
- **The best constellation with an ideal code** (green) follows it a little below, and stops at 8 bits: the MCS table used here goes up to 256QAM.
- **Only part of the time and frequency carries data** (orange). Reference signals for measuring the channel, a control region, and a short guard interval in every symbol (the *cyclic prefix*, which absorbs echoes) take the rest (notebook 6).
- **The simulator** (blue dots) reaches only two thirds to four fifths of the orange curve (printed above). Real LDPC codes need 1–2 dB more SNR than the limit (notebook 3); the system aims at about 10 % block errors and retransmits them; and it can only choose among 28 settings (notebook 4).

With several antennas, as in notebook 0, the curve can rise above Shannon's line. That is no contradiction: the SNR is measured at one antenna, and the antennas add gain and a second stream on top (notebook 7).
""")

    nb.md(r"""
## 6 · Check yourself

**1.** How many bits does one 64QAM symbol carry?
""")
    nb.co(r"""
answer = None
check("bits per 64QAM symbol", answer, expected=6, tol=0)
""")
    nb.md(r"""
<details><summary>Answer</summary>

6 bits, because 64 = 2⁶.

</details>

**2.** Using the capacity plot, at about what SNR can 16QAM carry 3 bits per symbol with an ideal code?
""")
    nb.co(r"""
answer = None
check("SNR for 3 bits per symbol with 16QAM (dB)", answer,
      expected=float(np.interp(3.0, bicm_capacity(snr_axis, 4), snr_axis)), tol=1.0)
""")
    nb.md(r"""
<details><summary>Answer</summary>

About 9 dB, 9.3 dB to be precise: `bicm_capacity(9.3, 4)` is about 3.

</details>

**3.** At 8 dB, why does 256QAM carry no more than 16QAM?

<details><summary>Answer</summary>

Its arrows are so close together that at 8 dB the noise blurs them. The code would have to add so much redundancy to repair the errors that the extra bits per symbol bring nothing.

</details>
""")

    nb.md(r"""
## Where in the code

- `nrdlsim/modulation.py`: `Modulator` builds every constellation from the TS 38.211 rules (`Modulator.constellation`).
- `nrdlsim/link_abstraction.py`: `bicm_capacity`, used by the simulator to predict errors (notebook 8).
- `nrdlsim/mcs_tables.py`: which constellations and code rates 5G offers (notebook 4).

## Going further

- 3GPP TS 38.211, clauses 5.1.3 to 5.1.7: the QPSK, 16QAM, 64QAM, 256QAM and 1024QAM mappings.
- C. E. Shannon, "A mathematical theory of communication", 1948: where the limit comes from.
- G. Caire, G. Taricco and E. Biglieri, "Bit-interleaved coded modulation", IEEE Trans. Information Theory, 1998.

## Recap

- **QAM** constellations carry 2 to 10 bits per symbol; at the same power, more bits means arrows closer together.
- Each 2 extra bits per symbol cost about **6 dB** of SNR.
- **Shannon's limit**, log₂(1 + SNR), caps what any system can carry; each constellation tops out at its own number of bits.
- The best constellation depends on the SNR, which is why the base station adapts it (**link adaptation**).
- Real systems sit below the limit: overheads, practical codes and coarse settings each cost a little.

**Words you met:** QAM, bits per symbol, Shannon capacity, BICM, link adaptation, cyclic prefix. All are in the [glossary](GLOSSARY.md).

**Next:** notebook 3, error-correcting codes: how redundancy repairs errors. It comes in the next part of the series.
""")
    return nb


BUILDERS = {0: notebook_0, 1: notebook_1, 2: notebook_2}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--execute", action="store_true",
                    help="run each notebook and store its outputs (needs nbclient, ipykernel)")
    args = ap.parse_args()
    for number, build in BUILDERS.items():
        nb = nbf.v4.new_notebook()
        nb.cells = build().cells
        nb.metadata = {"kernelspec": {"display_name": "Python 3", "language": "python",
                                      "name": "python3"},
                       "language_info": {"name": "python"}}
        if args.execute:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=900, kernel_name="python3", record_timing=False,
                           resources={"metadata": {"path": HERE}}).execute()
        path = os.path.join(HERE, FILES[number])
        nbf.write(nb, path)
        print("wrote", os.path.relpath(path), "(executed)" if args.execute else "")


if __name__ == "__main__":
    main()

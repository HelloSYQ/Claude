# Glossary

Plain-language meanings of the words used in [the beginner notebooks](README.md), with the notebook that explains each one. Words marked *(later)* are mentioned early but explained in a notebook still to come.

| Word | Meaning | Notebook |
|---|---|---|
| **3GPP** | The standards body that writes the 5G specifications ("TS 38.211" and so on). | 0 |
| **Amplitude** | How big a wave is. Together with the phase it forms a complex number. | 1 |
| **AWGN** (additive white Gaussian noise) | The standard noise model: a random Gaussian number added to every received symbol, equally strong in every direction. Also the name of a channel with nothing but such noise. | 1, 2 |
| **Bandwidth** | The width of the slice of spectrum used, in hertz. Our running example uses 51 resource blocks, 18.4 MHz of a 20 MHz carrier. | 0, 2 |
| **Base station (gNB)** | The radio equipment on a mast that serves the phones in its area. In 5G it is called a gNB. | 0 |
| **BICM** (bit-interleaved coded modulation) | How 5G combines a constellation with an error-correcting code. Its **BICM capacity** is the most a constellation can carry, per symbol, with an ideal code. | 2 |
| **Bit error rate (BER)** | The fraction of bits the receiver gets wrong. | 1, 2 |
| **Bits per symbol** | How many bits one symbol carries: 2 for QPSK, 4 for 16QAM, up to 10 for 1024QAM. | 1, 2 |
| **Block error rate (BLER)** | The fraction of transport blocks (chunks of data) that fail their error check. 5G aims at about 10 % and retransmits the failures. | 3 *(later)* |
| **CDL / TDL** | 3GPP's standard radio-channel models: *tapped delay line* (echoes at fixed delays) and *clustered delay line* (echoes that also come from directions). | 0, 5 *(later)* |
| **Code rate** | The share of transmitted bits that is data rather than redundancy. A code rate of 0.93 means 7 % redundancy. | 0, 3 *(later)* |
| **Complex number** | A pair of numbers written a + jb, drawn as an arrow from the origin. Its length is the amplitude and its angle the phase. | 1 |
| **Constellation** | The set of arrows a transmitter may send, for example the 4 arrows of QPSK. | 1, 2 |
| **CQI** (channel quality indicator) | The phone's report of how well it hears the base station. | 4 *(later)* |
| **Cyclic prefix** | A short guard interval at the start of every OFDM symbol that absorbs the echoes of the previous one. | 2, 6 *(later)* |
| **Decibel (dB)** | A ratio on a logarithmic scale: 10·log₁₀(ratio). +3 dB is about ×2, +10 dB is ×10. | 0 |
| **DM-RS** (demodulation reference signal) | Known symbols the phone uses to measure the channel. They take room that could otherwise carry data. | 0, 6 *(later)* |
| **Downlink** | The direction from the base station to the phone. | 0 |
| **Fading** | The signal getting stronger and weaker as the echoes add up or cancel, over frequency and over time. | 0, 5 *(later)* |
| **Gray mapping** | Labelling neighbouring constellation points so that they differ in only one bit. | 1 |
| **HARQ** | Retransmission: when a block fails, the base station sends it again and the phone combines the attempts. | 9 *(later)* |
| **Layer (stream)** | One of several independent data streams sent at the same time over several antennas. | 0, 7 *(later)* |
| **Link** | One transmitter and one receiver, here a base station and a phone. | 0 |
| **Link adaptation** | Choosing, slot by slot, how many bits per symbol to send and how much redundancy to add, according to the signal quality. | 2, 4 *(later)* |
| **Link-level simulator (LLS)** | A program that imitates one link in detail to find how much data gets through at a given SNR. | 0 |
| **LLR** (log-likelihood ratio) | The receiver's soft decision on a bit: positive means probably 0, negative probably 1, and the size says how sure it is. | 1 |
| **MCS** (modulation and coding scheme) | One row of the table the base station chooses from: a constellation plus a code rate. | 0, 4 *(later)* |
| **N₀** | The noise power. With symbols of power 1, SNR = 1/N₀. | 1 |
| **Noise** | Random disturbances every receiver adds to the signal. | 0, 1 |
| **OFDM** | How 5G divides the band into many narrow subcarriers that each carry slow symbols. | 6 *(later)* |
| **Phase** | How far along its cycle a wave is; the angle of its arrow. | 1 |
| **QAM** (quadrature amplitude modulation) | Square constellations in which both the length and the angle of the arrow carry bits: 16QAM, 64QAM, 256QAM, 1024QAM. | 2 |
| **QPSK** (quadrature phase-shift keying) | The constellation with 4 arrows of equal length: 2 bits per symbol. | 1 |
| **Resource block** | 12 neighbouring subcarriers, the unit 5G allocates bandwidth in. 51 of them at 30 kHz make our 18.4 MHz. | 0, 6 *(later)* |
| **Shannon capacity** | The most bits per symbol any system can carry at a given SNR: log₂(1 + SNR). | 2 |
| **Slot** | The basic time unit of 5G scheduling: 0.5 ms in our running example, 14 OFDM symbols. | 0 |
| **SNR** (signal-to-noise ratio) | Signal power divided by noise power, usually in dB. In this simulator it is measured at one receive antenna. | 0, 1 |
| **Spectral efficiency (SE)** | Bits per second carried by each hertz of spectrum (bit/s/Hz): the data rate divided by the bandwidth. | 0, 2 |
| **Symbol** | One arrow, sent for a short moment. | 1 |
| **System-level simulator** | A simulator of many base stations and phones, with simplified links calibrated by link-level results. | 0 |
| **UE** (user equipment) | 3GPP's name for the phone. | 0 |

"""Calibration test: uncoded QAM BER in AWGN vs closed-form theory.

This is the foundational correctness check for a link-level simulator: if the
modulation, soft-demapper and noise-variance convention are right, the measured
uncoded bit-error rate must match the analytical Gray-coded M-QAM curve.

Theory (Gray-mapped square M-QAM, Es/N0 = snr):
    BER ~ (4/k)(1 - 1/sqrt(M)) * Q( sqrt( 3*snr/(M-1) ) )
with k = log2(M).  QPSK reduces to BER = Q(sqrt(snr)).
"""

import os, sys
import numpy as np
from scipy.special import erfc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from nrdlsim.modulation import Modulator


def qfunc(x):
    return 0.5 * erfc(x / np.sqrt(2))


def theory_ber(snr_db, qm):
    snr = 10 ** (snr_db / 10.0)          # Es/N0 (per complex symbol)
    k = qm
    M = 2 ** qm
    return (4.0 / k) * (1 - 1 / np.sqrt(M)) * qfunc(np.sqrt(3 * snr / (M - 1)))


def measured_ber(snr_db, qm, n_bits=2_000_000, rng=None):
    rng = rng or np.random.default_rng(0)
    snr = 10 ** (snr_db / 10.0)
    noise_var = 1.0 / snr                 # Es=1 (unit-energy constellation)
    mod = Modulator(qm)
    n_bits -= n_bits % qm
    bits = rng.integers(0, 2, size=n_bits).astype(np.int8)
    tx = mod.modulate(bits)
    n = np.sqrt(noise_var / 2) * (rng.standard_normal(tx.size)
                                  + 1j * rng.standard_normal(tx.size))
    llr = mod.demodulate_llr(tx + n, noise_var)
    rx_bits = (llr < 0).astype(np.int8)   # LLR>0 -> bit 0
    return np.mean(rx_bits != bits)


def main():
    rng = np.random.default_rng(1)
    print(f"{'Mod':>7} {'SNR':>6} {'measured BER':>14} {'theory BER':>12} "
          f"{'ratio':>8}")
    print("-" * 52)
    ok = True
    for qm, snrs in [(2, [0, 4, 8]), (4, [6, 10, 14]),
                     (6, [12, 16, 20]), (8, [18, 22, 26])]:
        mod_name = {2: "QPSK", 4: "16QAM", 6: "64QAM", 8: "256QAM"}[qm]
        for snr_db in snrs:
            meas = measured_ber(snr_db, qm, rng=rng)
            th = theory_ber(snr_db, qm)
            ratio = meas / th if th > 0 else float('nan')
            flag = "" if 0.5 < ratio < 2.0 or meas == 0 else "  <-- OFF"
            if flag:
                ok = False
            print(f"{mod_name:>7} {snr_db:>5}dB {meas:>14.3e} "
                  f"{th:>12.3e} {ratio:>8.2f}{flag}")
    print("-" * 52)
    print("PASS: measured BER tracks theory within 2x"
          if ok else "FAIL: BER deviates from theory")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main() else 1)

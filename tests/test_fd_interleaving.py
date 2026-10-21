"""FD interleaving study: VRB-to-PRB mappings, code-block segmentation, rate
matching and the code-block RE mapping.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim import mcs_tables, tbs as tbs_mod                               # noqa: E402
from nrdlsim.ldpc import segment_transport_block, rate_match_lengths           # noqa: E402
from nrdlsim.re_mapping import (vrb_to_prb_non_interleaved,                    # noqa: E402
                                vrb_to_prb_nr_interleaved,
                                vrb_to_prb_block_interleaver, subcarrier_map,
                                code_block_spans, code_block_means,
                                code_block_prb_fraction)


def test_nr_interleaver_32_prb_bundle_4():
    prb = vrb_to_prb_nr_interleaved(32, 4)
    # VRB bundle j -> PRB bundle f(j) = r C + c, j = 2c + r, C = 4
    bundles = [0, 4, 1, 5, 2, 6, 3, 7]
    expected = np.concatenate([np.arange(4) + 4 * b for b in bundles])
    assert np.array_equal(prb, expected)


def test_nr_interleaver_is_a_bundle_permutation_for_any_bwp():
    for L in (2, 4):
        for start in range(L):
            for n in range(1, 120):
                prb = vrb_to_prb_nr_interleaved(n, L, start)
                assert np.array_equal(np.sort(prb), np.arange(n)), (L, start, n)
                # bundle 0 and the last bundle stay in place
                first = L - start % L
                assert np.array_equal(prb[:min(first, n)], np.arange(min(first, n)))
                assert prb[-1] == n - 1


def test_block_interleaver_4x8_write_rows_read_columns():
    prb = vrb_to_prb_block_interleaver(4, 8)
    v = np.arange(32)
    assert np.array_equal(prb, 4 * (v % 8) + v // 8)
    assert np.array_equal(prb[:8], np.arange(0, 32, 4))
    assert prb[8] == 1
    # reading the PRBs in order gives VRB 0, 8, 16, 24, 1, 9, ...
    assert np.array_equal(np.argsort(prb)[:6], [0, 8, 16, 24, 1, 9])
    g2 = vrb_to_prb_block_interleaver(4, 8, granularity=2)
    assert np.array_equal(g2[:4], [0, 1, 8, 9])
    assert np.array_equal(np.sort(g2), np.arange(64))


def test_segmentation_hand_examples():
    s = segment_transport_block(3824, 0.5)          # BG2, B = 3840 = K_cb
    assert (s.bg, s.c, s.k_prime, s.zc, s.k) == (2, 1, 3840, 384, 3840)
    s = segment_transport_block(8424, 0.8)          # BG1, B = 8448 = K_cb
    assert (s.bg, s.c, s.k_prime, s.zc, s.k) == (1, 1, 8448, 384, 8448)
    s = segment_transport_block(8456, 0.8)          # B = 8480 -> 2 CBs
    assert (s.bg, s.c, s.k_prime, s.zc, s.k) == (1, 2, 4264, 208, 4576)
    s = segment_transport_block(200, 0.3)           # BG2, B = 216 -> Kb = 8
    assert (s.bg, s.c, s.k_prime, s.zc) == (2, 1, 216, 28)


def test_segmentation_and_rate_matching_for_the_study_grid():
    n_re = tbs_mod.re_per_rb(13, 24, 0) * 32         # 4224 RE per layer
    for rank in (1, 2, 3, 4):
        for i in range(mcs_tables.num_mcs(2)):
            m = mcs_tables.get_mcs(i, 2)
            tb = tbs_mod.compute_tbs(132, 32, m.modulation_order,
                                     m.target_code_rate, rank)
            s = segment_transport_block(tb, m.target_code_rate)
            crc = (24 if tb > 3824 else 16) + (24 * s.c if s.c > 1 else 0)
            assert s.k_prime * s.c == tb + crc          # K' = B' / C exactly
            assert s.k_prime <= s.k
            g = n_re * m.modulation_order * rank
            e = rate_match_lengths(g, s.c, rank, m.modulation_order)
            unit = rank * m.modulation_order
            assert e.sum() == g and np.all(e % unit == 0)
            assert e.max() - e.min() <= unit and np.all(np.diff(e) >= 0)
            spans = code_block_spans(e, m.modulation_order, rank)
            assert spans[0, 0] == 0 and spans[-1, 1] == n_re
            assert np.array_equal(spans[1:, 0], spans[:-1, 1])


def test_code_block_means_match_an_explicit_re_loop():
    rng = np.random.default_rng(0)
    n_rb, n_sym, n_layers = 8, 5, 3
    n_sc = 12 * n_rb
    vals = rng.random((2, n_sc, n_layers))           # leading batch dimension
    e = rate_match_lengths(n_sc * n_sym * n_layers * 4, 7, n_layers, 4)
    spans = code_block_spans(e, 4, n_layers)
    for prb in (vrb_to_prb_non_interleaved(n_rb), vrb_to_prb_nr_interleaved(n_rb, 2),
                vrb_to_prb_block_interleaver(2, 4)):
        got = code_block_means(vals, spans, prb)
        sc = subcarrier_map(prb)
        for r, (a, b) in enumerate(spans):
            k = np.arange(a, b)
            ref = vals[:, sc[k % n_sc], :].mean(axis=(1, 2))
            assert np.allclose(got[:, r], ref)
        # the RE-weighted mean over the code blocks is the slot mean
        w = (spans[:, 1] - spans[:, 0]) / n_sc / n_sym
        assert np.allclose(got @ w, vals.mean(axis=(1, 2)))


def test_code_block_prb_fraction():
    e = rate_match_lengths(4224 * 8 * 4, 15, 4, 8)   # rank 4, 256QAM, 15 CBs
    spans = code_block_spans(e, 8, 4)
    for prb in (vrb_to_prb_non_interleaved(32), vrb_to_prb_block_interleaver(4, 8)):
        fr = code_block_prb_fraction(spans, prb)
        assert fr.shape == (15, 32) and np.allclose(fr.sum(1), 1.0)
    # CB 0 takes 281 consecutive REs per layer (VRBs 0..23): PRBs 0..23
    # without interleaving, the first 3 PRBs of every PRG with the 4 x 8 one
    assert spans[0, 1] == 281
    fr0 = code_block_prb_fraction(spans, vrb_to_prb_non_interleaved(32))
    fr1 = code_block_prb_fraction(spans, vrb_to_prb_block_interleaver(4, 8))
    assert np.all(fr0[0, :24] > 0) and np.all(fr0[0, 24:] == 0)
    prg = fr1[0].reshape(8, 4)
    assert np.all(prg[:, :3] > 0) and np.all(prg[:, 3] == 0)


def test_block_interleaver_32x8_for_256_prb():
    prb = vrb_to_prb_block_interleaver(32, 8)
    v = np.arange(256)
    assert np.array_equal(prb, 32 * (v % 8) + v // 8)
    assert np.array_equal(prb[:9], [0, 32, 64, 96, 128, 160, 192, 224, 1])
    assert np.array_equal(np.sort(prb), v)


def test_forced_minimum_number_of_code_blocks():
    nr = segment_transport_block(7944, 0.117)       # BG2, 3 CBs in NR
    assert nr.c == 3
    s = segment_transport_block(7944, 0.117, min_cbs=44)
    assert s.c == 44 and s.bg == 2
    assert s.k_prime == -(-(7944 + 24 + 24 * 44) // 44)   # ceil(B' / C)
    assert segment_transport_block(125016, 0.926).c == 15
    assert segment_transport_block(125016, 0.926, min_cbs=44).c == 44
    many = segment_transport_block(1_000_000, 0.926, min_cbs=44)
    assert many.c == segment_transport_block(1_000_000, 0.926).c > 44


def test_256_prb_every_symbol_holds_at_least_4_code_blocks():
    n_sc, n_sym = 12 * 256, 11
    n_re = tbs_mod.re_per_rb(13, 24, 0) * 256
    assert n_re == n_sc * n_sym
    for rank in (1, 2, 3, 4):
        for i in range(mcs_tables.num_mcs(2)):
            m = mcs_tables.get_mcs(i, 2)
            qm = m.modulation_order
            tb = tbs_mod.compute_tbs(132, 256, qm, m.target_code_rate, rank)
            s = segment_transport_block(tb, m.target_code_rate, min_cbs=4 * n_sym)
            assert s.c >= 44 and s.k_prime * s.c >= tb
            e = rate_match_lengths(n_re * qm * rank, s.c, rank, qm)
            spans = code_block_spans(e, qm, rank)
            for sym in range(n_sym):
                lo, hi = sym * n_sc, (sym + 1) * n_sc
                n_cb = np.sum((spans[:, 0] < hi) & (spans[:, 1] > lo))
                assert n_cb >= 4, (rank, i, sym, n_cb)

"""Regression tests: the CDL generator reproduces the TR 38.901 PDP.

Fast version of examples/verify_cdl_pdp.py (fewer drops, looser bounds that
still sit well above the Monte-Carlo noise at that drop count).
"""

import sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from nrdlsim.channel_validation import (table_pdp, merge_equal_delays,
                                        rms_delay_spread, los_k_factor_db,
                                        estimate_cluster_pdp, estimate_k_factor_db,
                                        frequency_correlation)

MODELS = ("CDL-A", "CDL-B", "CDL-C", "CDL-D", "CDL-E")


def _sionna_tables_dir():
    """TR 38.901 CDL tables shipped with Sionna, if its package files exist.

    Only the data files are read (the package is never imported), so
    ``pip install --no-deps sionna`` is enough to enable the check.
    """
    import importlib.util
    spec = importlib.util.find_spec("sionna")
    if spec is None or not spec.submodule_search_locations:
        return None
    base = os.path.join(list(spec.submodule_search_locations)[0],
                        "phy", "channel", "tr38901", "models")
    if not os.path.isdir(base):
        return None
    versions = sorted(v for v in os.listdir(base) if v.startswith("v"))
    return os.path.join(base, versions[-1]) if versions else None


def test_pdp_tables_match_reference():
    # Guards the tables themselves: the generator tests below compare the
    # channel with *our* tables and cannot catch a typo in them.
    import json
    from nrdlsim.channel_models import _CDL
    ref_dir = _sionna_tables_dir()
    if ref_dir is None:
        print("PDP tables vs reference: SKIPPED (pip install --no-deps sionna)")
        return
    for m in MODELS:
        ref = json.load(open(os.path.join(ref_dir, f"{m}.json")))
        off = 1 if ref["los"] else 0
        cl = np.array(_CDL[m]["clusters"])
        assert len(ref["delays"]) - off == len(cl), m
        np.testing.assert_allclose(cl[:, 0], ref["delays"][off:], atol=1e-9, err_msg=m)
        np.testing.assert_allclose(cl[:, 1], ref["powers"][off:], atol=1e-9, err_msg=m)
        if off:
            assert ref["delays"][0] == 0 and abs(ref["powers"][0]
                                                 - _CDL[m]["los_power_db"]) < 1e-9, m
    print(f"PDP tables (delays, powers, LOS) == TR 38.901 reference "
          f"({os.path.basename(ref_dir)}): OK")


def test_table_normalisation():
    # TR 38.901 normalises every CDL delay profile to unit RMS delay spread
    # (LOS models: including the LOS ray)
    for m in MODELS:
        ds = rms_delay_spread(*table_pdp(m))
        assert abs(ds - 1.0) < 0.01, (m, ds)
    assert abs(los_k_factor_db("CDL-D") - 13.3) < 1e-9
    assert abs(los_k_factor_db("CDL-E") - 22.0) < 1e-9
    print("CDL tables: unit normalised DS, K = 13.3 / 22.0 dB: OK")


def test_generated_cluster_powers_match_table():
    # Per-bin absolute powers (not renormalised) vs the table. At 800 drops a
    # correct generator stays <= 0.30 dB in every bin, while making any single
    # cluster 1 dB too strong shows >= 0.70 dB, so 0.5 dB separates the two.
    # (Only clusters sharing a delay with another -- the LOS ray's cluster 1
    # in CDL-D/E, CDL-E clusters 3/5 -- can't be isolated here; the K-factor
    # test covers the LOS pair.)
    for m in MODELS:
        r = estimate_cluster_pdp(m, 100.0, n_drops=800, n_times=3)
        err = 10 * np.log10(r["p_abs"] / r["p_ref"])
        assert r["worst_residual"] < 1e-20, (m, r["worst_residual"])  # exact delays
        assert np.abs(err).max() < 0.5, (m, np.round(err, 2))
        assert abs(r["total_power"] - 1.0) < 0.05, (m, r["total_power"])
        print(f"{m}: per-cluster power max err {np.abs(err).max():.2f} dB, "
              f"power/port {r['total_power']:.3f}: OK")


def test_realised_delay_spread():
    for m in MODELS:
        for ds in (30.0, 300.0):
            r = estimate_cluster_pdp(m, ds, n_drops=150, n_times=3)
            got = rms_delay_spread(r["tau_ns"], r["p_est"])
            want = rms_delay_spread(r["tau_ns"], r["p_ref"])
            assert abs(got / want - 1) < 0.04, (m, ds, got, want)
            assert abs(want / ds - 1) < 0.01, (m, ds, want)
    print("realised RMS delay spread == configured DS: OK")


def test_frequency_correlation():
    for m in MODELS:
        _, R_emp, R_th = frequency_correlation(m, 100.0, n_drops=120, n_times=3)
        assert np.abs(R_emp - R_th).max() < 0.06, (m, np.abs(R_emp - R_th).max())
    print("E[H(f)H*(f+df)] == Fourier transform of the table PDP: OK")


def test_los_k_factor():
    for m in ("CDL-D", "CDL-E"):
        k = estimate_k_factor_db(m, n_drops=600)
        assert abs(k - los_k_factor_db(m)) < 0.6, (m, k)
        print(f"{m}: recovered K {k:.2f} dB: OK")


if __name__ == "__main__":
    test_pdp_tables_match_reference()
    test_table_normalisation()
    test_generated_cluster_powers_match_table()
    test_realised_delay_spread()
    test_frequency_correlation()
    test_los_k_factor()
    print("\nAll CDL PDP tests passed.")

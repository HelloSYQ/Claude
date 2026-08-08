"""Top-level NR downlink link-level simulator.

Per-slot processing chain (TS 38.211/212/214 aligned):

    channel (TR 38.901 TDL)  ->  DM-RS channel estimation
      ->  CSI feedback (RI/PMI/CQI, delayed)
      ->  scheduler (resource + MCS link adaptation, OLLA)
      ->  TBS / DL-SCH coding  ->  layer mapping + precoding
      ->  MIMO transmission + AWGN
      ->  MMSE equalisation  ->  demap/decode (LDPC or MIESM abstraction)
      ->  HARQ  ->  throughput / spectral-efficiency accounting

Two FEC modes:
  * 'miesm' (default): fast link abstraction predicting BLER from post-equaliser
    SINRs — used to sweep full spectral-efficiency curves.
  * 'ldpc' : bit-true QC-LDPC coding on a representative code block, for
    validation on small configurations.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

from .config import SimConfig
from . import mcs_tables
from . import tbs as tbs_mod
from . import resource_grid as rg
from .channel_models import TDLChannel, CDLChannel, awgn_frequency_response
from .csi import compute_csi, CSIFeedbackChannel
from .scheduler import Scheduler
from .layer_mapping import svd_precoder
from .receiver import estimate_channel_from_dmrs, per_re_sinr, mmse_equalize
from .link_abstraction import (effective_sinr_miesm,
                               bler_from_effective_sinr)
from .modulation import Modulator
from . import ldpc as ldpc_mod


@dataclass
class SNRPointResult:
    snr_db: float
    spectral_efficiency: float     # bits/s/Hz
    throughput_bps: float
    bler: float
    avg_mcs: float
    avg_rank: float
    avg_cqi: float


class NRDownlinkSimulator:
    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

    # ------------------------------------------------------------------
    def _make_channel(self, snr_seed: int):
        c = self.cfg
        rng = np.random.default_rng(c.seed + snr_seed)
        model = c.channel.model
        if model.upper() == "AWGN":
            return None, rng
        if model.upper().startswith("CDL"):
            a = c.antenna
            chan = CDLChannel(
                model=model,
                delay_spread_ns=c.channel.delay_spread_ns,
                max_doppler_hz=c.channel.max_doppler_hz,
                n_tx=a.n_tx, n_rx=a.n_rx,
                carrier_freq_hz=c.channel.carrier_freq_hz,
                tx_pol=a.tx_pol, rx_pol=a.rx_pol,
                tx_layout=a.tx_layout, rx_layout=a.rx_layout,
                spacing_v=a.spacing_v, spacing_h=a.spacing_h,
                tx_pattern=a.tx_pattern, rx_pattern=a.rx_pattern,
                boresight_az_deg=a.boresight_az_deg, downtilt_deg=a.downtilt_deg,
                rx_boresight_az_deg=a.rx_boresight_az_deg,
                rx_downtilt_deg=a.rx_downtilt_deg,
                element_max_gain_dbi=a.element_max_gain_dbi,
                element_hpbw_deg=a.element_hpbw_deg,
                element_front_back_db=a.element_front_back_db,
                rng=rng)
            return chan, rng
        chan = TDLChannel(
            model=model,
            delay_spread_ns=c.channel.delay_spread_ns,
            max_doppler_hz=c.channel.max_doppler_hz,
            n_tx=c.antenna.n_tx, n_rx=c.antenna.n_rx,
            correlation=c.antenna.correlation,
            sample_rate_hz=c.carrier.subcarrier_spacing_hz * c.carrier.n_subcarriers,
            rng=rng)
        return chan, rng

    def _channel_response(self, chan, t, n_rb):
        """RB-granularity MIMO frequency response [n_rb, n_rx, n_tx]."""
        c = self.cfg
        if chan is None:
            return awgn_frequency_response(n_rb, c.antenna.n_tx, c.antenna.n_rx)
        # centre frequency (baseband) of each RB
        rb_freqs = (np.arange(n_rb) - n_rb / 2) * 12 * c.carrier.subcarrier_spacing_hz
        return chan.frequency_response(rb_freqs, t)

    # ------------------------------------------------------------------
    def run_point(self, snr_db: float, snr_seed: int = 0) -> SNRPointResult:
        c = self.cfg
        chan, rng = self._make_channel(snr_seed)
        n_rb = c.pdsch.num_rb
        max_rank = min(c.antenna.n_tx, c.antenna.n_rx)

        scheduler = Scheduler(n_rb, c.pdsch.mcs_table, c.pdsch.target_bler,
                              c.link_adaptation)
        csi_fb = CSIFeedbackChannel(c.csi_feedback_delay_slots)

        noise_var = 10 ** (-snr_db / 10.0)

        delivered_bits = 0
        total_blocks = 0
        error_blocks = 0
        mcs_hist = []
        rank_hist = []
        cqi_hist = []

        # HARQ state (single process, chase combining modelled by SINR sum)
        harq_sinr_lin = None
        harq_tx_count = 0
        harq_payload = None

        for slot in range(c.num_slots):
            t = slot * c.carrier.slot_duration_s
            H_true = self._channel_response(chan, t, n_rb)

            # --- UE: channel estimation + CSI feedback ---
            H_est = estimate_channel_from_dmrs(H_true, noise_var, rng)
            report = compute_csi(H_est, noise_var, max_rank,
                                 c.pdsch.mcs_table, c.pdsch.target_bler)
            csi_fb.push(report)
            active_csi = csi_fb.get()

            # --- scheduler decision ---
            decision = scheduler.schedule(active_csi, c.pdsch.mcs_index,
                                          c.pdsch.num_layers)
            rank = max(1, min(decision.num_layers, max_rank))
            mcs = decision.mcs_index
            info = mcs_tables.get_mcs(mcs, c.pdsch.mcs_table)

            # --- resource / TBS ---
            pdsch = c.pdsch
            pdsch.num_layers = rank
            n_dmrs = rg.dmrs_re_per_rb(pdsch)
            n_re_prb = tbs_mod.re_per_rb(pdsch.num_symbols, n_dmrs, pdsch.n_oh)
            tb_bits = tbs_mod.compute_tbs(n_re_prb, n_rb, info.modulation_order,
                                          info.target_code_rate, rank)

            # --- precoder actually used (from the delayed CSI report) ---
            W = active_csi.precoder if active_csi is not None else \
                svd_precoder(H_est, rank)
            if W.shape[2] != rank:
                W = svd_precoder(H_est, rank)

            # --- post-equaliser SINR over the (true) channel ---
            re_to_rb = np.arange(n_rb)
            sinr_lin = per_re_sinr(H_true, W, noise_var, re_to_rb)

            # --- decide block success ---
            if c.fec_mode == "ldpc":
                ok = self._ldpc_block(info, sinr_lin, tb_bits, rng)
                eff_db = effective_sinr_miesm(sinr_lin, info.modulation_order)
            else:
                # HARQ chase combining: accumulate linear SINR across retx.
                # Only combine when the retransmission uses a matching layout
                # (same rank -> same SINR vector length); otherwise restart.
                if (harq_sinr_lin is not None and harq_tx_count > 0
                        and harq_sinr_lin.shape == sinr_lin.shape):
                    comb = sinr_lin + harq_sinr_lin
                else:
                    comb = sinr_lin
                eff_db = effective_sinr_miesm(comb, info.modulation_order)
                bler = bler_from_effective_sinr(eff_db, info.modulation_order,
                                                info.target_code_rate, tb_bits)
                ok = rng.random() > bler
                harq_sinr_lin = comb

            total_blocks += 1
            harq_tx_count += 1

            if ok:
                delivered_bits += tb_bits
                scheduler.update_olla(True)
                harq_sinr_lin = None
                harq_tx_count = 0
            else:
                error_blocks += 1
                scheduler.update_olla(False)
                if (not c.harq.enabled
                        or harq_tx_count >= c.harq.max_transmissions):
                    harq_sinr_lin = None
                    harq_tx_count = 0

            mcs_hist.append(mcs)
            rank_hist.append(rank)
            cqi_hist.append(active_csi.cqi if active_csi else 0)

        total_time = c.num_slots * c.carrier.slot_duration_s
        bw = c.carrier.occupied_bandwidth_hz
        throughput = delivered_bits / total_time
        se = throughput / bw
        bler = error_blocks / max(total_blocks, 1)
        return SNRPointResult(
            snr_db=snr_db,
            spectral_efficiency=se,
            throughput_bps=throughput,
            bler=bler,
            avg_mcs=float(np.mean(mcs_hist)),
            avg_rank=float(np.mean(rank_hist)),
            avg_cqi=float(np.mean(cqi_hist)),
        )

    # ------------------------------------------------------------------
    def _ldpc_block(self, info, sinr_lin, tb_bits, rng) -> bool:
        """Bit-true single-code-block transmission for validation runs."""
        qm = info.modulation_order
        rate = info.target_code_rate
        # size a small code block for tractable min-sum decoding
        z = 32
        bg = 2
        code = ldpc_mod.build_code(bg, z)
        k = code.k_bits
        n = code.n_bits
        info_bits = rng.integers(0, 2, size=k).astype(np.int8)
        cw = ldpc_mod.encode(code, info_bits)

        mod = Modulator(qm)
        pad = (-cw.size) % qm
        cw_p = np.concatenate([cw, np.zeros(pad, dtype=np.int8)])
        tx = mod.modulate(cw_p)

        # average SINR over the allocation for this block
        snr_eff = np.mean(sinr_lin)
        noise = np.sqrt(1.0 / (2 * max(snr_eff, 1e-3)))
        rx = tx + noise * (rng.standard_normal(tx.size)
                           + 1j * rng.standard_normal(tx.size))
        llr = mod.demodulate_llr(rx, 2 * noise ** 2)[:cw.size]
        _, ok = ldpc_mod.decode(code, llr, max_iter=20)
        return ok

    # ------------------------------------------------------------------
    def run(self):
        start, stop, step = self.cfg.snr_db_range
        snrs = np.arange(start, stop + 1e-9, step)
        results = []
        for i, snr in enumerate(snrs):
            res = self.run_point(float(snr), snr_seed=i)
            results.append(res)
        return results

"""Configuration objects for the NR downlink link-level simulator.

All parameters follow 3GPP NR (5G) numerology conventions defined in
TS 38.211 (physical channels), TS 38.212 (multiplexing/coding),
TS 38.214 (physical layer procedures for data) and TR 38.901 (channel model).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

# Subcarrier spacing (kHz) for each numerology mu (TS 38.211 Table 4.2-1)
SCS_KHZ = {0: 15, 1: 30, 2: 60, 3: 120, 4: 240}

# Number of OFDM symbols per slot for normal cyclic prefix (TS 38.211 4.3.2)
SYMBOLS_PER_SLOT_NCP = 14


@dataclass
class CarrierConfig:
    """Carrier / bandwidth-part configuration."""

    mu: int = 1                    # numerology (0..4) -> SCS = 15*2^mu kHz
    n_size_grid: int = 51          # number of resource blocks in the BWP
    n_start_grid: int = 0          # first RB of the BWP relative to CRB 0
    cp: str = "normal"             # cyclic prefix type

    @property
    def scs_khz(self) -> int:
        return SCS_KHZ[self.mu]

    @property
    def symbols_per_slot(self) -> int:
        return SYMBOLS_PER_SLOT_NCP if self.cp == "normal" else 12

    @property
    def slots_per_subframe(self) -> int:
        return 2 ** self.mu

    @property
    def subcarrier_spacing_hz(self) -> float:
        return self.scs_khz * 1e3

    @property
    def n_subcarriers(self) -> int:
        return self.n_size_grid * 12

    @property
    def occupied_bandwidth_hz(self) -> float:
        """Occupied bandwidth = number of subcarriers * SCS."""
        return self.n_subcarriers * self.subcarrier_spacing_hz

    @property
    def slot_duration_s(self) -> float:
        return 1e-3 / self.slots_per_subframe


@dataclass
class DMRSConfig:
    """PDSCH DM-RS configuration (TS 38.211 7.4.1.1)."""

    config_type: int = 1           # type 1 (6 RE/RB/symbol) or type 2 (4 RE)
    num_front_load_symbols: int = 1
    additional_positions: int = 1  # number of additional DM-RS symbols
    num_cdm_groups_without_data: int = 2

    def re_per_rb_per_dmrs_symbol(self) -> int:
        # Type 1: 6 subcarriers used for DM-RS per CDM group column set.
        return 6 if self.config_type == 1 else 4

    def dmrs_symbols(self, n_pdsch_symbols: int) -> int:
        """Number of OFDM symbols carrying DM-RS in the allocation."""
        return self.num_front_load_symbols + self.additional_positions


@dataclass
class PDSCHConfig:
    """PDSCH transmission configuration (TS 38.214)."""

    num_layers: int = 1                 # transmission rank (1..8)
    num_rb: int = 51                    # allocated resource blocks
    start_symbol: int = 1               # first OFDM symbol of PDSCH
    num_symbols: int = 13               # number of OFDM symbols for PDSCH
    mcs_index: int = 16                 # initial MCS index
    mcs_table: int = 2                  # 1: up to 64QAM, 2: up to 256QAM
    n_oh: int = 0                       # overhead RE per RB (xOverhead)
    dmrs: DMRSConfig = field(default_factory=DMRSConfig)
    target_bler: float = 0.1            # link-adaptation BLER target


@dataclass
class AntennaConfig:
    """MIMO antenna configuration."""

    n_tx: int = 4                       # gNB transmit antennas
    n_rx: int = 2                       # UE receive antennas
    correlation: str = "low"            # low / medium / high (TS 36.101 style)


@dataclass
class ChannelConfig:
    """NR channel model configuration (TR 38.901)."""

    model: str = "TDL-C"                # TDL-A..E or 'AWGN'
    delay_spread_ns: float = 100.0      # desired RMS delay spread
    max_doppler_hz: float = 100.0       # maximum Doppler shift
    carrier_freq_hz: float = 3.5e9      # carrier frequency
    seed: Optional[int] = None


@dataclass
class HARQConfig:
    enabled: bool = True
    max_transmissions: int = 4          # 1 initial + up to 3 retransmissions


@dataclass
class SimConfig:
    """Top-level simulation configuration."""

    carrier: CarrierConfig = field(default_factory=CarrierConfig)
    pdsch: PDSCHConfig = field(default_factory=PDSCHConfig)
    antenna: AntennaConfig = field(default_factory=AntennaConfig)
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    harq: HARQConfig = field(default_factory=HARQConfig)

    snr_db_range: tuple = (-5.0, 30.0, 2.5)   # start, stop, step
    num_slots: int = 200                       # slots simulated per SNR point
    csi_feedback_delay_slots: int = 4          # CSI report delay
    link_adaptation: bool = True               # use CQI-driven MCS selection
    fec_mode: str = "miesm"                    # 'miesm' (fast) or 'ldpc' (bit-true)
    seed: int = 2025

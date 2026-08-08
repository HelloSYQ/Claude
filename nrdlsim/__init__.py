"""nrdlsim: an NR-compatible downlink link-level simulator.

Modules
-------
config            simulation / carrier / PDSCH / channel configuration
mcs_tables        MCS and CQI tables (TS 38.214)
tbs               transport block size (TS 38.214 5.1.3.2)
resource_grid     resource grid + DM-RS RE accounting (TS 38.211 7.4.1)
modulation        QAM mod / soft demod (TS 38.211 5.1)
ofdm              time-domain CP-OFDM waveform + CFO/timing (TS 38.211 5.3)
channel_models    NR TDL + CDL channel models (TR 38.901)
layer_mapping     codeword-to-layer mapping + precoding (TS 38.211 7.3.1)
ldpc              DL-SCH LDPC coding chain (TS 38.212)
link_abstraction  BICM capacity / MIESM / LDPC BLER model
csi               CSI feedback: RI / PMI / CQI (TS 38.214 5.2)
scheduler         resource scheduling + MCS link adaptation (OLLA)
receiver          MMSE MIMO equalisation + per-RE SINR
link_simulator    top-level per-slot chain + spectral efficiency
"""

from .config import (SimConfig, CarrierConfig, PDSCHConfig, AntennaConfig,
                     ChannelConfig, HARQConfig, DMRSConfig)
from .link_simulator import NRDownlinkSimulator, SNRPointResult

__all__ = [
    "SimConfig", "CarrierConfig", "PDSCHConfig", "AntennaConfig",
    "ChannelConfig", "HARQConfig", "DMRSConfig",
    "NRDownlinkSimulator", "SNRPointResult",
]

__version__ = "1.0.0"

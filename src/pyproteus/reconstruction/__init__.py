"""Delay-and-sum reconstruction (MATLAB ``delay-and-sum/``).

MATLAB name → Python:

==============================  ==========================================
``das_transmit_distance``       :func:`das_transmit_distance`
``compute_das_matrix``          :func:`compute_das_matrix`
``compute_das_matrix_general``  :func:`compute_das_matrix_general`
``load_RF_data``                :func:`load_rf_data` (+ :func:`iter_rf_frames`)
``DAS_reconstruction``          :func:`das_reconstruction`
``initialize_das_parameters``   :func:`initialize_das_parameters`
``DAS_reconstruction_general``  :func:`das_reconstruction_general`
==============================  ==========================================

Coordinates follow PROTEUS: ``x`` axial (depth), ``y`` lateral. Images are
``Nx``-by-``Ny``-by-``Nframes`` (axial-by-lateral). See ``docs/PARITY.md``
for the oracle status and the design differences with MATLAB (D-15).
"""

from .beamform import (
    beamform_frames,
    das_reconstruction,
    das_time_axis,
    log_compress,
    time_gain_compensation,
    time_to_peak,
)
from .das import (
    DasRecon,
    DasTransducer,
    DasTransmit,
    compute_das_matrix,
    compute_das_matrix_general,
    das_transmit_distance,
)
from .general import (
    DasDelayMismatchWarning,
    das_reconstruction_general,
    initialize_das_parameters,
)
from .rf_data import PULSING_SCHEMES, frame_files, iter_rf_frames, load_rf_data

__all__ = [
    "PULSING_SCHEMES",
    "DasDelayMismatchWarning",
    "DasRecon",
    "DasTransducer",
    "DasTransmit",
    "beamform_frames",
    "compute_das_matrix",
    "compute_das_matrix_general",
    "das_reconstruction",
    "das_reconstruction_general",
    "das_time_axis",
    "das_transmit_distance",
    "frame_files",
    "initialize_das_parameters",
    "iter_rf_frames",
    "load_rf_data",
    "log_compress",
    "time_gain_compensation",
    "time_to_peak",
]

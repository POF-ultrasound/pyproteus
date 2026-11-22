"""Default settings: ports of the seven ``GUIfunctions/reset_*.m`` (PROTEUS 8d37938).

The lookup leaves they call live in :mod:`pyproteus.transducer`,
:mod:`pyproteus.medium` and :mod:`pyproteus.microbubble` (ROADMAP §7).

Each function returns the struct MATLAB returns, field for field, with the
same classes (double vs logical) and sizes.

Pending M1 (signal leaves, not ported yet): ``reset_transducer`` does not
call ``estimate_impulse_response`` (``TransmitImpulseResponse``,
``ReceiveImpulseResponse`` stay absent for estimated types) and
``reset_transmit`` does not call ``get_voltage_signal``/``get_pressure_signal``
(``VoltageSignal``, ``PressureSignal``, ``MechanicalIndex`` stay absent).
:data:`PENDING_FIELDS` lists them; the oracle test skips exactly these.
"""

from __future__ import annotations

from os import PathLike
from pathlib import Path
from typing import Any

import numpy as np

from .matfile import loadmat
from .models import (
    Acquisition,
    BoundingBox,
    Distribution,
    Domain,
    Gas,
    Geometry,
    Medium,
    Microbubble,
    Settings,
    Shell,
    SimulationParameters,
    Transducer,
    Transmit,
    Vessel,
    Visualization,
)

__all__ = [
    "PENDING_FIELDS",
    "default_settings",
    "reset_acquisition",
    "reset_geometry",
    "reset_medium",
    "reset_microbubble",
    "reset_simulation_parameters",
    "reset_transducer",
    "reset_transmit",
]

#: Fields MATLAB sets that need the M1 signal leaves (see module docstring).
PENDING_FIELDS: dict[str, tuple[str, ...]] = {
    "Transducer": ("TransmitImpulseResponse", "ReceiveImpulseResponse"),
    "Transmit": ("VoltageSignal", "PressureSignal", "MechanicalIndex"),
}


def reset_acquisition() -> Acquisition:
    """Port of ``reset_acquisition.m``."""
    return Acquisition(
        FrameRate=500.0,
        NumberOfFrames=10.0,
        PulsingScheme="Standard",
        TimeBetweenPulses=100e-6,
        NumberOfPulses=1.0,
        Precompute=False,
        Folder="",
        ParallelTracking=False,
    )


def reset_simulation_parameters() -> SimulationParameters:
    """Port of ``reset_simulation_parameters.m``."""
    return SimulationParameters(
        IndependentVariable="CFL",
        CFL=0.3,
        PointsPerWavelength=6.0,
        NumberOfInteractions=0.0,
        HybridSimulation=True,
        SensorOnGrid=False,
        TransducerOnGrid=False,
        IntegrationDensity=1.0,
        Solver="3DC",
        DeviceNumber=0.0,
    )


def reset_microbubble() -> Microbubble:
    """Port of ``reset_microbubble.m`` (SonoVue, SF6, Marmottant shell)."""
    from pyproteus.microbubble import compute_polydisperse_distribution, get_gas_properties

    R, P = compute_polydisperse_distribution()
    return Microbubble(
        Type="SonoVue",
        Number=10.0,
        Advanced=False,
        Gas=get_gas_properties(Gas(Type="Sulfur hexafluoride")),
        ThermalModel="Prosperetti",
        SamplingRate=250e6,
        Distribution=Distribution(MeanRadius=2.14e-6, PDI=5.0, Radii=R, Probabilities=P),
        Shell=Shell(
            Model="Marmottant",
            Elasticity=0.56,
            InitialSurfaceTension=0.010,
            Viscosity=1e-8,
            FitFileSegers="fit_SigmaR_evaluated.mat",
            FitFile=np.zeros((0, 0)),
        ),
    )


def reset_medium() -> Medium:
    """Port of ``reset_medium.m`` (general tissue, blood vessel)."""
    from pyproteus.medium import (
        assign_liquid_properties,
        assign_medium_properties,
        compute_sound_speed_extrema,
    )

    medium = Medium(Tissue="General tissue", Inhomogeneity=0.02)
    medium = assign_medium_properties(medium)
    medium.InhomogeneityCutoff = 2.0
    vessel = Vessel(Tissue="Blood")
    vessel = assign_medium_properties(vessel)
    vessel = assign_liquid_properties(vessel, vessel.Tissue)
    medium.Vessel = vessel
    medium = compute_sound_speed_extrema(medium)
    medium.Save = True
    return medium


def reset_transducer(models_folder: str | PathLike[str] | None = None) -> Transducer:
    """Port of ``reset_transducer.m`` (P4-1).

    For an estimated impulse response MATLAB also calls
    ``estimate_impulse_response`` (pending M1, :data:`PENDING_FIELDS`).
    """
    from pyproteus.transducer import get_transducer_properties

    T = get_transducer_properties(Transducer(Type="P4-1"), models_folder)
    if T.ImpulseResponseType == "Estimate":
        T.SamplingRate = 250e6
        # Pending M1: T = estimate_impulse_response(T)
    return T


def reset_transmit(transducer: Transducer) -> Transmit:
    """Port of ``reset_transmit.m``.

    MATLAB also calls ``get_voltage_signal`` and ``get_pressure_signal``
    (pending M1, :data:`PENDING_FIELDS`). In the default ``'Pressure'``
    amplitude mode they do not change ``AcousticPressure``.
    """
    configuration = transducer.Configuration if transducer.isfield("Configuration") else "Linear"
    tx = Transmit(
        Type="Three-level",
        CenterFrequency=transducer.CenterFrequency,
        NumberOfCycles=2.0,
        AcousticPressure=200e3,
        Envelope="Rectangular",
        SamplingRate=250e6,
        AmplitudeMode="Pressure",
        VoltageAmplitude=1.0,
    )
    # Pending M1: tx = get_voltage_signal(tx); tx = get_pressure_signal(tx, transducer)
    tx.Advanced = False
    tx.LateralFocus = float("inf")
    tx.Angle = 0.0
    tx.ApodizationType = "Uniform apodization"
    tx.SignalsDefined = True
    if configuration in ("Linear", "Curved"):
        tx.DelayType = "Compute delays"
    elif configuration in ("RCA", "Custom"):
        tx.DelayType = "No delays"
    return tx


def _geometry_properties(folder: str, geometries_path: str | PathLike[str] | None) -> Any:
    if geometries_path is None:
        from pyproteus import data

        return data.load(f"{folder}/GeometryProperties.mat")
    return loadmat(Path(geometries_path) / folder / "GeometryProperties.mat")


def reset_geometry(
    geometry: Geometry | None = None, geometries_path: str | PathLike[str] | None = None
) -> Geometry:
    """Port of ``reset_geometry.m``.

    ``geometries_path`` is MATLAB's ``PATHS.GeometriesPath``; ``None`` uses the
    ``GeometryProperties`` shipped in :mod:`pyproteus.data` (``renal_tree``,
    ``mouse_brain``). Like MATLAB, the other fields of ``geometry`` are kept.
    """
    G = Geometry() if geometry is None else geometry.copy_struct()
    if not G.isfield("Folder"):
        G.Folder = "renal_tree"
    props = _geometry_properties(G.Folder, geometries_path)

    G.STLfile = props["STLfile"]
    G.STLunit = props["STLunit"]
    G.EmbedVessel = props["EmbedVessel"]
    BB = props["BoundingBox"]
    G.Rotation = props["Rotation"]
    domain = G.Domain.copy_struct() if G.Domain is not None else Domain()
    domain.Manual = False
    G.Domain = domain
    G.Visualization = Visualization.from_tree(props["Visualization"])

    def s(name: str) -> float:
        return float(np.asarray(BB[name]).reshape(-1)[0])

    bbox = G.BoundingBox.copy_struct() if G.BoundingBox is not None else BoundingBox()
    bbox.Center = np.array(
        [
            [(s("Xmax") + s("Xmin")) / 2],
            [(s("Ymax") + s("Ymin")) / 2],
            [(s("Zmax") + s("Zmin")) / 2],
        ]
    )
    bbox.Diagonal = np.array(
        [[s("Xmax") - s("Xmin")], [s("Ymax") - s("Ymin")], [s("Zmax") - s("Zmin")]]
    )
    G.BoundingBox = bbox
    G.startDepth = 25e-3
    G.Visualization.ShowTransducer = 1.0
    G.Visualization.ShowBeam = 0.0
    G.Visualization.ShowDomain = 1.0
    return G


def default_settings(
    geometry_folder: str = "renal_tree", geometries_path: str | PathLike[str] | None = None
) -> Settings:
    """The seven ``reset_*`` results, as MainGUI's ``startupFcn`` builds them.

    Derived settings (``Geometry.Domain`` limits, ``SimulationParameters.SamplingRate``,
    from ``compute_simulation_domain``/``update_simulation_parameters``) are
    not computed here (M1).
    """
    transducer = reset_transducer()
    return Settings(
        Microbubble=reset_microbubble(),
        Medium=reset_medium(),
        Acquisition=reset_acquisition(),
        Transducer=transducer,
        Transmit=reset_transmit(transducer),
        SimulationParameters=reset_simulation_parameters(),
        Geometry=reset_geometry(Geometry(Folder=geometry_folder), geometries_path),
    )

"""Settings models: the seven PROTEUS settings structs (pydantic v2).

The models mirror the MATLAB structs that the PROTEUS GUI builds and that
``write_to_file.m`` saves (variables ``Transducer``, ``Transmit``, ``Medium``,
``Microbubble``, ``Geometry``, ``Acquisition``, ``SimulationParameters``).
Faithful port (Phase A): attribute names *are* the MATLAB field names, field
types are what MATLAB stores, nothing is renamed or "improved".

Conventions (MATLAB → Python):

* **double scalar** → ``float``. Counts such as ``Number`` or
  ``NumberOfElements`` are doubles in MATLAB, so they are ``float`` here.
* **logical scalar** → ``bool``. Flag fields (:data:`Flag`) also accept a
  double (MATLAB sets some flags to ``1``/``0``, the GUI check boxes set
  ``true``/``false``) and keep whichever class they were given.
* **char** → ``str``.
* **double array** → :data:`MatArray`: a NumPy ``float64`` array with MATLAB's
  size, always ≥ 2-D (a 1×N row is shape ``(1, N)``, an N×1 column ``(N, 1)``).
  1-D input is rejected because its orientation is ambiguous. MATLAB ``[]`` is
  a 0×0 array.
* **absent field** → ``None``. MATLAB code tests ``isfield`` (for example
  ``reset_transmit`` for ``Transducer.Configuration``), so a missing field is
  not the same as ``[]``. Every field defaults to ``None``: an empty model is
  an empty struct. Defaults come from :mod:`pyproteus.config.defaults`
  (the ``reset_*`` functions), exactly as in MATLAB.
* **field order** is MATLAB's: the order fields were created in (input dict or
  keyword-argument order, then later assignments appended), so a struct read
  from a ``.mat`` file is written back with the same field order.
* **extra fields** (fields PROTEUS adds at run time, such as
  ``Transducer.integration_points``, or fields from other versions) are kept
  as MATLAB value trees (see :mod:`pyproteus.config.matfile`).

:meth:`MatStruct.to_tree` and :meth:`MatStruct.from_tree` convert to and from
MATLAB value trees (as read by :func:`pyproteus.config.matfile.loadmat`).
"""

from typing import Annotated, Any, ClassVar

import numpy as np
from pydantic import BaseModel, BeforeValidator, ConfigDict, PrivateAttr, model_validator

__all__ = [
    "SETTINGS_GROUPS",
    "Acquisition",
    "BoundingBox",
    "Distribution",
    "Domain",
    "Flag",
    "Gas",
    "Geometry",
    "MatArray",
    "MatStruct",
    "Medium",
    "Microbubble",
    "Settings",
    "Shell",
    "SimulationParameters",
    "Transducer",
    "Transmit",
    "Vessel",
    "Visualization",
    "scalar_to_tree",
]


# ----------------------------------------------------------------- field types


def _unwrap_scalar(v: Any) -> Any:
    """1×1 (or any size-1) array → Python scalar; other values unchanged."""
    if isinstance(v, np.ndarray | np.generic):
        arr = np.asarray(v)
        if arr.size != 1:
            raise ValueError(f"expected a MATLAB scalar, got size {arr.shape}")
        if arr.dtype == np.bool_:
            return bool(arr.reshape(-1)[0])
        if arr.dtype.kind in "iuf":
            return float(arr.reshape(-1)[0])
        raise ValueError(f"expected a real numeric or logical scalar, got dtype {arr.dtype}")
    return v


def _to_double(v: Any) -> Any:
    v = _unwrap_scalar(v)
    if isinstance(v, bool):
        raise ValueError("expected a double, got a logical")
    if isinstance(v, int):
        return float(v)
    return v


def _to_flag(v: Any) -> Any:
    v = _unwrap_scalar(v)
    if isinstance(v, int) and not isinstance(v, bool):
        return float(v)
    return v


def _to_matarray(v: Any) -> np.ndarray:
    if isinstance(v, list) and len(v) == 0:
        return np.zeros((0, 0))
    if isinstance(v, int | float) and not isinstance(v, bool):
        return np.array([[float(v)]])
    try:
        arr = np.asarray(v)
    except Exception as exc:
        raise ValueError(f"cannot convert to a MATLAB array: {exc}") from exc
    if arr.dtype == object or arr.dtype.kind not in "biuf":
        raise ValueError(f"expected a real numeric array, got dtype {arr.dtype}")
    if arr.ndim < 2:
        raise ValueError(
            "1-D arrays are ambiguous in MATLAB: give a 2-D array, e.g. shape (1, N) "
            "for a row vector or (N, 1) for a column vector"
        )
    return np.array(arr, dtype=np.float64)


Double = Annotated[float, BeforeValidator(_to_double)]
"""MATLAB double scalar."""

Logical = Annotated[bool, BeforeValidator(_unwrap_scalar)]
"""MATLAB logical scalar."""

Flag = Annotated[bool | float, BeforeValidator(_to_flag)]
"""MATLAB flag: logical or double scalar; the class given is kept."""

MatArray = Annotated[np.ndarray, BeforeValidator(_to_matarray)]
"""MATLAB double array with MATLAB's size (≥ 2-D)."""


def scalar_to_tree(v: Any) -> Any:
    """Python scalar → MATLAB value-tree leaf (1×1 array); str and arrays unchanged."""
    if isinstance(v, bool):
        return np.array([[v]], dtype=np.bool_)
    if isinstance(v, float | int):
        return np.array([[float(v)]])
    return v


# ---------------------------------------------------------------- base struct


class MatStruct(BaseModel):
    """A MATLAB 1×1 struct: declared fields typed, extra fields kept as-is."""

    model_config = ConfigDict(
        extra="allow",
        arbitrary_types_allowed=True,
        validate_assignment=True,
        validate_default=False,
        revalidate_instances="never",
    )

    # MATLAB field order: order of creation, as in the input dict / keyword
    # arguments, then later assignments appended (MATLAB appends new fields).
    _order: list[str] = PrivateAttr(default_factory=list)

    @model_validator(mode="wrap")
    @classmethod
    def _keep_field_order(cls, data: Any, handler: Any) -> Any:
        obj = handler(data)
        if isinstance(data, dict) and isinstance(obj, MatStruct):
            obj._order = [k for k in data if data[k] is not None]
        return obj

    def __setattr__(self, name: str, value: Any) -> None:
        if name.startswith("_"):
            super().__setattr__(name, value)
            return
        order = list(self._order)
        super().__setattr__(name, value)
        if value is None:
            order = [n for n in order if n != name]
        elif name not in order:
            order.append(name)
        self._order = order

    def present_fields(self) -> list[str]:
        """Names of the fields that exist in the MATLAB struct, in MATLAB order."""
        names = [n for n in self._order if self.isfield(n)]
        seen = set(names)
        declared = [n for n in type(self).model_fields if getattr(self, n) is not None]
        names += [n for n in [*declared, *(self.model_extra or {})] if n not in seen]
        return names

    def isfield(self, name: str) -> bool:
        """MATLAB ``isfield(S, name)``."""
        if name in type(self).model_fields:
            return getattr(self, name) is not None
        return name in (self.model_extra or {})

    def rmfield(self, name: str) -> None:
        """MATLAB ``S = rmfield(S, name)`` (in place)."""
        if not self.isfield(name):
            raise KeyError(f"no field {name!r}")
        if name in type(self).model_fields:
            setattr(self, name, None)
        else:
            del self.model_extra[name]  # type: ignore[union-attr]
            self._order = [n for n in self._order if n != name]

    def to_tree(self) -> dict[str, Any]:
        """MATLAB value tree (``dict``), absent fields left out."""
        out: dict[str, Any] = {}
        for name in self.present_fields():
            v = getattr(self, name)
            if isinstance(v, MatStruct):
                out[name] = v.to_tree()
            else:
                out[name] = scalar_to_tree(v)
        return out

    @classmethod
    def from_tree(cls, tree: dict[str, Any]) -> "MatStruct":
        """Build from a MATLAB value tree (``dict`` of a 1×1 struct)."""
        if not isinstance(tree, dict):
            raise TypeError(f"{cls.__name__}: expected a 1×1 struct (dict), got {type(tree)!r}")
        return cls.model_validate(tree)

    def copy_struct(self) -> "MatStruct":
        """Deep copy (MATLAB structs have value semantics)."""
        return type(self).from_tree(_copy_tree(self.to_tree()))


def _copy_tree(v: Any) -> Any:
    from .matfile import StructArray

    if isinstance(v, StructArray):
        return StructArray(_copy_tree(v.elements), v.fields)
    if isinstance(v, dict):
        return {k: _copy_tree(x) for k, x in v.items()}
    if isinstance(v, np.ndarray):
        if v.dtype == object:
            out = np.empty(v.shape, dtype=object)
            for idx in np.ndindex(v.shape):
                out[idx] = _copy_tree(v[idx])
            return out
        return v.copy()
    return v


# ------------------------------------------------------------------ Transducer


class Transducer(MatStruct):
    """``Transducer`` (``reset_transducer``, ``get_transducer_properties``, TransducerGUI)."""

    Type: str | None = None
    calibrated: Flag | None = None
    ImpulseResponseType: str | None = None
    Configuration: str | None = None
    NumberOfElements: Double | None = None
    Pitch: Double | None = None
    ElementWidth: Double | None = None
    ElementHeight: Double | None = None
    ElevationFocus: Double | None = None
    BandwidthLow: Double | None = None
    BandwidthHigh: Double | None = None
    CenterFrequency: Double | None = None
    Radius: Double | None = None
    SamplingRate: Double | None = None
    TransmitImpulseResponse: MatArray | None = None
    ReceiveImpulseResponse: MatArray | None = None


_Transducer = Transducer


# -------------------------------------------------------------------- Transmit


class Transmit(MatStruct):
    """``Transmit`` (``reset_transmit``, TransmitGUI, ``write_to_file``)."""

    Type: str | None = None
    CenterFrequency: Double | None = None
    NumberOfCycles: Double | None = None
    AcousticPressure: Double | None = None
    Envelope: str | None = None
    SamplingRate: Double | None = None
    AmplitudeMode: str | None = None
    VoltageAmplitude: Double | None = None
    VoltageSignal: MatArray | None = None
    PressureSignal: MatArray | None = None
    MechanicalIndex: Double | None = None
    Advanced: Flag | None = None
    LateralFocus: Double | None = None
    Angle: Double | None = None
    ApodizationType: str | None = None
    SignalsDefined: Flag | None = None
    DelayType: str | None = None
    Delays: MatArray | None = None
    Apodization: MatArray | None = None


_Transmit = Transmit


# ---------------------------------------------------------------------- Medium


class Vessel(MatStruct):
    """``Medium.Vessel``: blood (``assign_medium_properties``, ``assign_liquid_properties``)."""

    Tissue: str | None = None
    Density: Double | None = None
    SpeedOfSound: Double | None = None
    AttenuationA: Double | None = None
    AttenuationB: Double | None = None
    BonA: Double | None = None
    ThermalConductivity: Double | None = None
    SpecificHeat: Double | None = None
    DynamicViscosity: Double | None = None
    SurfaceTension: Double | None = None
    Temperature: Double | None = None
    Pressure: Double | None = None


_Vessel = Vessel


class Medium(MatStruct):
    """``Medium`` (``reset_medium``, MediumGUI)."""

    Tissue: str | None = None
    Inhomogeneity: Double | None = None
    SpeedOfSound: Double | None = None
    Density: Double | None = None
    BonA: Double | None = None
    AttenuationA: Double | None = None
    AttenuationB: Double | None = None
    InhomogeneityCutoff: Double | None = None
    Vessel: _Vessel | None = None
    SpeedOfSoundMinimum: Double | None = None
    SpeedOfSoundMaximum: Double | None = None
    Save: Flag | None = None


_Medium = Medium


# ----------------------------------------------------------------- Microbubble


class Gas(MatStruct):
    """``Microbubble.Gas`` (``get_gas_properties``)."""

    Type: str | None = None
    ThermalConductivity: Double | None = None
    Density: Double | None = None
    MolarMass: Double | None = None
    SpecificHeat: Double | None = None
    HeatCapacityRatio: Double | None = None


_Gas = Gas


class Distribution(MatStruct):
    """``Microbubble.Distribution`` (size distribution)."""

    MeanRadius: Double | None = None
    PDI: Double | None = None
    Radii: MatArray | None = None
    Probabilities: MatArray | None = None


_Distribution = Distribution


class Shell(MatStruct):
    """``Microbubble.Shell``. ``FitFile`` is ``[]`` (0×0 array) or a file path."""

    Model: str | None = None
    Elasticity: Double | None = None
    InitialSurfaceTension: Double | None = None
    Viscosity: Double | None = None
    FitFileSegers: str | None = None
    FitFile: str | MatArray | None = None
    NormalizedArea: MatArray | None = None
    SurfaceTension: MatArray | None = None


_Shell = Shell


class Microbubble(MatStruct):
    """``Microbubble`` (``reset_microbubble``, MicrobubblesGUI)."""

    Type: str | None = None
    Number: Double | None = None
    Advanced: Flag | None = None
    Gas: _Gas | None = None
    ThermalModel: str | None = None
    SamplingRate: Double | None = None
    Distribution: _Distribution | None = None
    Shell: _Shell | None = None


_Microbubble = Microbubble


# -------------------------------------------------------------------- Geometry


class Domain(MatStruct):
    """``Geometry.Domain`` (``reset_geometry``, ``compute_simulation_domain``, GeometryGUI)."""

    Manual: Flag | None = None
    Margin: Double | None = None
    TransducerSurface: MatArray | None = None
    TransducerProjection: MatArray | None = None
    Xmax: Double | None = None
    Ymax: Double | None = None
    Zmax: Double | None = None
    Xmin: Double | None = None
    Ymin: Double | None = None
    Zmin: Double | None = None
    Vertices: MatArray | None = None


_Domain = Domain


class Visualization(MatStruct):
    """``Geometry.Visualization`` (GUI display options)."""

    Image: str | None = None
    Fraction: Double | None = None
    ShowTransducer: Flag | None = None
    ShowBeam: Flag | None = None
    ShowDomain: Flag | None = None


_Visualization = Visualization


class BoundingBox(MatStruct):
    """``Geometry.BoundingBox``: ``Center`` and ``Diagonal`` are 3×1 columns."""

    Center: MatArray | None = None
    Diagonal: MatArray | None = None


_BoundingBox = BoundingBox


class Geometry(MatStruct):
    """``Geometry`` (``reset_geometry``, GeometryGUI, ``compute_simulation_domain``)."""

    Folder: str | None = None
    STLfile: str | None = None
    STLunit: Double | None = None
    EmbedVessel: Flag | None = None
    Rotation: MatArray | None = None
    Domain: _Domain | None = None
    Visualization: _Visualization | None = None
    BoundingBox: _BoundingBox | None = None
    startDepth: Double | None = None
    Center: MatArray | None = None


_Geometry = Geometry


# ----------------------------------------------------------------- Acquisition


class Acquisition(MatStruct):
    """``Acquisition`` (``reset_acquisition``, AcquisitionGUI)."""

    FrameRate: Double | None = None
    NumberOfFrames: Double | None = None
    PulsingScheme: str | None = None
    TimeBetweenPulses: Double | None = None
    NumberOfPulses: Double | None = None
    Precompute: Flag | None = None
    Folder: str | None = None
    ParallelTracking: Flag | None = None


_Acquisition = Acquisition


# -------------------------------------------------------- SimulationParameters


class SimulationParameters(MatStruct):
    """``SimulationParameters`` (``reset_simulation_parameters``, SimulationParametersGUI).

    ``SamplingRate`` is set by ``update_simulation_parameters``, ``GridSize`` by
    ``write_to_file``; ``MaxPrime`` and ``PMLmin`` are optional overrides read
    by ``define_grid`` (``isfield``).
    """

    IndependentVariable: str | None = None
    CFL: Double | None = None
    PointsPerWavelength: Double | None = None
    NumberOfInteractions: Double | None = None
    HybridSimulation: Flag | None = None
    SensorOnGrid: Flag | None = None
    TransducerOnGrid: Flag | None = None
    IntegrationDensity: Double | None = None
    Solver: str | None = None
    DeviceNumber: Double | None = None
    SamplingRate: Double | None = None
    GridSize: Double | None = None
    MaxPrime: Double | None = None
    PMLmin: Double | None = None


_SimulationParameters = SimulationParameters


# -------------------------------------------------------------------- Settings

#: The seven settings variables, in the order ``write_to_file`` saves them.
SETTINGS_GROUPS: tuple[str, ...] = (
    "Microbubble",
    "SimulationParameters",
    "Geometry",
    "Transducer",
    "Acquisition",
    "Medium",
    "Transmit",
)


_GROUP_MODELS: dict[str, type[MatStruct]] = {g: globals()[g] for g in SETTINGS_GROUPS}


class Settings(BaseModel):
    """The seven settings structs of one simulation (a settings ``.mat`` file).

    A group is ``None`` when the file does not contain that variable.
    """

    model_config = ConfigDict(validate_assignment=True, arbitrary_types_allowed=True)

    GROUPS: ClassVar[tuple[str, ...]] = SETTINGS_GROUPS

    Microbubble: _Microbubble | None = None
    SimulationParameters: _SimulationParameters | None = None
    Geometry: _Geometry | None = None
    Transducer: _Transducer | None = None
    Acquisition: _Acquisition | None = None
    Medium: _Medium | None = None
    Transmit: _Transmit | None = None

    def to_tree(self) -> dict[str, Any]:
        """``{variable name: struct tree}`` for the groups present."""
        return {g: getattr(self, g).to_tree() for g in self.GROUPS if getattr(self, g) is not None}

    @classmethod
    def from_tree(cls, variables: dict[str, Any]) -> "Settings":
        """Build from ``{variable name: struct tree}`` (other variables are ignored)."""
        kwargs = {}
        for g in cls.GROUPS:
            if g in variables:
                model = _GROUP_MODELS[g]
                kwargs[g] = model.from_tree(variables[g])
        return cls(**kwargs)

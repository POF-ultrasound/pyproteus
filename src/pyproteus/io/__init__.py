"""Input/output: frozen MATLAB reference data (HDF5 format + pooch registry)."""

from .reference import ReferenceDataUnavailable, ReferenceHashMismatch, fetch, load, registry
from .reference_format import (
    FunctionHandle,
    MatlabString,
    Reference,
    ReferenceFormatError,
    StructArray,
    read_reference,
    write_reference,
)

__all__ = [
    "FunctionHandle",
    "MatlabString",
    "Reference",
    "ReferenceDataUnavailable",
    "ReferenceFormatError",
    "ReferenceHashMismatch",
    "StructArray",
    "fetch",
    "load",
    "read_reference",
    "registry",
    "write_reference",
]

"""Tiny PROTEUS-style hybrid k-Wave cases for the tracer bullet.

Arrays are indexed ``[ix, iy, iz]`` (same element positions as MATLAB).
Per-voxel source rows are stored in MATLAB ``find`` (column-major) order,
as PROTEUS builds them (``source.p = source_weights * signals`` with rows
ordered by ``find(source.p_mask)``).

Settings follow PROTEUS' hybrid path (``run_simulation.m``):
``PMLInside=false``, ``Smooth=false``, ``PMLAlpha=2``, ``DataCast='single'``,
``kgrid.dt``/``kgrid.Nt`` set directly, binary sensor mask recording ``p``,
time-varying pressure source (``Transducer.SourceType='monopole'``) or
velocity source (``'dipole'``, curved array: ``ux``/``uy`` along the
surface normal). The medium mimics ``define_medium.m``: heterogeneous sound
speed and density from one background map, nonlinear (``BonA``), power-law
absorption, plus an off-centre "vessel".
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.ndimage import gaussian_filter

SHAPE = (37, 29, 23)  # odd, non-cubic
DX = 1e-4  # [m], isotropic like PROTEUS
DT = 2e-8  # [s], CFL ~ 0.32 at 1600 m/s
NT = 300
PML_SIZE = (10, 12, 9)  # different per axis on purpose
PML_ALPHA = 2.0
F0 = 2.5e6  # [Hz]
P0 = 2e5  # [Pa]


@dataclass
class Case:
    name: str
    shape: tuple[int, int, int]
    dx: float
    dt: float
    Nt: int
    pml_size: tuple[int, int, int]
    pml_alpha: float
    medium: dict[str, np.ndarray | float | str]
    # "p_mask"+"p" and/or "u_mask"+"ux"/"uy"/"uz"; rows in find order
    source: dict[str, np.ndarray | str]
    sensor_mask: np.ndarray
    info: dict = field(default_factory=dict)
    record: tuple[str, ...] = ("p",)  # sensor.record


def _medium(rng: np.random.Generator, shape) -> dict:
    noise = gaussian_filter(rng.standard_normal(shape), sigma=2.0, mode="wrap")
    background = 1.0 + 0.03 * noise / np.abs(noise).max()
    ix, iy, iz = np.indices(shape)
    vessel = ((ix - 22) / 6.0) ** 2 + ((iy - 12) / 4.0) ** 2 + ((iz - 9) / 3.0) ** 2 <= 1.0

    c = 1540.0 * background
    rho = 1000.0 * background
    bona = np.full(shape, 6.0)
    alpha = np.full(shape, 0.5)
    c[vessel] = 1580.0
    rho[vessel] = 1050.0
    bona[vessel] = 5.0
    alpha[vessel] = 0.2
    return {
        "sound_speed": c,
        "density": rho,
        "BonA": bona,
        "alpha_coeff": alpha,
        "alpha_power": 1.5,
    }


def _burst(t: np.ndarray) -> np.ndarray:
    """Gaussian-windowed tone burst centred at 1 us."""
    return np.sin(2 * np.pi * F0 * (t - 1e-6)) * np.exp(-(((t - 1e-6) / 4e-7) ** 2))


def _rows_from_voxels(ijk_f: np.ndarray, rng: np.random.Generator, M: int, steer_deg: float):
    """Distinct delayed, scaled bursts, one per voxel (voxels given in find order)."""
    amp = rng.uniform(0.5, 1.5, size=len(ijk_f))
    delay = (ijk_f[:, 1] * DX * np.sin(np.deg2rad(steer_deg)) / 1540.0) + rng.uniform(
        0, 5 * DT, len(ijk_f)
    )
    t = np.arange(M) * DT
    return P0 * amp[:, None] * _burst(t[None, :] - delay[:, None])


def _voxels_f(mask: np.ndarray) -> np.ndarray:
    """(N, 3) voxel subscripts of mask in MATLAB find order."""
    return np.stack(
        np.unravel_index(np.flatnonzero(mask.ravel(order="F")), mask.shape, order="F"), axis=1
    )


def _sensor(rng, shape, transducer_mask):
    """Transducer voxels + scattered "bubble" voxels (as main_RF.m adds the two masks)."""
    mask = transducer_mask.copy()
    lin = rng.choice(np.prod(shape), size=90, replace=False)
    mb = np.zeros(shape, bool)
    mb.ravel()[lin] = True
    mb[:8] = False  # bubbles in front of the transducer
    return mask | mb


def pressure_case(seed: int = 1) -> Case:
    """Linear-array-like monopole (pressure) source, asymmetric masks."""
    rng = np.random.default_rng(seed)
    shape = SHAPE
    p_mask = np.zeros(shape, bool)
    # elements: 2 voxels wide in y, kerf 1, elevation iz 8..14, at ix = 3
    for iy0 in range(4, 25, 3):
        p_mask[3, iy0 : iy0 + 2, 8:15] = True
    # break the symmetry: drop some voxels, add strays
    drop = rng.choice(np.flatnonzero(p_mask.ravel()), size=6, replace=False)
    p_mask.ravel()[drop] = False
    p_mask[5, 20, 3] = True
    p_mask[2, 27, 18] = True
    p_mask[4, 6, 15] = True

    ijk = _voxels_f(p_mask)
    p = _rows_from_voxels(ijk, rng, M=150, steer_deg=10.0)  # shorter than Nt, like PROTEUS
    sensor_mask = _sensor(rng, shape, p_mask)
    return Case(
        name="pressure",
        shape=shape,
        dx=DX,
        dt=DT,
        Nt=NT,
        pml_size=PML_SIZE,
        pml_alpha=PML_ALPHA,
        medium=_medium(rng, shape),
        source={"p_mask": p_mask, "p": p},
        sensor_mask=sensor_mask,
        info={"n_source": int(p_mask.sum()), "n_sensor": int(sensor_mask.sum())},
    )


def velocity_case(seed: int = 2) -> Case:
    """Curved-array-like dipole (velocity) source: ux/uy along the surface normal."""
    rng = np.random.default_rng(seed)
    shape = SHAPE
    cx, cy, R = -20.0, 14.0, 24.0  # centre of curvature (voxels) and radius
    u_mask = np.zeros(shape, bool)
    phis = np.linspace(-0.5, 0.45, 60)
    for k, phi in enumerate(phis):
        ix = int(np.floor(cx + R * np.cos(phi)))  # voxel containing the point
        iy = int(np.floor(cy + R * np.sin(phi)))
        z0, z1 = 8 + (k % 3 == 0), 14 + (k % 5 == 0)  # ragged elevation extent
        u_mask[ix, iy, z0:z1] = True
    u_mask[6, 25, 4] = True  # stray voxel

    ijk = _voxels_f(u_mask)
    phi = np.arctan2(ijk[:, 1] - cy, ijk[:, 0] - cx)  # outward normal angle per voxel
    sig = _rows_from_voxels(ijk, rng, M=160, steer_deg=-7.0)
    Z = 1540.0 * 1000.0
    ux = sig * np.cos(phi)[:, None] / Z
    uy = sig * np.sin(phi)[:, None] / Z
    sensor_mask = _sensor(rng, shape, u_mask)
    return Case(
        name="velocity",
        shape=shape,
        dx=DX,
        dt=DT,
        Nt=NT,
        pml_size=PML_SIZE,
        pml_alpha=PML_ALPHA,
        medium=_medium(rng, shape),
        source={"u_mask": u_mask, "ux": ux, "uy": uy},
        sensor_mask=sensor_mask,
        info={"n_source": int(u_mask.sum()), "n_sensor": int(sensor_mask.sum())},
    )


def pressure_nodisp_case(seed: int = 1) -> Case:
    """Pressure case with y = 1.05: define_medium.m then sets alpha_mode='no_dispersion'."""
    c = pressure_case(seed)
    c.name = "pressure_nodisp"
    c.medium = {**c.medium, "alpha_power": 1.05, "alpha_mode": "no_dispersion"}
    return c


def velocity_ux_case(seed: int = 3) -> Case:
    """Linear-array-like dipole: ``ux`` only (``define_source_transducer.m``, linear arrays)."""
    c = pressure_case(seed)
    mask = c.source["p_mask"]
    c.name = "velocity_ux"
    c.source = {"u_mask": mask, "ux": c.source["p"] / (1540.0 * 1000.0)}
    return c


def mixed_case(seed: int = 4) -> Case:
    """Velocity transducer (``ux`` + ``uz``) plus pressure point sources at once.

    ``full_simulator.m``: ``update_source.m`` adds bubble pressure sources
    (``p_mask``/``p``) to a dipole transducer source (``u_mask``/``ux``...).
    ``uz`` covers the third velocity component (not used by PROTEUS itself).
    """
    rng = np.random.default_rng(seed)
    c = velocity_case(seed)
    c.name = "mixed"
    u_mask = c.source["u_mask"]
    p_mask = np.zeros(c.shape, bool)
    p_mask.ravel()[rng.choice(p_mask.size, 40, replace=False)] = True
    p_mask[:10] = False  # "bubbles" away from the transducer
    p_mask[25, 3, 20] = True
    ijk = _voxels_f(p_mask)
    c.source = {
        "u_mask": u_mask,
        "ux": c.source["ux"],
        "uz": 0.3 * c.source["uy"][:, ::-1],  # differs from ux in shape
        "p_mask": p_mask,
        "p": 0.1 * _rows_from_voxels(ijk, rng, M=NT, steer_deg=3.0),  # as long as Nt
    }
    return c


def nt_equals_n_sensor_case(seed: int = 5) -> Case:
    """Pressure case with exactly ``Nt`` sensor voxels, recording ``p``, ``p_max``, ``p_min``.

    ``Nt == n_sensor`` makes k-wave-python 0.6.2 permute the time axis of cpp
    output (B-10). ``main_pressure_field.m`` records ``{'p_max','p_min'}``.
    """
    rng = np.random.default_rng(seed)
    c = pressure_case(seed)
    c.name = "nt_eq_n_sensor"
    mask = c.sensor_mask.copy()
    free = np.flatnonzero(~mask.ravel() & (np.indices(c.shape)[0].ravel() >= 8))
    mask.ravel()[rng.choice(free, NT - int(mask.sum()), replace=False)] = True
    assert mask.sum() == NT
    c.sensor_mask = mask
    c.record = ("p", "p_max", "p_min")
    c.info["n_sensor"] = NT
    return c


def pressure_field_case(seed: int = 5) -> Case:
    """``Nt == n_sensor`` recording only ``{'p_max','p_min'}``, as ``main_pressure_field.m``."""
    c = nt_equals_n_sensor_case(seed)
    c.name = "pressure_field"
    c.record = ("p_max", "p_min")
    return c


CASES = {
    "pressure": pressure_case,
    "velocity": velocity_case,
    "pressure_nodisp": pressure_nodisp_case,
    "velocity_ux": velocity_ux_case,
    "mixed": mixed_case,
    "nt_eq_n_sensor": nt_equals_n_sensor_case,
    "pressure_field": pressure_field_case,
}

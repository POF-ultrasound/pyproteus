"""Microbubble model parameters: ports of PROTEUS ``microbubble-simulator/functions``.

* ``getMaterialProperties`` → :func:`get_material_properties`
* ``calc_surface_tension`` → :func:`calc_surface_tension`
* ``calc_thermal_damp`` → :func:`calc_thermal_damp`
* ``find_A_m0``, ``find_domain_boundaries`` → :func:`find_A_m0`, :func:`find_domain_boundaries`
* ``getShellProperties`` → :func:`get_shell_properties`
* ``compress_bubble`` → :func:`compress_bubble`
* ``getEqParam`` → :func:`get_eq_param`

PROTEUS works on MATLAB structs (``liquid``, ``gas``, ``shell``, ``bubble``),
with one ``shell``/``bubble`` struct per microbubble in struct arrays. Here
(D-15, ``docs/PARITY.md``) the liquid and gas are small dataclasses with the
MATLAB field names, and a :class:`Shell` holds a whole batch of bubbles as
arrays (one entry per bubble, or a scalar shared by all), so every function
is vectorised over bubbles instead of looping per struct. Bubble radii are
passed as arrays ``R0`` instead of ``bubble`` structs.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field, fields, replace
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from pyproteus.numerics import GriddedInterpolant, gridded_interpolant, interp1

from .properties import linspace

__all__ = [
    "EqParam",
    "GasProps",
    "Liquid",
    "Shell",
    "calc_surface_tension",
    "calc_thermal_damp",
    "compress_bubble",
    "find_A_m0",
    "find_domain_boundaries",
    "get_eq_param",
    "get_material_properties",
    "get_shell_properties",
    "stack_shells",
]

#: Gas constant used by ``calc_thermal_damp.m`` (J/K/mol).
RG = 8.314
#: Shell viscosity fit ``Ks = c_1 exp(c_2 R0)`` (Segers et al., Soft Matter 14, 2018, Fig. 6B).
KS_C1 = 1.5e-9
KS_C2 = 8e5


# ------------------------------------------------------------------ structs


@dataclass
class Liquid:
    """PROTEUS ``liquid`` struct (MATLAB field names, SI units)."""

    k: float  # thermal conductivity (W/m/K)
    rho: float  # density (kg/m^3)
    cp: float  # specific heat (J/kg/K)
    nu: float  # dynamic viscosity (Pa.s)
    c: float  # speed of sound (m/s)
    T0: float  # temperature (K)
    P0: float  # ambient pressure (Pa)
    sig: float | None = None  # surface tension (N/m); not set by compute_bubble_mass_source
    ThermalModel: str = "Prosperetti"  # 'Prosperetti', 'Adiabatic' or 'Isothermal'
    a: float | None = None  # attenuation (getMaterialProperties only)
    b: float | None = None
    BA: float | None = None  # nonlinearity B/A
    beta: float | None = None


@dataclass
class GasProps:
    """PROTEUS ``gas`` struct (MATLAB field names, SI units)."""

    k: float  # thermal conductivity (W/m/K)
    rho: float  # density (kg/m^3)
    Mg: float  # molar mass (kg/mol)
    gam: float  # heat capacity ratio
    cp: float  # specific heat at constant pressure (J/kg/K)


@dataclass
class Shell:
    """Shell properties of a batch of microbubbles (PROTEUS ``shell`` structs, stacked).

    Array fields have one entry per bubble or are scalars shared by the batch.
    ``model`` is ``'Marmottant'``, ``'Segers'`` (polynomial fit) or
    ``'SegersTable'`` (lookup table, interpolant ``sig``). ``chi`` is ``None``
    when PROTEUS' struct has no ``chi`` field (Segers models): ``getEqParam``
    then estimates it from the slope of the surface tension curve.
    """

    model: str
    sig_0: Any  # initial surface tension (N/m)
    sig_l: Any = None  # maximum surface tension (N/m)
    Ks: Any = None  # surface dilatational viscosity (N.s/m)
    chi: Any = None  # shell elasticity (N/m), Marmottant
    Rb: Any = None  # buckling radius (m), Marmottant
    coeff: np.ndarray | None = None  # polynomial coefficients, Segers
    A_m1: Any = None  # left domain boundary (normalised area), Segers models
    A_m2: Any = None  # right domain boundary
    A_N: Any = None  # reference area (m^2), Segers models
    sig: GriddedInterpolant | None = None  # surface tension table, SegersTable

    def __post_init__(self) -> None:
        for f in ("sig_0", "sig_l", "Ks", "chi", "Rb", "A_m1", "A_m2", "A_N"):
            v = getattr(self, f)
            if v is not None:
                setattr(self, f, np.asarray(v, dtype=np.float64))

    @property
    def size(self) -> int:
        """Number of bubbles (1 if every field is a scalar)."""
        n = 1
        for f in ("sig_0", "sig_l", "Ks", "chi", "Rb", "A_N"):
            v = getattr(self, f)
            if v is not None and v.ndim:
                n = max(n, v.size)
        return n

    def take(self, idx: ArrayLike) -> Shell:
        """Sub-batch ``idx`` (scalar fields are kept)."""
        out = {}
        for f in fields(self):
            v = getattr(self, f.name)
            if isinstance(v, np.ndarray) and v.ndim and f.name != "coeff":
                v = v[idx]
            out[f.name] = v
        return replace(self, **out)


def stack_shells(shells: Shell | list[Shell] | tuple[Shell, ...]) -> Shell:
    """One :class:`Shell` for a batch from per-bubble shells (MATLAB struct array).

    All shells must share the model (and, for the Segers models, the same
    fit); PROTEUS' ``compute_bubble_mass_source`` always builds such batches.
    """
    if isinstance(shells, Shell):
        return shells
    shells = list(shells)
    if not shells:
        raise ValueError("no shells")
    model = shells[0].model
    if any(s.model != model for s in shells):
        raise NotImplementedError("shells of one batch must share the shell model")
    first = shells[0]
    if model == "Segers" and any(not np.array_equal(s.coeff, first.coeff) for s in shells):
        raise NotImplementedError("Segers shells of one batch must share the fit")
    if model == "SegersTable" and any(
        s.sig is not first.sig
        and not (
            np.array_equal(s.sig.grid, first.sig.grid)
            and np.array_equal(s.sig.values, first.sig.values)
        )
        for s in shells
    ):
        raise NotImplementedError("SegersTable shells of one batch must share the table")
    out: dict[str, Any] = {"model": model, "coeff": first.coeff, "sig": first.sig}
    for f in ("sig_0", "sig_l", "Ks", "chi", "Rb", "A_m1", "A_m2", "A_N"):
        vals = [getattr(s, f) for s in shells]
        if all(v is None for v in vals):
            out[f] = None
        elif any(v is None for v in vals):
            raise ValueError(f"field {f} missing in some shells")
        else:
            out[f] = np.array([float(np.asarray(v).reshape(())) for v in vals])
    return Shell(**out)


@dataclass
class EqParam:
    """``getEqParam`` output (one entry per bubble)."""

    nu_th: np.ndarray
    kappa: np.ndarray
    nu_vis: np.ndarray
    nu: np.ndarray
    nu_rad: np.ndarray
    chi: np.ndarray
    omega_0: np.ndarray
    delta_vis: np.ndarray
    delta_shell: np.ndarray
    delta_th: np.ndarray
    delta_rad: np.ndarray
    delta: np.ndarray = field(default=None)  # type: ignore[assignment]


# ------------------------------------------------------------------ material


def get_material_properties() -> tuple[Liquid, GasProps]:
    """Port of ``getMaterialProperties.m``: water at 20 °C and perfluorobutane."""
    liquid = Liquid(
        k=0.6,
        rho=998.0,
        cp=4186.0,
        nu=1e-3,
        c=1480.0,
        sig=0.072,
        T0=293.0,
        P0=1.013e5,
        a=0.002,
        b=2.0,
        BA=5.2,
    )
    liquid.beta = 1 + liquid.BA / 2
    gas = GasProps(k=0.0138, rho=10.1, Mg=0.238, gam=1.07, cp=809.0)
    return liquid, gas


# ------------------------------------------------------------ surface tension


def calc_surface_tension(R: ArrayLike, shell: Shell) -> np.ndarray:
    """Port of ``calc_surface_tension.m``: shell surface tension at radius ``R``.

    ``R`` broadcasts against the shell's per-bubble fields. Behaviour kept from
    PROTEUS: NaN radii give NaN; in the ``'SegersTable'`` model a normalised
    area exactly equal to ``A_m1`` or ``A_m2`` gives 0 (Q-120).
    """
    R = np.asarray(R, dtype=np.float64)
    if shell.model == "Marmottant":
        # Marmottant et al., JASA 118, 2005
        sig = shell.chi * (R**2 / shell.Rb**2 - 1)
        sig = np.where(sig < 0, 0.0, sig)  # buckling
        return np.where(sig > shell.sig_l, shell.sig_l, sig)  # ruptured
    if shell.model == "Segers":
        # Segers et al., Soft Matter 14, 2018: polynomial fit
        A_m = 4 * np.pi * R**2 / shell.A_N
        sig = np.polyval(shell.coeff, A_m)
        sig = np.where(A_m < shell.A_m1, 0.0, sig)
        return np.where(A_m > shell.A_m2, shell.sig_l, sig)
    if shell.model == "SegersTable":
        A_m = 4 * np.pi * R**2 / shell.A_N
        A_m, A_m1, A_m2, sig_l = np.broadcast_arrays(A_m, shell.A_m1, shell.A_m2, shell.sig_l)
        sig = A_m * 0  # buckling (NaN stays NaN)
        sig = np.where(A_m > A_m2, sig_l, sig)
        elastic = (A_m > A_m1) & (A_m < A_m2)
        if elastic.any():
            sig = np.array(sig, dtype=np.float64)
            sig[elastic] = shell.sig(A_m[elastic])
        return sig
    raise ValueError(f"unknown shell model {shell.model!r}")


# ------------------------------------------------------------ thermal model


def calc_thermal_damp(
    liquid: Liquid, gas: GasProps, R0: ArrayLike, sig_0: ArrayLike, w: float
) -> tuple[np.ndarray, np.ndarray]:
    """Port of ``calc_thermal_damp.m`` (Prosperetti, JASA 61, 1977), vectorised.

    Returns ``(nuth, kappa)``: thermal damping (Pa.s) and effective polytropic
    exponent for each radius ``R0`` with initial surface tension ``sig_0``.
    """
    R0 = np.asarray(R0, dtype=np.float64)
    sig_0 = np.asarray(sig_0, dtype=np.float64)
    kl, rhol, T0, Pat, cvl = liquid.k, liquid.rho, liquid.T0, liquid.P0, liquid.cp
    kg, cpg, gam, Mg = gas.k, gas.cp, gas.gam, gas.Mg
    cvg = cpg / gam

    P_eq = Pat + 2 * sig_0 / R0  # equilibrium pressure
    rhog = P_eq / Pat * gas.rho  # equilibrium density (isothermal compression)

    Dg = kg / rhog / cvg  # thermal diffusivity gas
    Dl = kl / rhol / cvl  # thermal diffusivity liquid
    G1 = Mg * Dg * w / gam / RG / T0
    G2 = w * R0**2 / Dg
    G3 = w * R0**2 / Dl

    root = np.sqrt((1j - G1) ** 2 + 4j * G1 / gam)
    beta1 = np.sqrt(1 / 2 * gam * G2 * (1j - G1 + root))  # Eq. 17
    beta2 = np.sqrt(1 / 2 * gam * G2 * (1j - G1 - root))
    lamb1 = beta1 / np.tanh(beta1) - 1  # beta*coth(beta) - 1
    lamb2 = beta2 / np.tanh(beta2) - 1
    Gamma1 = 1j + G1 + root
    Gamma2 = 1j + G1 - root
    f = 1 + (1 + 1j) * np.sqrt(G3 / 2)
    k = kl / kg
    phi = (k * f * (Gamma2 - Gamma1) + lamb2 * Gamma2 - lamb1 * Gamma1) / (
        k * f * (lamb2 * Gamma1 - lamb1 * Gamma2) - lamb1 * lamb2 * (Gamma2 - Gamma1)
    )  # Eq. 16
    nuth = 1 / 4 * w * rhog * R0**2 * np.imag(phi)  # Eq. 14
    kappa = 1 / 3 * (w**2 * rhog * R0**2 / P_eq) * np.real(phi)  # Eq. 15
    return nuth, kappa


# ------------------------------------------------------------- Segers fit


def _real_roots(coeff: np.ndarray) -> np.ndarray:
    """Real roots of a polynomial (MATLAB ``R = roots(c); R(imag(R)==0)``).

    ``numpy.roots`` takes the eigenvalues of the companion matrix, as MATLAB's
    documented ``roots``; real eigenvalues of a real matrix come out with an
    exactly zero imaginary part.
    """
    r = np.roots(np.asarray(coeff, dtype=np.float64).reshape(-1))
    return np.real(r[np.imag(r) == 0])


def find_domain_boundaries(coeff: ArrayLike) -> tuple[float, float]:
    """Port of ``find_domain_boundaries.m``: ``(A_m1, A_m2)`` of the Segers fit.

    ``A_m1``: real root of the fit closest to 0.92; ``A_m2``: real extremum
    closest to 1.12 (first one on a tie, as MATLAB ``min``).
    """
    coeff = np.asarray(coeff, dtype=np.float64).reshape(-1)
    r = _real_roots(coeff)
    A_m1 = float(r[np.argmin(np.abs(r - 0.92))])
    r = _real_roots(np.polyder(coeff))
    A_m2 = float(r[np.argmin(np.abs(r - 1.12))])
    return A_m1, A_m2


def find_A_m0(coeff: ArrayLike, A_m1: float, A_m2: float, sig_0: float) -> float:
    """Port of ``find_A_m0.m``: normalised area where the Segers fit equals ``sig_0``.

    MATLAB returns an empty result when no root lies strictly inside
    ``(A_m1, A_m2)``, which fails later; this port raises ``ValueError`` at once.
    """
    c = np.array(coeff, dtype=np.float64).reshape(-1)
    c[-1] = c[-1] - sig_0
    r = _real_roots(c)
    r = r[(r > A_m1) & (r < A_m2)]
    if r.size > 1:
        raise ValueError("Multiple solutions found.")
    if r.size == 0:
        raise ValueError("no normalised area in the fit domain gives sig_0")
    return float(r[0])


# ------------------------------------------------------------ shell properties


def _ks_fit(R0: np.ndarray) -> np.ndarray:
    return KS_C1 * np.exp(KS_C2 * R0)


def get_shell_properties(R0: ArrayLike, model: str, sig_0: ArrayLike, liquid: Liquid) -> Shell:
    """Port of ``getShellProperties.m`` for bubbles of radii ``R0``.

    ``model`` and ``sig_0`` are the input ``shell.model`` and ``shell.sig_0``.
    The Segers fits come from :mod:`pyproteus.data` (``fit_SigmaR_04-08-2017.mat``
    for ``'Segers'``, ``fit_SigmaR_evaluated.mat`` for ``'SegersTable'``).
    MATLAB warns once per bubble outside 1–4 µm; this port warns once per call.
    """
    from pyproteus import data

    R0 = np.asarray(R0, dtype=np.float64)
    sig0 = np.asarray(sig_0, dtype=np.float64)
    if np.any((R0 < 1e-6) | (R0 > 4e-6)):
        warnings.warn("Shell viscosity uncertain for given microbubble radius.", stacklevel=2)
    if model == "Marmottant":
        sig_l = liquid.sig
    elif model in ("Segers", "SegersTable"):
        sig_l = 0.072
        if sig_l != liquid.sig:
            raise ValueError("Segers shell model only valid in water.")
    else:
        raise ValueError("Shell model not recognized.")
    shell = Shell(model=model, sig_0=sig0, sig_l=sig_l, Ks=_ks_fit(R0))
    if model == "Marmottant":
        shell.chi = np.asarray(0.55)
        shell.Rb = R0 / np.sqrt(1 + shell.sig_0 / shell.chi)
    elif model == "Segers":
        coeff = np.asarray(data.load("fit_SigmaR_04-08-2017.mat")["fit"]["coeff"], float).ravel()
        A_0 = 4 * np.pi * R0**2
        A_m1, A_m2 = find_domain_boundaries(coeff)
        shell.coeff, shell.A_m1, shell.A_m2 = coeff, np.asarray(A_m1), np.asarray(A_m2)
        s0 = np.broadcast_to(shell.sig_0, np.broadcast(R0, shell.sig_0).shape)
        A_m0 = np.array([find_A_m0(coeff, A_m1, A_m2, s) for s in s0.ravel()]).reshape(s0.shape)
        shell.A_N = A_0 / A_m0
    else:  # SegersTable
        fit = data.load("fit_SigmaR_evaluated.mat")
        A_m = np.asarray(fit["A_m"], float).ravel()
        sig = np.asarray(fit["sig"], float).ravel()
        A_0 = 4 * np.pi * R0**2
        shell.A_m1, shell.A_m2 = np.asarray(A_m.min()), np.asarray(A_m.max())
        A_m0 = interp1(sig, A_m, shell.sig_0)
        shell.A_N = A_0 / A_m0
        shell.sig = gridded_interpolant(A_m, sig)
    return shell


# ------------------------------------------------------------ compression


def compress_bubble(
    liquid: Liquid, gas: GasProps, R0: float, shell: Shell, P1: float
) -> tuple[Liquid, GasProps, float, Shell]:
    """Port of ``compress_bubble.m``: new equilibrium after an overpressure ``P1`` (Pa).

    Returns new ``(liquid, gas, R0, shell)`` (inputs are not modified):
    ambient pressure ``P0 + P1``, equilibrium radius, surface tension and
    isothermally compressed gas density. One bubble (scalar ``R0`` and shell).
    """
    kappa = 1  # isothermal
    P0 = liquid.P0
    sig_0 = float(np.asarray(shell.sig_0).reshape(()))
    R = R0 * linspace(0.5, 2, 10_000)[0]  # search array
    sig = calc_surface_tension(R, shell)
    P_in = (P0 + 2 * sig_0 / R0) * (R / R0) ** (-3 * kappa)
    P_eq = P0 + 2 * sig / R + P1
    R_new = _find_intersections(P_in, P_eq, R)
    if R_new.size == 0:
        raise ValueError("Overpressure too low or too high to find solution.")
    if R_new.size > 1:
        warnings.warn(
            "Multiple solution found. Selecting new value of R0 closest to old value of R0.",
            stacklevel=2,
        )
        R_new = R_new[np.argmin(np.abs(R_new - R0))]
    R_new = float(np.asarray(R_new).reshape(-1)[0])
    sig_new = float(np.asarray(calc_surface_tension(R_new, shell)).reshape(()))
    return (
        replace(liquid, P0=P0 + P1),
        replace(gas, rho=gas.rho * (P0 + P1) / P0),
        R_new,
        replace(shell, sig_0=np.asarray(sig_new)),
    )


def _find_intersections(y2: np.ndarray, y1: np.ndarray, x: np.ndarray) -> np.ndarray:
    """``x`` where ``y2 == y1``: sign changes of ``y2 - y1``, linearly interpolated."""
    d = y2 - y1
    idx = np.flatnonzero(np.diff(np.sign(d)) != 0)
    x_solve = np.array([float(interp1(d[m : m + 2], x[m : m + 2], 0.0)) for m in idx])
    return np.unique(x_solve)  # sorted, duplicates removed (MATLAB unique)


# ---------------------------------------------------------- equation params


def get_eq_param(liquid: Liquid, gas: GasProps, shell: Shell, R0: ArrayLike, w: float) -> EqParam:
    """Port of ``getEqParam.m``, vectorised over the bubbles of ``shell``/``R0``.

    ``w`` is the angular frequency of the pulse (``pulse.w``).
    """
    R0 = np.asarray(R0, dtype=np.float64)
    nu_l, rhol, c, P0 = liquid.nu, liquid.rho, liquid.c, liquid.P0
    sig_0, Ks = shell.sig_0, shell.Ks
    model = liquid.ThermalModel
    if model == "Prosperetti" and w == 0:
        raise ValueError("Prosperetti model only valid for w>0")
    shape = np.broadcast(R0, sig_0).shape
    if model == "Prosperetti":
        nu_th, kappa = calc_thermal_damp(liquid, gas, R0, sig_0, w)
    elif model == "Adiabatic":
        nu_th, kappa = np.zeros(shape), np.full(shape, float(gas.gam))
    elif model == "Isothermal":
        nu_th, kappa = np.zeros(shape), np.ones(shape)
    else:
        raise ValueError("Thermal model not recognized")
    nu_th = np.broadcast_to(nu_th, shape).astype(np.float64)
    kappa = np.broadcast_to(kappa, shape).astype(np.float64)
    nu_vis = np.full(shape, float(nu_l))
    nu = nu_vis + nu_th
    x = w * R0 / c
    nu_rad = rhol * R0**2 / 4 * (x / (1 + x**2)) * w  # not added: RadiationDamping = false

    if shell.chi is None:
        epsilon = 0.001
        sig1 = calc_surface_tension(R0 * (1 - epsilon), shell)
        sig2 = calc_surface_tension(R0 * (1 + epsilon), shell)
        chi = (sig2 - sig1) / epsilon / 4
    else:
        chi = shell.chi
    chi = np.broadcast_to(chi, shape).astype(np.float64)

    # van der Meer et al., JASA 121, 2007
    omega_0 = np.sqrt(
        1 / (rhol * R0**2) * (3 * kappa * (P0 + 2 * sig_0 / R0) - 2 * sig_0 / R0 + 4 * chi / R0)
    )
    delta_vis = 4 * nu_l / (rhol * R0**2 * omega_0)
    delta_shell = 4 * Ks / (rhol * R0**3 * omega_0)
    delta_th = 4 * nu_th / (rhol * R0**2 * omega_0)
    delta_rad = 1 / (rhol * R0 * omega_0) * (P0 + 2 * sig_0 / R0) * 3 * kappa / c
    delta = delta_rad + delta_th + delta_vis + delta_shell
    b = lambda a: np.broadcast_to(a, shape).astype(np.float64)  # noqa: E731
    return EqParam(
        nu_th=nu_th,
        kappa=kappa,
        nu_vis=nu_vis,
        nu=b(nu),
        nu_rad=b(nu_rad),
        chi=chi,
        omega_0=b(omega_0),
        delta_vis=b(delta_vis),
        delta_shell=b(delta_shell),
        delta_th=b(delta_th),
        delta_rad=b(delta_rad),
        delta=b(delta),
    )

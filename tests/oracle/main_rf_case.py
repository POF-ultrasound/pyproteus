"""Shared builder for ``main_RF`` cases run through MATLAB PROTEUS and pyproteus.

Used by ``tests/oracle/test_main_rf.py`` and the M3 study scripts in
``tools/m3/``. A case is a MATLAB settings struct (MainGUI defaults, shrunk to a
tiny grid: few narrow elements, low points per wavelength, no vessel, a small
bounding box) plus a MATLAB-written ground truth (``FlowSimulationParameters.mat``
and ``Frame_<m>.mat`` with bubbles inside the bounding box).

MATLAB runs ``main_RF`` unchanged with ``PROTEUS_LOCAL_PATHS`` pointing every
writable folder into the case folder, k-Wave MATLAB from ``KWAVE_MATLAB_PATH``
and the D-10 binary wrapper; the medium's random draws come from
``rng(seed, 'twister')``, which pyproteus reproduces with ``MatlabRNG(seed)``.
"""

from __future__ import annotations

import dataclasses
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tests.oracle.test_acoustic_grid import build_settings, update_settings


@dataclass
class Case:
    """Settings of one tiny ``main_RF`` case (MATLAB field values)."""

    name: str
    n_el: int = 4
    element_height: float = 1.5e-3
    ppw: float = 3.0
    start_depth: float = 3e-3
    diagonal: tuple[float, float, float] = (3e-3, 2e-3, 1e-3)
    hybrid: bool = True
    n_interactions: int = 1
    scheme: str = "Standard"
    n_frames: int = 2
    n_mb: int = 5
    seed: int = 7
    inhomogeneity: float = 0.02
    bonA: float | None = None  # None: MainGUI default (6)
    alpha_power: float | None = None  # None: default (1.5)
    points: np.ndarray | None = None  # fixed bubble positions (N x 3, bounding-box frame)
    radii: np.ndarray | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def n_seq(self) -> int:
        return {"Standard": 1, "Pulse inversion": 2}.get(self.scheme, 3)


def matlab_settings(eng: Any, oracle_path: Path, case: Case, var: str = "S") -> None:
    """Build the settings struct ``var`` of ``case`` in the MATLAB workspace."""
    build_settings(eng, oracle_path, var)
    d = case.diagonal
    code = f"""
    {var}.Transducer.NumberOfElements = {case.n_el};
    {var}.Transducer.ElementHeight = {case.element_height!r};
    {var}.Geometry.EmbedVessel = false;
    {var}.Geometry.Rotation = eye(3);
    {var}.Geometry.BoundingBox.Center = [0; 0; 0];
    {var}.Geometry.BoundingBox.Diagonal = [{d[0]!r}; {d[1]!r}; {d[2]!r}];
    {var}.Geometry.startDepth = {case.start_depth!r};
    {var}.SimulationParameters.PointsPerWavelength = {case.ppw!r};
    {var}.SimulationParameters.NumberOfInteractions = {case.n_interactions};
    {var}.SimulationParameters.HybridSimulation = {str(case.hybrid).lower()};
    {var}.SimulationParameters.Solver = '3DC';
    {var}.Acquisition.NumberOfFrames = {case.n_frames};
    {var}.Acquisition.PulsingScheme = '{case.scheme}';
    {var}.Acquisition.NumberOfPulses = {case.n_seq};
    {var}.Microbubble.Number = {case.n_mb};
    {var}.Medium.Inhomogeneity = {case.inhomogeneity!r};
    """
    if case.bonA is not None:
        code += f"{var}.Medium.BonA = {case.bonA!r};\n"
    if case.alpha_power is not None:
        code += f"{var}.Medium.AttenuationB = {case.alpha_power!r};\n"
    eng.eval(textwrap.dedent(code), nargout=0)
    update_settings(eng, var)


def write_ground_truth(eng: Any, folder: Path, case: Case, var: str = "S") -> None:
    """``FlowSimulationParameters.mat`` and ``Frame_<m>.mat`` for ``case`` (MATLAB ``-v7``).

    Bubbles: uniform in the central 80 % of the bounding box (bounding-box
    frame, centred on its ``Center``), radii from the settings' size
    distribution support (uniform), zero velocity; fixed ``case.points`` /
    ``case.radii`` replace the draws in every frame and pulse.
    """
    folder.mkdir(parents=True, exist_ok=True)
    fixed = case.points is not None
    if fixed:
        eng.workspace["pyp_P__"] = np.asarray(case.points, dtype=float)
        eng.workspace["pyp_R__"] = np.asarray(case.radii, dtype=float).reshape(-1, 1)
    n = len(case.points) if fixed else case.n_mb
    code = f"""
    pyp_s__ = RandStream('mt19937ar','Seed',{case.seed + 1000});
    FlowSimulationParameters = struct('TimeBtwPulse', {var}.Acquisition.TimeBetweenPulses, ...
        'FrameRate', {var}.Acquisition.FrameRate, 'NBPulses', {case.n_seq}, ...
        'NMicrobubbles', {var}.Microbubble.Number, 'NumberOfFrames', {case.n_frames});
    FlowSimulationParameters.Microbubble.Distribution = {var}.Microbubble.Distribution;
    save(fullfile('{folder}','FlowSimulationParameters.mat'),'FlowSimulationParameters','-v7');
    pad = num2str(length(num2str({case.n_frames})));
    D = {var}.Geometry.BoundingBox.Diagonal(:)'; C = {var}.Geometry.BoundingBox.Center(:)';
    Rs = {var}.Microbubble.Distribution.Radii;
    for m = 1:{case.n_frames}
        Frame = struct();
        for k = 1:{case.n_seq}
            p = ['Pulse' num2str(k)];
            if {int(fixed)}
                Frame.(p).Points = pyp_P__; Frame.(p).Radius = pyp_R__;
            else
                Frame.(p).Points = C + 0.8*(rand(pyp_s__, {n}, 3) - 0.5).*D;
                Frame.(p).Radius = min(Rs) + (max(Rs)-min(Rs))*rand(pyp_s__, {n}, 1);
            end
            Frame.(p).Velocity = zeros({n}, 3);
            Frame.(p).StreamNumber = (1:{n})';
        end
        save(fullfile('{folder}',['Frame_' num2str(m,['%0' pad 'i']) '.mat']),'Frame','-v7');
    end
    clear pyp_s__ pyp_P__ pyp_R__ FlowSimulationParameters pad D C Rs m k p Frame
    """
    eng.eval(textwrap.dedent(code), nargout=0)


def write_local_paths(root: Path, kwave_matlab: Path, binary_dir: Path, oracle: Path) -> Path:
    """``PROTEUS_LOCAL_PATHS`` file: every writable PROTEUS folder under ``root``."""
    for name in ("results", "data", "gt", "settings"):
        (root / name).mkdir(parents=True, exist_ok=True)
    f = root / "proteus_local_paths.m"
    f.write_text(
        textwrap.dedent(f"""\
        PATHS_LOCAL.kWavePath       = '{kwave_matlab}';
        PATHS_LOCAL.BinaryPath      = '{binary_dir}';
        PATHS_LOCAL.GeometriesPath  = '{oracle / "geometry_data"}';
        PATHS_LOCAL.GroundTruthPath = '{root / "gt"}';
        PATHS_LOCAL.ResultsPath     = '{root / "results"}';
        PATHS_LOCAL.DataPath        = '{root / "data"}/';
        PATHS_LOCAL.SettingsPath    = '{root / "settings"}';
        """)
    )
    return f


def run_matlab_main_rf(
    eng: Any, settings_file: Path, gt: Path, save: Path, local_paths: Path, seed: int
) -> list[Any]:
    """MATLAB ``main_RF(settings_file, gt.name, save.name)`` after ``rng(seed, 'twister')``.

    ``gt`` must be in ``<root>/gt`` and ``save`` in ``<root>/results`` (``sim_startup.m``
    joins the folder names to ``GroundTruthPath``/``ResultsPath``). Returns the
    frames as :class:`pyproteus.io.frames.RFFrame` objects.
    """
    from pyproteus.io.frames import read_matlab_rf_frame

    saved = eng.path(nargout=1)
    old_env = eng.getenv("PROTEUS_LOCAL_PATHS", nargout=1)
    try:
        eng.setenv("PROTEUS_LOCAL_PATHS", str(local_paths), nargout=0)
        eng.eval(
            f"rng({seed}, 'twister'); warning('off','all');"
            f"pyp_log__ = evalc(\"main_RF('{settings_file}', '{gt.name}', '{save.name}')\");"
            "warning('on','all'); stop(timerfindall); delete(timerfindall);",
            nargout=0,
        )
    finally:
        eng.setenv("PROTEUS_LOCAL_PATHS", old_env, nargout=0)
        eng.path(saved, nargout=0)
        eng.eval("clear pyp_log__", nargout=0)
    files = sorted(Path(save).glob("Frame_*.mat"))
    return [read_matlab_rf_frame(f) for f in files]


def build(eng: Any, oracle_path: Path, root: Path, case: Case) -> tuple[Path, Path]:
    """Settings file and ground-truth folder of ``case`` under ``root``."""
    matlab_settings(eng, oracle_path, case)
    settings_file = root / f"{case.name}.mat"
    eng.eval(f"save('{settings_file}', '-struct', 'S');", nargout=0)
    gt = root / "gt" / case.name
    write_ground_truth(eng, gt, case)
    eng.eval("clear S", nargout=0)
    return settings_file, gt


def transmit_only_rf(
    settings_file: Path, gt: Path, *, binary_path: Any = None, binary_env: Any = None, seed: int = 0
) -> list[np.ndarray]:
    """pyproteus RF of the transducer's own transmit (no bubbles), one array per pulse.

    The RF of ``main_RF`` is this plus the bubble echoes; errors are reported
    against ``max |RF - transmit_only_rf|`` (the bubble part), which the
    transmit cross-talk would otherwise hide.
    """
    from pyproteus.acoustic import compute_RF_data, define_source_transducer, run_simulation
    from pyproteus.acoustic.main import _nt, _with_seq_pulse, prepare_simulation
    from pyproteus.acoustic.setup import setup_run
    from pyproteus.numerics import MatlabRNG

    setup = setup_run(settings_file, gt, gt.parent / "_transmit_only", binary_path=binary_path)
    setup.settings.Medium.Save = False
    prep = prepare_simulation(setup, rng=MatlabRNG(seed))
    rp = prep.run_param
    kgrid = dataclasses.replace(prep.kgrid, Nt=_nt(rp["tr"][2], prep.grid.dt))
    out = []
    for pulse in setup.sequence:
        src = define_source_transducer(
            prep.transducer,
            _with_seq_pulse(prep.transmit, pulse),
            setup.settings.Medium,
            prep.grid,
            prep.sensor_weights.T,
            prep.mask_idx_trans,
        )
        sd = run_simulation(
            rp, kgrid, prep.medium, src, prep.sensor_transducer, binary_env=binary_env
        )
        rf, rp = compute_RF_data(prep.transducer, sd, prep.sensor_weights, prep.grid, rp)
        out.append(np.asarray(rf, dtype=float))
    return out


def run_matlab_transmit_only(
    eng: Any, settings_file: Path, local_paths: Path, seed: int
) -> list[np.ndarray]:
    """MATLAB counterpart of :func:`transmit_only_rf` (``shims/transmit_only_rf.m``)."""
    saved = eng.path(nargout=1)
    old_env = eng.getenv("PROTEUS_LOCAL_PATHS", nargout=1)
    try:
        eng.setenv("PROTEUS_LOCAL_PATHS", str(local_paths), nargout=0)
        eng.eval(
            "addpath(fileparts(fileparts(which('main_RF'))));"  # path_setup.m (sim_startup does it)
            "warning('off','all');"
            f"pyp_log__ = evalc(\"pyp_T__ = transmit_only_rf('{settings_file}', {seed});\");"
            "warning('on','all');",
            nargout=0,
        )
        cells = eng.workspace["pyp_T__"]
    finally:
        eng.setenv("PROTEUS_LOCAL_PATHS", old_env, nargout=0)
        eng.path(saved, nargout=0)
        eng.eval("clear pyp_log__ pyp_T__", nargout=0)
    return [np.asarray(c, dtype=float) for c in cells]


def bubble_errors(
    ref: list[Any], py: list[Any], transmit_py: list[np.ndarray], transmit_ref: list[np.ndarray]
) -> np.ndarray:
    """Error of the bubble part per frame and pulse.

    ``max|(RF_m - T_m) - (RF_p - T_p)| / max|RF_p - T_p|`` with ``T`` the
    transmit-only RF of each implementation.
    """
    err = np.zeros((len(ref), len(transmit_py)))
    for m, (r, q) in enumerate(zip(ref, py, strict=True)):
        for k in range(len(transmit_py)):
            a = np.asarray(r.rf[k], dtype=float) - transmit_ref[k]
            b = np.asarray(q.rf[k], dtype=float) - transmit_py[k]
            err[m, k] = np.max(np.abs(a - b)) / np.max(np.abs(b))
    return err


def total_errors(ref: list[Any], py: list[Any]) -> np.ndarray:
    """``max|RF_matlab - RF_py| / max|RF_matlab|`` per frame and pulse (D-24 as written)."""
    return np.array(
        [
            [
                np.max(np.abs(np.asarray(a, float) - np.asarray(b, float)))
                / np.max(np.abs(np.asarray(a, float)))
                for a, b in zip(r.rf, q.rf, strict=True)
            ]
            for r, q in zip(ref, py, strict=True)
        ]
    )

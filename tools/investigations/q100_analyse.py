"""Q-100 investigation: analyse the point-scatterer runs of ``q100_point_scatterer.m``.

For each geometry, the RF of the 1 µm run minus the 0.5 µm run keeps only the
bubble echo (both runs share every deterministic clutter term). Per element,
the envelope peak of that difference is located on the DAS time axis
(``initialize_das_parameters``) with a parabolic sub-sample fit and compared
with the geometric two-way time. Prints the residual in ns, simulation
samples and transmit samples, and what each candidate time-axis term would
change.

Usage: python tools/investigations/q100_analyse.py <out_dir>
Investigation only; not part of the package.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import scipy.io as sio

from pyproteus.numerics import conv, hilbert


def load(path: Path):
    return sio.loadmat(path, squeeze_me=True, struct_as_record=False)


def analyse(out: Path) -> None:
    pairs = sorted({p.stem.rsplit("_", 1)[0] for p in out.glob("*_r05.mat")})
    for base in pairs:
        a, b = load(out / f"{base}_r10.mat"), load(out / f"{base}_r05.mat")
        r, Sx, tg = a["recon"], a["Sx"], np.atleast_1d(a["t_geom"])
        t, Fs = r.t, r.Fs
        D = a["RF"].astype(float) - b["RF"].astype(float)
        E = np.abs(hilbert(D, axis=1))
        res, snr = [], []
        for e in range(D.shape[0]):
            win = np.flatnonzero(np.abs(t - tg[e]) <= 1.5e-6)
            i = int(win[np.argmax(E[e, win])])
            y = E[e, i - 1 : i + 2]
            d = 0.5 * (y[0] - y[2]) / (y[0] - 2 * y[1] + y[2])
            res.append(t[0] + (i + d) / Fs - tg[e])
            far = (np.abs(t - tg[e]) > 1.5e-6) & (t > 0)
            snr.append(E[e, i] / E[e, far].max())
        res = np.array(res)
        T, Tr, SP = Sx.Transmit, Sx.Transducer, Sx.SimulationParameters
        c = Sx.Medium.SpeedOfSound
        V = conv(T.PressureSignal, Tr.ReceiveImpulseResponse)
        I = int(np.argmax(np.abs(hilbert(V)))) + 1
        dt, dx = 1 / SP.SamplingRate, SP.GridSize
        kw = dx / (2 * c) + (dt if SP.HybridSimulation else 1.5 * dt)
        print(
            f"{base}: residual {res.mean() * 1e9:+.2f} ns (std {res.std() * 1e9:.2f}, "
            f"echo/clutter >= {min(snr):.1f}) = {res.mean() * Fs:+.3f} sim samples = "
            f"{res.mean() * T.SamplingRate:+.2f} transmit samples | "
            f"I/Fs_tx = {I / T.SamplingRate * 1e9:.1f} ns "
            f"(1 sample = {1e9 / T.SamplingRate:.1f} ns), k-Wave correction {kw * 1e9:.2f} ns, "
            f"dt {dt * 1e9:.2f} ns, dx/2c {dx / 2 / c * 1e9:.2f} ns"
        )


if __name__ == "__main__":
    analyse(Path(sys.argv[1]))

#!/bin/sh
# Create the runtime libraries for k-wave-python's macOS arm64 k-Wave binary
# (kspaceFirstOrder-OMP v1.4.1 links /opt/homebrew/opt/{fftw,hdf5,libomp,zlib}).
# Puts them in an isolated, gitignored conda prefix next to this script, so
# neither Homebrew nor the shared `pyproteus` env is modified. Versions pinned:
# the binary needs libhdf5.320 (hdf5 2.2.x on conda-forge) and libfftw3f{,_omp}.3. If Homebrew's
# `brew install fftw hdf5 zlib libomp` is present, the binary uses that instead.
set -e
here=$(cd "$(dirname "$0")" && pwd)
"${CONDA_EXE:-$HOME/miniconda3/bin/conda}" create -y -p "$here/.kwave-libs" \
    -c conda-forge --override-channels "fftw=3.3.11" "hdf5=2.2.0" "zlib=1.3.2" llvm-openmp

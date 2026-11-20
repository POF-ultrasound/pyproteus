#!/bin/sh
# Create the runtime libraries for k-wave-python's macOS arm64 k-Wave binary
# (kspaceFirstOrder-OMP v1.4.1 links /opt/homebrew/opt/{fftw,hdf5,libomp,zlib}).
# Puts them in an isolated, gitignored conda prefix next to this script, so
# neither Homebrew nor the shared `pyproteus` env is modified. If Homebrew's
# `brew install fftw hdf5 zlib libomp` is present, the binary uses that instead.
set -e
here=$(cd "$(dirname "$0")" && pwd)
"${CONDA_EXE:-$HOME/miniconda3/bin/conda}" create -y -p "$here/.kwave-libs" \
    -c conda-forge --override-channels fftw llvm-openmp hdf5 zlib

# pyproteus

Python port of [PROTEUS](https://github.com/PROTEUS-SIM/PROTEUS), the simulator for contrast-enhanced ultrasound (microbubbles, k-Wave propagation, RF data and delay-and-sum images).

> **Status: early development.** The port is under way and not usable yet. Until 1.0, pyproteus aims to reproduce MATLAB PROTEUS v1.1.0 results; see [docs/ROADMAP.md](docs/ROADMAP.md).

## Goals

- Same behaviour and results as MATLAB PROTEUS v1.1.0 at 1.0 (checked against MATLAB as a test oracle), then Python-only development.
- Runs on CPU on Linux, Windows and macOS, and on NVIDIA GPUs (CUDA).
- Usable as a Python package, from the command line and through a GUI (after 1.0).

## Development setup

```bash
conda env create -f environment.yml
conda activate pyproteus
pip install -e ".[dev]"
pre-commit install
pytest -m "not oracle"
```

The MATLAB oracle tests also need MATLAB R2025a and `pip install "matlabengine==25.1.*"`; they are skipped without MATLAB. Contributors never need MATLAB. See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT, like PROTEUS. See [LICENSE](LICENSE).

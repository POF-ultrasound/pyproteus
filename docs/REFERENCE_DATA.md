# Reference data

Frozen outputs of MATLAB PROTEUS (oracle commit `8d37938`, D-7) that let CI
check parity without MATLAB (ROADMAP §3, §4 layer 2). Exported by
`tools/export_reference/` through the MATLAB Engine, published on Zenodo,
fetched with pooch by `pyproteus.io.reference`. The files are never committed:
git holds only their names and SHA-256 hashes
(`src/pyproteus/io/reference_registry.txt`).

## Files

One HDF5 file per case, flat names (Zenodo records have no folders):

| Kind | Name | Content |
|---|---|---|
| `module` | `<PROTEUS folder>.<function>[.<variant>].h5`, e.g. `GUIfunctions.reset_transmit.h5` | one call of one oracle function: its inputs and outputs |
| `full_run` | `full_run.<case>.h5` (`tiny`, `small`, `medium`) | `main_RF` on the real k-Wave C++ binary: settings, ground truth, RF, k-Wave medium |

## Layout

```
/                     attributes: format, provenance (below)
/inputs               struct: the function's arguments, in call order, as MATLAB received them
/outputs              struct: its results
/extra                struct (optional): anything else worth keeping (full runs: medium.mat)
```

Every node carries `matlab_class` and `matlab_size` attributes.

| MATLAB value | HDF5 node | Python (`read_reference`) |
|---|---|---|
| numeric / logical array | dataset, dims = `size(x)`, dtype of the class (complex: compound `r`,`i`) | `np.ndarray`, `shape == size(x)` (≥ 2-D; a scalar is `(1, 1)`) |
| sparse (double / logical) | full dataset + `matlab_sparse = True` | `scipy.sparse.csc_array` |
| char row vector | UTF-8 string dataset, `matlab_size` `[1 n]` (`[0 0]` for any empty char) | `str` |
| string scalar | UTF-8 string dataset, class `string` | `MatlabString` (a `str`) |
| function handle | `func2str` text, class `function_handle` | `FunctionHandle` (a `str`) |
| scalar struct | group, `matlab_fields` = field order, one child per field | `dict` (field order kept) |
| struct array (not 1×1) | group, children `"0"`, `"1"`, … = elements by **linear index** | `StructArray` (`elements` object array of shape `size`) |
| cell array | group, children `"0"`, `"1"`, … by **linear index** | `np.ndarray(dtype=object)` of shape `size` |

Not supported (the export fails rather than guess): char matrices, string
arrays, objects, tables, containers.Map. Convert them in the export spec.

### Array orientation (`array_layout = "matlab-size"`)

The dataset's dimensions are MATLAB's `size(x)`, in that order, and element
`x(i,j,k)` is stored at HDF5 index `[i-1, j-1, k-1]`. h5py therefore returns an
array with MATLAB's shape and MATLAB's indexing (0-based):

```python
a = read_reference(path).outputs["y"]   # a.shape == size(y)
a[i, j, k] == y(i+1, j+1, k+1)
a.ravel(order="F") == y(:)              # MATLAB linear order
```

The bytes on disk are a C-order (row-major) array of shape `size(x)`. This is
the transpose of what MATLAB's own `h5write`/`-v7.3` files contain (they store
column-major bytes with reversed dimensions): **read these files with
pyproteus or h5py, not with MATLAB's `h5read`** (it would return `permute(y,
ndims(y):-1:1)`). Trailing singleton dimensions follow MATLAB's `size`
(`size` never ends in 1 beyond the second dimension). 1-D arrays do not
exist in the format: a row vector is `(1, n)`, a column `(n, 1)`.

Values are stored as MATLAB produced them. In particular, MATLAB indices
(`find`, `sub2ind`, sensor mask indices…) stay **1-based**, and a linear index
`k` addresses `a.ravel(order="F")[k - 1]`. Cell and struct-array elements are
numbered by linear index, so element `"n"` of a 2×3 cell is `c{n+1}` in MATLAB
and `c[np.unravel_index(n, (2, 3), order="F")]` in Python.

The layout is checked from both sides by `tests/oracle/test_reference_export.py`:
arrays built in MATLAB with a known column-major numbering (`reshape(1:24,2,3,4)`)
must read back as `1 + i + 2j + 6k`, and every value read from the file
(arrays, cells, structs, struct arrays, text, sparse), rebuilt in MATLAB, must
equal the original node by node in class, size, sparsity and data.

## Provenance attributes (root)

All are required (`write_reference` refuses a file without them; empty string
where not applicable).

| Attribute | Meaning |
|---|---|
| `format`, `format_version`, `array_layout` | `pyproteus-reference`, `1`, `matlab-size` |
| `name`, `kind`, `module`, `function`, `description` | what the file is (`kind`: `module` or `full_run`) |
| `call` | the exact MATLAB statement evaluated (inputs in `pyp_in`, results in `pyp_out`) |
| `matlab_version` | `version()` of the MATLAB that ran it |
| `proteus_version`, `oracle_commit`, `oracle_dirty` | `proteus_version()`, full commit hash of the oracle checkout, whether it had local changes (the exporter refuses a dirty or wrong-commit oracle unless told `PYPROTEUS_ORACLE_COMMIT=any`) |
| `kwave_matlab_version`, `kwave_matlab_path` | MATLAB k-Wave (from `Contents.m`) on the path, if any |
| `kwave_solver` | `kspaceFirstOrder3DC` for full runs |
| `kwave_binary_path`, `kwave_binary_version`, `kwave_binary_sha256` | the C++ binary actually executed (D-10), its `--version` build line, its hash; required for `full_run` |
| `seeds` | JSON object: every seed set (`matlab_rng_twister`: `rng(seed,'twister')` before building the inputs; full runs add the ground-truth `RandStream` seed and the `rng` seed before `main_RF`) |
| `matlab_path_extra` | every MATLAB path entry outside `matlabroot` and the oracle's root/module folders, in search order (`<pyproteus>/…`, `<oracle>/…` abbreviated): export helpers, k-Wave, oracle `tests/helpers`, user folders such as `~/Documents/MATLAB`, shims of a reused session |
| `created_utc` | ISO 8601 UTC timestamp |
| `exporter_version`, `exporter_commit` | pyproteus version and commit of the exporter (`-dirty` if modified) |
| `platform` | OS, release, machine of the export |

Further attributes written by the exporter: `matlab_parallel_pool_workers`
(open pool size; PROTEUS uses `parfor` for bubbles when a pool exists),
`numpy_version`; for k-Wave runs `kwave_matlab_sha256` (hash of all `.m` files
of the MATLAB k-Wave toolbox), `kwave_runtime` (JSON: CPU count, `KWAVE_LIBS`,
`OMP_NUM_THREADS`, library path variables, Homebrew libraries present on macOS,
which take precedence over `KWAVE_LIBS` in the wrapper), `geometries_path`,
and, when `BinaryPath` holds a wrapper script (macOS arm64),
`kwave_binary_wrapper` and `kwave_binary_wrapper_sha256`; with k-wave-python's
binary `kwave_python_version` and `kwave_python_binary_release`.

## Full-run cases

Settings: `tools/export_reference/matlab/pyp_full_run_settings.m`, built with the
oracle's own headless helpers (`tests/helpers/make_tiny_settings.m`…). Synthetic
ground truth (`write_synthetic_ground_truth`, seed 1), `rng(0,'twister')`
before `main_RF` (the medium background is random), solver `3DC`.

| Case | Settings | Runtime |
|---|---|---|
| `tiny` | PROTEUS regression case `rf_hybrid_standard`: `tiny_box`, 8 elements, 4 ppw, 4 bubbles, 2 frames | seconds |
| `small` | `tiny_box`, 16 elements, 6 ppw, 12 bubbles, 3 frames, pulse inversion | minutes |
| `medium` | default settings on `renal_tree`, 2 frames | GPU workstation; not exported yet (needs the renal tree STL from Zenodo 10573662) |

The tiny case exported on the developer's Mac matches PROTEUS' own regression
baseline (`tests/baseline/rf_hybrid_standard.mat`, recorded with the MATLAB
solver instead of the binary) to 1.4e-6 relative.

Binary (D-10): on macOS arm64 MATLAB `kspaceFirstOrder3DC` runs k-wave-python's
`kspaceFirstOrder-OMP` v1.4.1 through `tools/tracer_bullet/bin/`; its runtime
libraries come from `tools/tracer_bullet/setup_libs.sh` (or Homebrew).

## Using reference data in tests

```python
import pytest
from pyproteus.io import reference


def test_reset_transmit_frozen():
    try:
        r = reference.load("GUIfunctions.reset_transmit")
    except reference.ReferenceDataUnavailable as exc:
        pytest.skip(str(exc))
    expected = r.outputs["Transmit"]
    ...
```

| Environment variable | Effect |
|---|---|
| `PYPROTEUS_REFERENCE_DIR` | read files from this folder (e.g. a fresh export), no download; registered files must match their hash |
| `PYPROTEUS_REFERENCE_VERIFY=0` | skip that hash check (while iterating on an export) |
| `PYPROTEUS_REFERENCE_URL` | base URL overriding the registry's `# base_url:` (`https://…/` or `doi:10.5281/zenodo.<id>/`) |
| `PYPROTEUS_REFERENCE_CACHE` | download cache (default: OS cache folder `pyproteus/reference`) |

## Versioning

- Bump `format_version` for any incompatible change of the layout; the reader
  refuses other versions.
- New oracle commit (e.g. the `v1.1.0` tag) or new k-Wave binary: re-export
  everything, new Zenodo version, update the registry in one PR.
- Each Zenodo version holds the complete set of files; the registry names the
  version's DOI as `base_url`.

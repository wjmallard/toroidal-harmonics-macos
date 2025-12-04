# Toroidal Harmonics — macOS/Apple Silicon Port

A macOS port of [Vassili Savinov's toroidal harmonics library](https://github.com/vasily-savinov/TORH_GIT), which provides Python bindings to Fortran routines for computing toroidal harmonics (associated Legendre functions of half-integer degree for arguments > 1).

The underlying Fortran code is by Segura and Gil: *Computer Physics Communications* 124 (2000) 104–122, "Evaluation of toroidal harmonics."

For Windows, use the original repository.

## Requirements

- macOS on Apple Silicon
- Homebrew GCC (for gfortran): `brew install gcc`
- Python 3.x, NumPy, SciPy

## Build

```bash
cd DTORH64
bash build.sh

bash run_test.sh  # optional, verifies the build
bash run_testpro.sh  # optional, verifies the build
```

This produces `wrapDTORH64.dylib`.

## Usage

Copy `wrapDTORH64.dylib` into `ToroidalHarmonics/`, then:

```python
from DTORH import DTORH

with DTORH() as dtorh:
    P, Q = dtorh.FixedM(z_values, m=2, n=100)
```

See `ToroidalHarmonics/DemoTorHarmRep.py` for a full example.

## What's Included

- `DTORH64/` — Fortran source and C wrappers for the core computation
- `ToroidalHarmonics/` — Python package for harmonic decomposition of scalar and vector fields in toroidal coordinates

## License

The original repository has no explicit license. This port is provided for convenience; please respect the original authors' rights.

## Changes from Original

- macOS/Apple Silicon build scripts (replacing Windows batch files and DLLs)
- Updated Python code for NumPy 1.24+ compatibility (`np.int` → `np.int32`, etc.)
- Removed Windows-specific ctypes calls

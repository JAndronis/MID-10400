# EXtra-geom — Codebase Summary

**Repo:** [European-XFEL/EXtra-geom](https://github.com/European-XFEL/EXtra-geom)
**Version reviewed:** 1.16.0 | **License:** BSD-3-Clause | **Python:** ≥3.10

## What it does

EXtra-geom is European XFEL's Python library for handling **detector geometry** and **assembling multi-module detector images**. EuXFEL's large-area detectors (AGIPD, LPD, DSSC, JUNGFRAU, ePix, pnCCD) are built from independent modules/tiles that each record a separate data stream. To turn that raw per-module data into a physically meaningful image (or to do geometry-aware analysis without assembling an image at all), you need to know where every tile sits in 3D space relative to the beam axis. That bookkeeping — reading/writing geometry files, computing pixel positions, and reassembling frames — is the library's whole job.

Core capabilities:
- **Load/save geometry** from CrystFEL `.geom` files, HDF5 quadrant-position files, or simple analytic descriptions (`from_crystfel_geom`, `from_quad_positions`, `from_h5_file`, `from_simple_description`).
- **Assemble images** from per-module arrays via three strategies with different speed/accuracy trade-offs:
  - `position_modules` / `position_modules_fast` — snap pixels to a regular grid (fast, slightly lossy).
  - `position_modules_interpolate` — pixel-splitting reassembly via pyFAI's Distortion module (slower, conserves signal).
  - `position_modules_symmetric` — output centred/padded symmetrically about the beam.
- **Geometry queries/edits**: `get_pixel_positions`, `data_coords_to_positions`, `offset`, `rotate`, `compare` (diff two geometries visually).
- **Interop**: export to pyFAI `Detector` objects (`to_pyfai_detector`), export distortion arrays (`to_distortion_array`), read/write CrystFEL `.geom` format, and (for AGIPD) track motor-stage-based geometry changes over a run.
- **Visualization**: `inspect()` (quick module-layout plot) and `plot_data()` (assembled-image plot) built on matplotlib.

It has no I/O dependency on raw EuXFEL run data itself — it operates purely on numpy arrays of module data plus geometry metadata — which is what lets it compose with `EXtra-data` (data access), `pyFAI` (azimuthal integration/distortion), and `CrystFEL` (crystallography) in the wider EuXFEL analysis stack.

## Package layout

```
extra_geom/
├── __init__.py        # public API surface (see below)
├── base.py             # DetectorGeometryBase, GeometryFragment — shared logic
├── detectors.py        # one class per detector type (largest module, ~2570 lines)
├── snapped.py          # SnappedGeometry / GridGeometryFragment — grid-snap assembly backend
├── motors.py           # BaseMotorTracker + AGIPD_1MMotors, JF4MMotors — motor-driven geometry updates
├── pyfai.py            # thin pyFAI Detector subclasses per instrument
├── crystfel_fmt.py     # CrystFEL .geom file read/write helpers
├── lpd_old.py          # legacy LPD geometry format support
└── tests/              # per-detector pytest suites + example .h5 geometry files
```

**Architecture pattern:** a single abstract base (`DetectorGeometryBase` in `base.py`) implements all the generic geometry math (pixel positions, image assembly, rotation/offset, CrystFEL I/O, pyFAI export). Each concrete detector in `detectors.py` subclasses it and supplies only what's detector-specific: module/tile counts, pixel size, tile layout within a module, and one or more `from_*` constructors matching how that instrument's geometry is normally calibrated/stored (quadrant positions for AGIPD/LPD, HDF5 quad files, simple analytic description for the generic case, etc.). This keeps the assembly/rotation/export logic written once and shared across all seven-plus detector types.

## Exposed API layer

Everything public is re-exported from the package root (`extra_geom/__init__.py`), so users do `from extra_geom import AGIPD_1MGeometry` etc. rather than reaching into submodules:

```python
__all__ = [
    'AGIPD_1MGeometry', 'AGIPD_500K2GGeometry', 'agipd_asic_seams',
    'GenericGeometry', 'DSSC_1MGeometry', 'JUNGFRAUGeometry',
    'LPD_1MGeometry', 'LPD_MiniGeometry', 'PNCCDGeometry',
    'Epix100Geometry', 'Epix10KGeometry',
]
```

### Detector geometry classes (the main API)
Each is a `DetectorGeometryBase` subclass for a specific instrument:

| Class | Detector | Notes |
|---|---|---|
| `AGIPD_1MGeometry` | AGIPD-1M | 16 modules × 8 tiles; `from_quad_positions`, `from_crystfel_geom` |
| `AGIPD_500K2GGeometry` | AGIPD-500K2G | 8 modules × 8 tiles; `from_origin` |
| `LPD_1MGeometry` | LPD-1M | quad-position and HDF5-based constructors |
| `LPD_MiniGeometry` | LPD Mini | variable module count |
| `DSSC_1MGeometry` (+ `DSSC_1MGeometryCartesian`, `DSSC_Geometry`) | DSSC-1M | hexagonal-pixel handling, Cartesian remapping variant |
| `JUNGFRAUGeometry` | JUNGFRAU | `from_module_positions` |
| `PNCCDGeometry` | pnCCD | |
| `Epix100Geometry` / `Epix10KGeometry` | ePix100 / ePix10K | share `EpixGeometryBase` |
| `GenericGeometry` | arbitrary/custom layouts | `from_simple_description` for detectors without a dedicated class |

### Common methods inherited from `DetectorGeometryBase` (available on every detector class above)
- **Construction/IO:** `from_crystfel_geom`, `write_crystfel_geom`, plus detector-specific `from_*`/`example()` factory methods
- **Assembly:** `position_modules`, `position_modules_fast`, `position_modules_symmetric`, `position_modules_interpolate`, `output_array_for_position`
- **Geometry math:** `get_pixel_positions`, `data_coords_to_positions`, `offset`, `rotate`
- **Export:** `to_distortion_array` (pyFAI-compatible distortion map), `to_pyfai_detector`
- **Visualization:** `inspect`, `plot_data`, `compare`

### Supporting/advanced modules (used less directly, mostly by power users)
- `agipd_asic_seams()` — standalone function giving ASIC boundary masks for AGIPD.
- `motors.BaseMotorTracker`, `AGIPD_1MMotors`, `JF4MMotors` — recompute geometry from motor-stage positions logged during a run (useful when detector modules physically move mid-experiment).
- `snapped.SnappedGeometry` — the object returned internally by the grid-snap assembly path; not usually constructed directly by users.
- `pyfai.py` — pre-configured pyFAI `Detector` subclasses per instrument (`AGIPD1M`, `DSSC1M`, `LPD1M`, `JUNGFRAU_EuXFEL`, etc.), used internally by `to_pyfai_detector()`.
- `crystfel_fmt.py` — internal CrystFEL `.geom` serialization helpers used by `from_crystfel_geom`/`write_crystfel_geom`.

### Dependencies
Core: `numpy`, `h5py`, `matplotlib`, `cfel_fmt`. Optional: `pyFAI` (required only for `position_modules_interpolate` and pyFAI export). Test extras additionally pull in `xarray` and `EXtra-data`, reflecting its intended use alongside those packages in the broader EuXFEL analysis ecosystem.

---
*Given your work with pyBeamtime and the EXtra library ecosystem at MID, the `DetectorGeometryBase` / per-instrument-subclass pattern here is a clean reference point if you're weighing similar abstract-base-plus-concrete-readers designs for your own framework.*

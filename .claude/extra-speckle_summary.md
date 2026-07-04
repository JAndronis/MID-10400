# EXtra-speckle — Codebase Summary (v0.2, from local clone)

**Source:** `/Users/iasonandronis/Downloads/extra-speckle` (local clone of `git.xfel.eu/dataAnalysis/extra-speckle`)
**Package version:** 0.2 · **License:** BSD-3 · **Authors:** A. Leonau, J. Möller, J. Wrigley, T. Guest, M. B. Jakobsen (all EuXFEL)

> This replaces the earlier version of this report, which was reconstructed from
> GitLab's issue tracker and search snippets because the web UI is a JS SPA that
> blocks scraping. Everything below is read directly from source, so treat this
> as ground truth rather than inference — a few details in the earlier version
> turn out to have been slightly off (noted where relevant).

## What it does

EXtra-speckle is EuXFEL's Python package for reducing coherent (and some
non-coherent) X-ray scattering data into physical observables. Per the package
`README`, it covers **XPCS**, **XSVS**, and **XCCA**, plus non-coherent
**SAXS/XRD**. In practice, XPCS is by far the most complete and production-ready
part; XSVS is functional but newer; XCCA is real but still has an unfinished
public-facing layer (see maturity notes below).

It's built specifically around EuXFEL's burst-mode timing structure (up to
2700 pulses/train at 4.5 MHz, 10 Hz train rate) and the AGIPD 1M detector at
MID, with geometry handled via `extra_geom` and azimuthal integration via
`pyFAI`. It's the same package documented in Leonau et al., *"A pipeline for
Megahertz XPCS on soft matter samples at MID"* (arXiv:2506.08668) — that paper
is cited directly in `docs/index.md` as the reference publication.

## Package layout

```
extra_speckle/
├── __init__.py          # exposes: pipeline, utils, xpcs, stats, xsvs, xcca, saxs
├── setup/                # experiment/detector/geometry configuration
│   ├── setup.py          #   Setup, BaseSetup classes
│   ├── detector.py       #   DetectorAGIPD1M/65k, DetectorJungfrau500k, DetectorEPix100,
│   │                     #   DetectorDSSC1M, DetectorPoni, DetectorGeneric
│   ├── configuration.py  #   ConfigSAXS, ConfigWAXS, ConfigLFoV (pyFAI AzimuthalIntegrator setup)
│   └── tools.py          #   LFoV geometry-center refinement
├── stats/                # per-pixel-cell mean intensity + outlier/train filtering
│   ├── calc_stats.py     #   calc_stats_agipd, chunk_trains_agipd, pulse-coord helpers
│   ├── get.py            #   get_stats (load-from-cache-or-compute wrapper)
│   └── tools.py          #   mask_outlier, mask_intensity_outliers
├── saxs/                 # azimuthal integration (I(q))
│   ├── get.py, plot.py, tools.py (fit_saxs_beamcenter)
├── xpcs/                 # two-time correlation / g2
│   ├── get.py            #   get() — in-memory TTCF for a single array
│   ├── calc_xpcs.py      #   calc_xpcs_agipd() — full-run, parallelized, cached
│   ├── kernel.py         #   TTCFCalculator, TTCFSparseCalculator
│   ├── fit.py, fit_dr.py #   g2 fitting + diffusion-relation (Stokes-Einstein) fit
│   ├── plot.py           #   plot() (dispatches to plot_ttcf / plot_g2), plot_kbar
│   ├── tools.py          #   reduce_ttcf, kbar_filter, baseline_subtraction_and_averaging, g2_to_bins
│   └── rolling.py        #   rolling-bunch-pattern XPCS (MID-specific, non-standard timing)
├── xsvs/                 # photon-counting statistics / speckle contrast
│   ├── calc_xsvs.py, kernel.py (PhotonCalculator), fit.py (binomial-gamma model), plot.py
│   └── get.py            #   ⚠ stub — see below
├── xcca/                 # angular cross-correlation
│   ├── calc_xcca.py, corr.py (get_angular_cc via FFT), polar.py, kernel.py, plot.py
└── utils.py              # open_run, photon energy/wavelength, AGIPD geometry-from-encoders, misc
```

## Core concept: the `Setup` object

Almost everything in the package takes a `setup` object as an argument. It's
built once per experiment/run and bundles three things (this is documented
explicitly in `docs/setup.md`, and matches the code exactly):

1. **Detector** — one of `AGIPD 1M`, `AGIPD 65k` (single module), `Jungfrau 500k`,
   `ePix100`, `DSSC 1M`, or a `.poni`-file / fully generic detector. Each has a
   dedicated class in `setup/detector.py` wrapping the right `extra_geom`
   geometry and exposing it as a `pyFAI` detector object.
2. **Configuration** — `SAXS` (fit2D beam-center convention), `WAXS` (PONI
   convention, detector rotated around the sample), or `LFoV` (large field of
   view, short sample-detector distance, AGIPD-1M/65k only — geometry entirely
   from the `.geom` file, no separate rotation). Each builds and owns a
   `pyFAI.AzimuthalIntegrator` as `setup.config.ai`.
3. **General experimental metadata** — photon energy/wavelength, repetition
   rate, sample-detector distance, threshold, q-range/q-bin ROI masks.

`Setup.__init__` pulls sdd, rep-rate, twotheta, etc. from the EuXFEL run
metadata automatically when possible (`utils.get_metadata_from_detector`,
`utils.geometry_from_encoders` — the latter literally reads AGIPD quadrant
motor encoder positions from `MID_AGIPD_MOTION/MDL/DOWNSAMPLER`), with manual
> `.poni`/`.geom` file > metadata as the override priority, exactly as
documented.

Useful `Setup` methods: `add_rect_roi`, `calc_q_rois` / `set_qrange_default`
(q-bin mask generation), `update_geom` / `update_beamcenter` (for iterative
refinement), `get_assembled_image`, `plot_data`, `show_qroi`,
`show_image_qroi`.

## Exposed API layer

### Top level
```python
from extra_speckle import pipeline, utils, xpcs, stats, xsvs, xcca, saxs
from extra_speckle.setup import Setup
```

### Per-technique `.get` / `.fit` / `.plot` convention
This is real and intentional — `docs/key-concepts.md` calls it the "unified,
intuitive API," and it's implemented consistently for SAXS and XPCS:

```python
from extra_speckle import saxs, xpcs

I_q = saxs.get(data, setup)          # azimuthal integration -> xr.DataArray
saxs.plot(I_q)

TTCF = xpcs.get(data, setup, data_off=data_off)   # in-memory TTCF for one array
g2   = xpcs.tools.reduce_ttcf(TTCF, rep_rate=setup.rep_rate)
g2_fit, params, errors = xpcs.fit(g2, model='exp', beta='local', dr=True)
xpcs.plot(g2, fit=g2_fit)
```

Maturity varies by technique:
- **SAXS**: `saxs.get` / `saxs.plot` fully implemented (`extra_speckle/saxs/__init__.py` exports both cleanly).
- **XPCS**: fully implemented — `.get`, `.fit`, `.fit_dr`, `.plot`, plus the low-level `TTCFCalculator`/`TTCFSparseCalculator` kernels and `calc_xpcs_agipd` for full-run processing.
- **XSVS**: `.get` (from `xsvs.get.get`) is a **placeholder stub** — its docstring literally says "Placeholder: This counts the number of 0, 1, 2, and 3s..." and the function body is just a `print()`. The real computation lives in `calc_xsvs_agipd` (full pipeline) instead. `.plot` and `.fit` exist as real modules (binomial-gamma contrast fitting) but are **not re-exported** in `xsvs/__init__.py` (commented out) — you have to import `extra_speckle.xsvs.fit` / `.plot` directly.
- **XCCA**: `xcca/__init__.py` has `.plot`, `.fit`, `.get` **all commented out** — there is currently no clean public entry point for this technique. The real logic (`calc_xcca_agipd`, `get_angular_cc`, polar-coordinate mapping) exists and works, and is called from the `pipeline` module directly, but if you want to use XCCA standalone you need to reach into `extra_speckle.xcca.calc_xcca` / `.corr` yourself.

### `xpcs.get(data, setup, return_format='merged', data_off=None, return_kbar=False, **kwargs)`
In-memory TTCF calculation for a single array already loaded (numpy, xarray,
or `scipy.sparse.csr_matrix`). Handles AGIPD1M's native `(16,512,128)` shape
transparently. Returns an `xr.DataArray` (or a tuple with off-correlation /
kbar if requested). This is the one you'd call directly if you're
prototyping on a small chunk of data rather than running the full-run
pipeline.

### `xpcs.calc_xpcs_agipd(run, setup, ...)` — the heavy lifter
Full-run, parallelized, disk-cached TTCF computation. Key parameters:
`selected_pulses`, `pulse_coord` (`'idx'|'pulseId'|'cellId'`),
`good_trains_mask`, `sparse_thrs=0.05`, `photonize`, `sparsify=True`,
`extra_filtering`/`filters`, `workers_no=50`, `ram_limit=300`, `cache_dir`.
Internally: builds q-bin masks → estimates per-module sparse-vs-dense density
→ `multiprocessing.Pool` over module/sequence HDF5 files (`calc_single_module`)
→ caches per-module `.npy` results to disk → collects and merges into a single
`output_xpcs_AGIPD.h5` → converts to xarray (`convert_TTCF`), writing
`TTCF_raw.nc`, `TTCF_off.nc`, `kbar.nc`. This matches the arXiv paper's
pipeline description almost verbatim (mean-density check → sparsify threshold
5×10⁻² → default 50 workers, RAM-limited).

A `calc_single_module_deprecated` variant is still in the file (old, slower
per-train indexing via `np.where`), left in place but unused by the current
call path — worth knowing if you ever find yourself in that code path by
mistake.

### `xpcs.fit(g2, model='exp', beta='local'|'global', dr=False, plot=False, ...)`
Fits g₂ to a single exponential: `g = β·exp(-2·t·Γ) + baseline`, with `beta`
either fit per-q-bin (`'local'`) or shared across all q-bins (`'global'`).
**Confirms what the issue tracker suggested**: stretched-exponential (KWW,
`model='str'`) is scaffolded but not implemented — both branches just
`print('This model is not implemented yet')`. If `dr=True`, it also calls
`xpcs.fit_dr` to extract a diffusion coefficient / hydrodynamic radius from
Γ(q) via a weighted-average or `curve_fit` linear fit of Γ vs. q² (Stokes-Einstein).

### `xsvs.calc_xsvs_agipd(run, setup, ...)`
Photon-counting-statistics pipeline: per-q-bin histograms of pixels with 1/2/3
photons (via `PhotonCalculator`), normalized into probabilities P(k) (with P(0)
added back in), then contrast β fit per q-bin using a **binomial-gamma
(negative-binomial) model** (`xsvs/fit.py`) via `scipy.optimize.minimize`
(default Nelder-Mead), objective `'nll'` (negative log-likelihood) or `'mse'`
(adjusted-residual MSE), with optional kbar/probability/physical-boundary
filtering. Produces both single-shot and train-averaged contrast estimates,
plus an independent analytical cross-check (`beta_estimator`, from the
P(2)/P(1) ratio). Photon counting currently caps at k=3 (`PhotonCalculator`
only bins counts of exactly 1, 2, 3) — fine for the low-photon-density regime
XSVS targets, but worth knowing if you need higher orders.

### `xcca.calc_xcca_agipd(run, setup, npt_azim, npt_rad, azimuth_range, radial_range, ...)`
Polar-coordinate (azimuthal vs. radial) remapping of each frame via
`pyFAI.integrate_radial`/`integrate2d`, then an FFT-based angular
cross-correlation (`corr.get_angular_cc`, Wiener–Khinchin via `np.fft.rfft`)
per train/pulse. Uses `pasha` for parallel mapping across trains; there's a
GPU-worker code path (`process_chunk_gpu`/`gpu_worker` with a `use_gpu_target`
flag) alongside the CPU path, but it's wired through queues/threads that look
still experimental — I'd sanity-check this path carefully before relying on
it for a real analysis. There's also a fully commented-out
`calc_single_train` function suggesting an earlier, simpler implementation
that was superseded.

### `pipeline.xpcs_offline(...)` (alias: `saxs_xpcs_offline`)
The single "run everything" entry point, and the one used for actual beamtime
analysis. It's a long function (~50 keyword arguments) that chains, in order:
`open_run` → `Setup(...)` → `stats.get_stats` (mean pixel-cell/intensity,
cached to disk) → optional beam-center refinement
(`saxs.tools.fit_saxs_beamcenter`) → optional outlier masking
(`stats.tools.mask_outlier`) → optional intensity-outlier train filtering
(`stats.tools.mask_intensity_outliers`) → `xpcs.calc_xpcs_agipd` → kbar
thresholding → baseline subtraction/averaging → g₂ extraction/plotting →
optional g₂ fitting → optional `saxs.get` → optional `xsvs.calc_xsvs_agipd` →
optional `xcca.calc_xcca_agipd`. Returns a dict with every intermediate
array and every diagnostic figure it made along the way. Currently
**hard-restricted to `detector == 'AGIPD 1M'`** (raises `RuntimeError`
otherwise) — the multi-detector support in `setup/detector.py` isn't yet
wired into the pipeline orchestration layer.

Note: the XSVS branch inside `xpcs_offline` references
`force_xsvs`, `k_slice`, `photon_orders`, `weights`, `minimisation_options`,
`minimisation_method`, `diagnostics` as if they were parameters of
`xpcs_offline` itself, but they aren't declared in its signature — this looks
like an in-progress wiring gap between the pipeline and the newer XSVS fitting
machinery, so `run_xsvs_analysis=True` will currently raise a `NameError`
until that's patched.

## Under the hood: TTCF kernels

`xpcs/kernel.py` has two classes with an identical interface:
- **`TTCFCalculator`** — dense-array TTCF via `self._mimg.dot(self._mimg.T)`.
- **`TTCFSparseCalculator`** — same math on `scipy.sparse.csr_matrix` inputs.

Both compute, per q-bin mask: `ttc_on` (the raw numerator of the two-time
correlation), `s_on`/`s_cnt` (per-frame intensity sum / pixel count for
normalization), and `ttc_off`/`s_off` for the adjacent-frame off-correlation
used to subtract detector artifacts — this is the exact TTCF/off-correlation
scheme described in the arXiv methods paper (Eq. 3–6 there), just at the
single-module, single-chunk level; `calc_xpcs_agipd` is what stitches this
across all 16 AGIPD modules and all trains and does the final normalization
(`convert_TTCF`).

## Dependencies

From `pyproject.toml`:
```
numpy, scipy, pyFAI, extra_data, extra_geom, clusterfutures
```
plus (used in code but not pinned in core deps — presumably assumed present
via the Maxwell `exfel-python` environment): `xarray`, `h5py`,
`matplotlib`, `pasha` (used by `xcca`), `euxfel_bunch_pattern` (used by
`xpcs.rolling`). Docs extras: `mkdocs-material`, `mkdocs-jupyter`,
`mkdocstrings[-python]`, `pymdown-extensions`.

Recommended install path (from `docs/getting-started.md`): on Maxwell, just
`module load exfel exfel-python` and select the `xfel (current)` Jupyter
kernel — the package is preinstalled there. Manual install elsewhere is
`pip install git+https://git.xfel.eu/dataAnalysis/extra-speckle.git`.

## Documentation & examples

Real mkdocs site (source in `docs/`, built via `.gitlab-ci.yml` → GitLab
Pages), with genuinely well-written conceptual pages (`index.md`,
`key-concepts.md`, `techniques.md`, `setup.md`, `pipeline.md`,
`detectors.md`) plus `mkdocstrings`-generated API reference pages
(`docs/api/api_{pipeline,saxs,setup,stats,utils,xpcs,xsvs}.md` — note there's
no `api_xcca.md`, consistent with XCCA's API not being finalized).
`docs/notebooks/` has real example notebooks per detector/geometry
combination: AGIPD1M (SAXS + WAXS), AGIPD65k (WAXS), Jungfrau500k (SAXS +
WAXS), EPix100 (WAXS), JF1M (WAXS), plus a combined
`use_extra_speckle_XPCS_AGIPD1M_SAXS_JF500K_WAXS.ipynb` and a
`setup_offline_pipeline_SAXS_AGIPD.ipynb` walking through the pipeline call.
`docs/user-publications.md` exists but is currently empty.

## Things worth knowing before you build on this

- **No automated tests.** `.gitlab-ci.yml` only builds and deploys the docs
  site — there's no test job at all, consistent with the open "Unit Tests for
  EXtra-speckle" issue never having been closed.
- **`xpcs_offline` only supports AGIPD 1M** right now, despite `Setup`
  supporting five detector types — if you want Jungfrau/DSSC/ePix through the
  full pipeline (rather than the lower-level `.get`/`.fit`/`.plot` calls) you'd
  need to extend the pipeline function yourself.
- **XSVS wiring gap** in `pipeline.xpcs_offline` (see above) — `run_xsvs_analysis=True` looks like it will currently fail.
- **`xsvs.get()` is a stub**, `xcca` has no re-exported public API — both are
  usable via their internal modules, just not through the clean `.get/.fit/.plot`
  surface yet.
- **KWW/stretched-exponential fitting for XPCS g₂ is not implemented** (`model='str'` is a no-op) — if you want that for your double-KWW g₂ work, you'd still need your own fitting code or to extend `xpcs/fit.py`.
- Several `_deprecated` functions are left in place alongside their
  replacements (`calc_single_module_deprecated` in both `xpcs` and `stats`,
  `reduce_ttcf_rolling_deprecated`) — harmless, but grep for `deprecated` if
  you're ever unsure which code path is live.

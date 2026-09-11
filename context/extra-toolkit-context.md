# EXtra (European-XFEL/EXtra) — project context
 
Reviewed against `master` @ `5fae330` (2026-09-04) plus the published docs at
<https://extra.readthedocs.io>. Repo: <https://github.com/European-XFEL/EXtra>.
Written for the ferritin-crystallisation-in-levitated-droplets project (AGIPD SAXS/WAXS
at MID, p010400) with XPCS and XCCA as the target analyses.
 
---
 
## 1. What EXtra is
 
Umbrella package (`euxfel-EXtra`, import name `extra`) that (a) re-exports the other
EuXFEL libraries under one namespace, (b) wraps Karabo devices in high-level
"components", (c) holds a small set of technique-specific "applications", (d) provides
CalCat access.
 
```python
from extra.data   import open_run     # == extra_data
from extra.geom   import AGIPD_1MGeometry  # == extra_geom
from extra.damnit import Damnit       # == damnit
from extra.components import XGM, XrayPulses, AGIPD1M   # own + extra_data.components
```
 
`extra.components` also does `from extra_data.components import *`, so `AGIPD1M` etc.
are importable from either place.
 
**Install / environment**
 
- Maxwell: `module load exfel exfel-python` (the `xfel (current)` Jupyter kernel).
  This env is **redeployed from `master` nightly** by a cron job, not from tags.
  Consequence: EXtra behaviour on Maxwell can change between two runs of the same
  notebook. Pin with `pip install git+https://github.com/European-XFEL/EXtra.git@<sha>`
  in a personal env if a result needs to be reproducible.
- PyPI: `pip install euxfel-extra`. Requires Python ≥ 3.11.
- Deps of note: `extra_data>=1.13`, `extra_geom`, `extra_proposal`, `damnit`,
  `euxfel_bunch_pattern`, `pasha`, `pint`, `lmfit`, `xarray`, `scikit-learn`.
- Versions: `2026.1`, `2026.1.1` released; XCCA is currently **Unreleased** (i.e. on
  Maxwell already, not in any tag).
- Units: `extra.ureg` is a shared `pint` registry with an `xfel` context that converts
  `[length] <-> [energy]`. Several APIs return `pint.Quantity` by default.
---
 
## 2. `extra.applications.xcca` — XCCA toolbox (the headline find)
 
**This is directly one of our two target analyses and is easy to miss**: it is not
listed in the mkdocs `nav`, so the page `docs/applications/xcca.md` does not appear on
the published site (only reachable via the Applications index link). It is also *not*
re-exported from `extra.applications/__init__.py`. Correct import:
 
```python
from extra.applications.xcca import (
    AngularCorrelator,
    AveragedAngularCorrelation, AveragedAngularCorrelationMasked,
    CumulativeVariance, CumulativeVarianceMasked,
)
```
 
Tests exist: `tests/test_applications_xcca.py`.
 
### API and conventions
 
`AngularCorrelator(n_radial_samples=256, n_angular_samples=1024, use_cuda=False)`
 
- Input is **`I(q, φ)` already on a uniform polar grid**, shape `(n_q, n_phi)`.
  The module does *not* do Cartesian→polar regridding, geometry, or masking of the
  detector. That step is ours (pyFAI 2D `integrate2d`, or `extra_geom` pixel positions).
- `ccf(data, mask=None, max_order=None)` → `C(q1, q2, φ)`, shape `(n_q, n_q, n_phi)`;
  with a mask it returns `(ccf, ccf_mask)`.
- `ccn(data, mask=None, max_order=None)` → Fourier coefficients `C_n(q1, q2)`, complex,
  shape `(n_q, n_q, max_order+1)`; with a mask returns `(ccn, ccn_mask)`.
- `ccn_from_ccf` / `ccf_from_ccn` convert between the two.
- Mask convention: **masked = 0, unmasked = 1** (boolean array). Correction is
  `CCF(data·mask) / CCF(mask)`, per *J. Appl. Crystallogr.* **57**, 324 (2024), eq. 16.
  Bins where the mask autocorrelation falls below `1/(2·n_phi)` are declared undefined
  and returned in the companion `*_mask`.
- Symmetry used throughout: `C(q1,q2,φ) = C(q2,q1,−φ)`, hence `C_n(q1,q2) = C_n(q2,q1)*`.
  Only the upper triangle is computed.
- All FFTs use `norm='forward'`.
- `max_order` **does not save compute time**, only RAM (the docstring says so explicitly).
### Averaging classes
 
`AveragedAngularCorrelation` (constant/no mask) and `AveragedAngularCorrelationMasked`
accumulate mean **and variance** of the CCF/CCn over patterns using Welford's online
algorithm, with per-element counts in the masked case. `.update(I[, mask])`,
`.merge(other)`, `.mean`, `.variance`, `.count`. `bessels_correction=False` and
`no_data_to_nan=True` by default.
 
The masked class does mask correction on the fly, so the full `(n_q, n_q, n_phi)` CCF
never has to be stored when only `C_n` up to `max_order` is wanted. The class docstring
notes the manual route (accumulate CCF, correct once at the end) is ~2× faster but needs
the full CCF in memory. For a per-shot-mask experiment the masked class is the only
sane option; for a fixed mask, the manual route is worth benchmarking.
 
### Gotchas / defects found (2026-09)
 
1. **`_CumulativeVarianceBase.from_dataset` calls `obj.update(*args)` twice per sample.**
   Every point is counted twice. Mean and population variance are unchanged (which is
   why `test_variance_same_as_naive_computation` passes), but `.count` is 2× too large,
   `bessels_correction=True` gives the wrong denominator, and any subsequent `.merge()`
   is weighted wrongly. Affects `CumulativeVariance.from_dataset` and
   `CumulativeVarianceMasked.from_dataset`. The `AveragedAngularCorrelation*`
   subclasses override `from_dataset` and update once, so they are fine.
   → Use `.update()` in a loop, or the `Averaged*` classes, not the base `from_dataset`.
   Worth an upstream issue.
2. **Docstring examples import from paths that do not exist** (`extra.utils.xcca`,
   `EXtra.utils.xcca`). The real path is `extra.applications.xcca`.
3. **`use_cuda=True` looks untested.** It swaps in `cupyx.scipy.fft`, but the internal
   workspaces (`ccf_workspace`, `ccn_workspace`, `mask_workspace`) are still allocated
   with `np.empty` and passed as `out=`. Verify on a GPU node before relying on it.
4. **Memory.** At the defaults (256, 1024) a single `ccf` is
   256·256·1024·8 B ≈ **537 MB**, plus workspaces. Prefer `ccn` with a modest
   `max_order`, and size `n_q`/`n_phi` deliberately.
5. **Not thread-safe.** Workspaces live on the instance; give each worker its own
   `AngularCorrelator`. (Relevant if we reuse the `pasha`-based parallelism pattern from
   `analysis_helpers.py`.)
6. `n_angular_samples` should divide the angular sampling cleanly; the mask threshold
   `1/(2·n_phi)` assumes the mask CCF only takes multiples of `1/n_phi`.
---
 
## 3. Pulse patterns — `extra.components.pulses`
 
Directly fixes the "`pulseId` coordinate is really an index" defect in our current
DAMNIT integration, and gives real lag times for XPCS.
 
`XrayPulses(run, source=None, sase=None)` (FEL pulses from the timeserver / pulse
pattern decoder). Siblings: `OpticalLaserPulses`, `PumpProbePulses`, `MachinePulses`,
`ManualPulses`, `DldPulses`. Shared interface (`PulsePattern`):
 
| Call | Gives |
|---|---|
| `pulse_ids(labelled=True)` | pulse IDs indexed by `(trainId, pulseIndex)` |
| `peek_pulse_ids()` | same for the first train only — much faster |
| `build_pulse_index(pulse_dim='pulseId')` | a `MultiIndex` to attach to pulse-resolved arrays; `pulse_dim` ∈ {`pulseId`, `pulseIndex`, `pulseTime`} |
| `pulse_counts()` | pulses per train |
| `pulse_periods()` | period per train in units of 4.5 MHz |
| `pulse_repetition_rates()` | Hz per train |
| `train_durations()` | first→last pulse, seconds |
| `is_constant_pattern()` | whether pulse IDs are identical in every train |
| `search_pulse_patterns()` | contiguous train regions of constant pattern |
| `select_pulses` / `deselect_pulses` / `union` | → `ManualPulses` |
| `select_trains` / `split_trains` | same semantics as `DataCollection` |
| `inspect()`, `plot_xray_patterns()` | visualise the pattern |
| `bunch_repetition_rate` | 4.5 MHz (master clock / 288) |
 
**Terminology enforced by this module**: *pulse ID* = position in the machine bunch
pattern table (universal); *pulse index* = enumeration within a SASE/instrument/device;
*pulse time* = time relative to the start of the subtrain. Our arrays currently conflate
the first two.
 
Direct uses for us:
 
- Replace `pulseId = np.arange(n_pulses)` with `build_pulse_index(pulse_dim='pulseId')`,
  which also survives trains where the pattern changes.
- XPCS lag axis: `pulse_periods()` / `bunch_repetition_rate` gives Δt per train without
  hardcoding 4.5 MHz/2. We already know the pattern is not constant across runs
  (350 vs 155 pulses/train); `search_pulse_patterns()` is the right way to segment a run
  into stationary blocks before correlating.
- `is_constant_pattern()` as a per-run guard in the DAMNIT context file.
---
 
## 4. `extra.components.XGM` — I0, wavelength, photon energy
 
`XGM(run, device=None, default_sase=None)`. `device` accepts a device name, a
`SourceData`/`KeyData`, an EXtra-data alias, or a unique case-insensitive substring
(`XGM(run, "mid")`).
 
- `wavelength(with_units=True)` (nm) and `photon_energy(with_units=True)` (keV) call
  `KeyData.as_single_value()` internally and **raise if the value is not constant over
  the run**; `wavelength_by_train()` / `photon_energy_by_train()` are the safe variants.
  Our pipeline takes the wavelength from here — worth switching to `_by_train` and
  asserting constancy explicitly rather than eating an exception mid-run.
- `pulse_energy(sase=None, series=False)` → µJ, 2-D `DataArray` with dims
  **`(trainId, pulseIndex)`** — *pulse index, not pulse ID*. Sliced to the maximum pulse
  count in the run, missing pulses NaN. `series=True` drops empty trains.
  → When dividing I(q) by I0, align on pulse **index** on both sides, or convert both to
  pulse IDs. This is where a silent off-by-one would hide.
- `pulse_counts(force_slow_data=False)`: cross-checks the slow-data
  `numberOf[SAx]BunchesActual` against the fast data and prefers the fast counts. The
  docs explicitly recommend `XrayPulses` over this for the true pattern.
- Also: `slow_train_energy()`, `npulses()`, `max_npulses()`, `is_constant_pulse_count()`,
  `doocs_server()`, `select_trains()`, `split_trains()`, `plot()` and three component
  plots (pulse-energy heatmap, energy per pulse, energy per train).
- `default_sase` must be set explicitly when a run has XGMs from more than one SASE.
---
 
## 5. `extra.calibration` — constants and the bad-pixel bitfield
 
- `CalibrationData.from_condition(AGIPDConditions(...), "MID_DET_AGIPD1M-1", event_at=run)`,
  `.from_report(id)`, `.from_data(dc, detector_name)`, and — most useful for us —
  **`.from_correction(proposal, run, detector_name)`**, which recovers exactly which
  constants were used to produce the `proc` data. That is the provenance record we
  currently lack for the archived runs.
- `AGIPDConditions(sensor_bias_voltage, memory_cells, acquisition_rate, gain_setting,
  gain_mode, source_energy=9.2, integration_time=12, pixels_x=512, pixels_y=128)`.
- `MultiModuleConstant.ndarray(parallel=N)` / `.xarray(module_naming='modnum'|'aggregator'|'qm')`.
- `DetectorData` / `DetectorModule` expose the PDU↔module mapping (which physical module
  sat in which slot at a given time).
**`BadPixels` (IntFlag) — relevant to our masking.** `image.mask` is 0 for good pixels;
non-zero encodes one or more reasons. Bits include `OFFSET_OUT_OF_THRESHOLD` (1<<0),
`NOISE_OUT_OF_THRESHOLD` (1<<1), `NO_DARK_DATA` (1<<3), `VALUE_IS_NAN` (1<<11),
`VALUE_OUT_OF_RANGE` (1<<12), `GAIN_THRESHOLDING_ERROR` (1<<13), `INTERPOLATED` (1<<16),
`OVERSCAN` (1<<18), `NON_SENSITIVE` (1<<19), `NON_STANDARD_SIZE` (1<<22),
`EXPERT_CHOICE` (1<<23), `USER1/2` (1<<30, 1<<31).
 
The docs state plainly that not all of these mean "bad": `NON_STANDARD_SIZE` marks the
double-width ASIC-edge pixels, which are *intentionally* larger and catch more photons.
Our `data[mask > 0] = np.nan` therefore throws away more than it needs to, and it
overlaps with the separate `extra_geom.agipd_asic_seams()` static mask we already apply.
`extra_data`'s `AGIPD1M.masked_data(key, mask_bits=..., masked_value=...)` lets us
choose which bits actually mask — worth deciding the bit set deliberately and recording
it per run, especially since for XPCS a pixel that is merely double-size is a gain
problem (fixable by a flat-field / solid-angle correction), not a dead pixel.
 
---
 
## 6. Geometry per run — motors
 
`extra.components.AGIPD1MQuadrantMotors(run, detector_id="MID_EXP_AGIPD1M")` with
`.positions(labelled=True, compressed=False)`, `.positions_at(trainId)`,
`.most_frequent_positions()`. (`JF4MHalfMotors` is the JUNGFRAU equivalent — relevant if
the WAXS arm uses JF4M.)
 
These feed `extra_geom.motors.AGIPD_1MMotors` (`with_motor_axes`, `geom_at`,
`move_geom_by`), which needs a **reference geometry** as a starting point and applies the
motor deltas to it. That is the machinery behind the `geometry_from_encoders(run)` call
in our DAMNIT context file.
 
Practical point: `positions(compressed=True)` shows only the changes. If the quadrants
move *within* a run, a single per-run geometry (and therefore a single beam centre and
q-axis) is wrong for part of it. Cheap check to add: assert
`len(positions(compressed=True)) == 1` per run.
 
`extra.geom` re-exports all of `extra_geom` — `agipd_asic_seams()`,
`get_pixel_positions()`, `to_pyfai_detector()`, `to_distortion_array()`,
`from_quad_positions()`, `from_crystfel_geom()`, `position_modules()`, `plot_data()`.
 
---
 
## 7. Scans and proposal/DAMNIT access
 
- `Scantool(run)` — Karabacon config: `scan_type`, `motors`, `motor_devices`, `steps`,
  `start_positions`, `stop_positions`, `acquisition_time`, `active`.
  Warning in docs: `motor_devices` is parsed from a config string and may be `None` on
  older scantool versions.
- `Scan(motor, resolution=None, min_trains=None, intra_step_filtering=1, target=None)`
  detects steps from motor positions. Auto-detection of `resolution` is the fragile part;
  the docs insist on checking `Scan.plot()`. `Scan.from_motor_targets()` is more robust
  when target positions are recorded. Then `bin_by_steps(data, uncertainty_method='std'|'stderr')`,
  `split_by_steps(obj)` (works on anything with `.select_trains()`), `group_data(data)`.
- `extra.proposal.Proposal(10400)`: `.runs()`, `.title()`, `.samples_table()`,
  `.run_type(n)`, `.run_techniques(n)`, `.instrument`, `.damnit()`;
  `prop[42].data(**open_run_kwargs)`, `prop[42].damnit()`, `.plot_timeline()`.
  Sample names and run types come from myMdC — useful for auto-labelling which runs are
  ferritin vs buffer vs dark without maintaining a spreadsheet.
- `extra.damnit.Damnit` is just the DAMNIT API (<https://damnit.readthedocs.io/en/latest/api/>).
---
 
## 8. Utilities worth knowing
 
`extra.utils`: `imshow2` (sane `vmin`/`vmax`, `interpolation="none"`, colorbar,
`lognorm=`), `hyperslicer2` (interactive scroll through a stack of images — good for
per-pulse frames), `ridgeplot` (5–20 curves, better than a heatmap for a handful of I(q)
curves), `find_nearest_index/value`, `reorder_axes_to_shape`, `gaussian`, `gaussian2d`,
`lorentzian`, `fit_gaussian(..., nans_on_failure=True)`.
 
`extra.gui.widgets`: `roi_selection`, `peak_selection` Jupyter widgets.
 
---
 
## 9. What EXtra does **not** provide
 
Confirmed by grep over `src/` and `docs/` — no hits for azimuthal integration, pyFAI,
SAXS/WAXS, XPCS, g2, or speckle anywhere in the package:
 
- **No azimuthal integration.** Our pyFAI-based `analysis_helpers.integrate_run()` has no
  upstream counterpart; the known defects in it are ours to fix.
- **No polar regridding.** `AngularCorrelator` consumes `I(q,φ)` but nothing in EXtra
  produces it. This is the missing link between the SAXS pipeline and the XCCA toolbox.
- **No XPCS / g2 / TTC code at all.**
- **No droplet or image-segmentation code** (our ellipse fitting stays in-house).
- `extra.signal` is fast-timing discrimination for digitizers (`cfd`, `dled`,
  `sinc_interpolate`) — irrelevant to us.
- `extra.applications` otherwise = cookiebox / grating spectrometer calibration and
  `VSLight` (ADQ-digitizer based) — irrelevant to us.
- `extra.karabo_bridge` — online streaming only.
---
 
## 10. Implications for our pipeline
 
1. **Adopt `XrayPulses` for the pulse axis** — fixes the fake `pulseId` coordinate,
   gives per-train Δt for XPCS, and flags runs where the pattern is not constant.
2. **Build the polar-grid step ourselves** and feed `AveragedAngularCorrelationMasked`;
   that is the shortest path to XCCA on this data. Decide `n_q`/`n_phi`/`max_order` from
   the memory numbers in §2, and give each parallel worker its own correlator.
3. **Do not use `CumulativeVariance.from_dataset`** until the double-update is fixed.
4. **Revisit the mask policy**: choose explicit `mask_bits` rather than `mask > 0`, and
   stop double-masking the ASIC seams.
5. **Record provenance per run** with `CalibrationData.from_correction` — we currently
   cannot say which constants produced a given `proc` archive, and pyFAI's silent method
   substitution means we cannot say which integration method ran either.
6. **Guard the geometry**: assert quadrant motors did not move within a run.
7. **Pin EXtra** for anything that goes in a paper; the Maxwell env tracks `master`
   nightly and XCCA is currently unreleased code.
## Open items to check on Maxwell
 
- Does `use_cuda=True` in `AngularCorrelator` actually run on a GPU node?
- Which `mask_bits` set is right for AGIPD at 9–10 keV for XPCS contrast?
- Timing: `AveragedAngularCorrelationMasked.update` per pattern at our intended
  `(n_q, n_phi)` — is per-shot XCCA over a whole run feasible, or do we need CUDA / a
  restricted q-range?
- Is the JUNGFRAU WAXS arm covered by `JF4MHalfMotors`, or is it a different JF variant?

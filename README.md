# MID-10400-analysis — EuXFEL proposal 10400

Analysis code for European XFEL proposal **10400** (ferritin crystallization in
acoustically levitated droplets, MID instrument, cycle 202601).

[`CLAUDE.md`](CLAUDE.md) holds the scientific context, the environment, the
data-format facts and the pitfalls; the per-task design guides live in
[`context/`](context/).

[`context/repository-overview.md`](context/repository-overview.md) is the
extended version of this README: the same layout in full, module by module,
plus what each pass computes, the output-file schema and the invariants that
hold across the codebase. It is written to stand alone, so it is also the file
to paste into a conversation with an assistant that cannot read the repository.
Unlike `CLAUDE.md` it describes only what exists and works — no open tasks.

## Layout

```
src/analysis/           analysis package (<pkg>)
    common/             detector-agnostic layer: status codes, the train/block
                        row model, the mask vocabulary, config hashing and
                        pickling, CPU/pool helpers, the frame-table writer
    saxs/               AGIPD SAXS loader and integrator (`agipd_saxs`)
    waxs/               JUNGFRAU WAXS integrator (`jungfrau_waxs_*`)
    threadenv.py        thread pinning, importable before the numerical stack
src/readers/            pyBeamtime reader plugin (EuXFELMIDRawReader)
    io/readers/euxfel.py
src/amore/              DAMNIT deployment: context file + droplet helpers (not packaged)
scripts/                acceptance and probe scripts, run on a node
tests/                  pytest suite
```

Only `src/analysis` and `src/readers` are packaged. `src/amore` is the DAMNIT
context directory, mirroring its name on the cluster; heavy code must **not**
live there, because DAMNIT `exec`s the context file into a dict and functions
defined that way cannot be pickled to spawned workers.

## Setup (uv)

```bash
uv sync --extra euxfel
```

`uv sync` alone gives the core environment: pyBeamtime (editable) plus pytest,
on Python 3.12. The `euxfel` extra adds the EXtra-data / EXtra-geom / pyFAI
stack, which the reader's `load_run` and both integrators need. Versions are
pinned in `uv.lock`; the "Measured stack" table in `CLAUDE.md` is the set every
benchmark was taken against.

`pyBeamtime` is consumed as an **editable path dependency** pointing at a local
checkout (see `[tool.uv.sources]`; the path is machine-specific).

## The two integrator passes

Both passes store **per-q-bin sufficient statistics** rather than an averaged
I(q), so any later grouping pools exactly:

```
I(q) = Σ_G S / Σ_G N          σ(q) = sqrt(Σ_G V) / Σ_G N
```

Both write one HDF5 file per run holding those sums, a per-frame ledger, the
masks or cells they used, and a provenance record naming every input and its
sha256. Every row is addressed by its stored `trainId` and pulse or cell id,
never by position in an array. A frame that was not integrated carries a status
code and zeros — there are no NaN sentinels — and pooling selects on the status.

The output file's `config_hash` gates both resume and reopening: it covers every
config field that can change a stored number and excludes the ones that only
change how the pass runs. A run that resumes therefore cannot silently merge
blocks computed under different rules.

### AGIPD SAXS (`analysis.saxs`)

Integrates every AGIPD frame over its photon hits with a sparse full-split
operator lifted out of pyFAI, which suits frames that are ~1 % non-zero. Carries
per-memory-cell base masks, and gates on a sparse-vs-pyFAI comparison over real
frames before the worker pool starts.

Design: [`context/agipd-saxs-integrator.md`](context/agipd-saxs-integrator.md).

```python
from extra_geom import AGIPD_1MGeometry
from analysis.saxs import AgipdSaxsConfig, build_operator, denominator, integrate_frame

cfg = AgipdSaxsConfig(proposal=10400, run=423, geometry_file="…/geom_latest.geom")
op, ai = build_operator(AGIPD_1MGeometry.from_crystfel_geom(cfg.geometry_file), cfg)
result = integrate_frame(op, counts, bad, base_bad, denominator(op, base_bad))
```

### JUNGFRAU WAXS (`analysis.waxs`)

One config, PONI, static mask and output file per detector. JUNGFRAU frames are
dense, so this runs pyFAI per frame with the engine built once and the per-frame
mask carried as NaN in the data and variance. Which memory cells are lit is
measured per run and checked for shape rather than against a fixed set; the
readout noise for `Var = σ_read² + E·x` comes from the run's own dark cells where
it has any. `analysis.waxs.combine` scales the two detectors onto one curve over
their q overlap.

Design: [`context/jungfrau-waxs-integrator.md`](context/jungfrau-waxs-integrator.md).

```python
from analysis.waxs import config_for
from analysis.waxs.run import run_jungfrau_waxs

cfg = config_for(proposal=10400, run=423, detector="jf1")
grid = run_jungfrau_waxs(cfg)          # (trainId, cellId, q); None if no cell was lit
```

### From DAMNIT

`src/amore/context.py` declares the variables; each body is an import plus a
call, because anything heavier could not be pickled to the workers the passes
spawn. The functions behind them take a proposal and run number and open the
run themselves — proc for the frames, raw for the run checks:

```python
from analysis.saxs.damnit import agipd_saxs          # -> (trainId, pulseId, q)
from analysis.waxs.damnit import jungfrau_waxs       # -> (trainId, cellId, q)
from analysis.waxs.damnit import combined_curve      # jf2 scaled onto jf1
```

Variables: `agipd_saxs`, `agipd_iq_overview`, `jungfrau_waxs_jf1`,
`jungfrau_waxs_jf2`, `jungfrau_waxs_overview`, `jungfrau_waxs_combined`.

## Using the reader

Because this is a plugin (not built into pyBeamtime), **import the package once**
so it registers with `ReaderRegistry`:

```python
import readers                                # registers EuXFELMIDRawReader (slug "mid")
from pyBeamtime.beamtime import Beamtime

bt = Beamtime.from_path("/gpfs/exfel/exp/MID/202601/p010400")
ds = bt[500].load_raw()                       # lazy xr.Dataset for run r0500
```

The reader is written to be upstreamed into pyBeamtime near-verbatim: it uses
absolute `pyBeamtime.*` imports and self-registers at import time. Upstreaming
= move `euxfel.py` into `pyBeamtime/io/readers/` and add a `from . import
euxfel` line to that package's `__init__.py`.

pyBeamtime's `beamtime init --beamline mid` CLI only sees this reader in a
process where `readers` has been imported. For a one-off, write `beamtime.json`
by hand (`raw_reader: "EuXFELMIDRawReader"`) or upstream the reader.

### Elog

Map runs to samples with the canonical elog. Auto-detect will not match the
run-table's column names, so pass them explicitly:

```bash
beamtime import-elog run_table.csv --scan-col "Run Number" --sample-col "Sample Name"
```

## Scripts

Run on a node against real data, each writing its verdict as JSON beside itself.
They are deliberately **not** unit-tested, so a change to one is checked by
running it, not by pytest.

| Script | What it does |
|---|---|
| `p4_acceptance.py` | runs `agipd_saxs`, then gates it on configuration, the self-test, an independent dense reference, timing and the ledger |
| `w4_acceptance.py` | the same for one JUNGFRAU detector: `--run 423 --detector jf1` |
| `w1_facts.py` | per run and detector: files and sha256s, source names, the lit-cell split and readout noise, whether an ROI would save I/O |
| `w6_data_check.py` | why a run's frames reached `DATA_CHECK_FAILED`, by re-reading the offending frames against both masks |

## Tests

```bash
uv run pytest                    # unit tests, no data and no Maxwell needed
uv run pytest tests/saxs         # the AGIPD suite
uv run pytest tests/waxs         # the JUNGFRAU suite
uv run pytest tests/common       # the shared layer, the pins, config hashing
uv run pytest -m integration     # load_run smoke test (needs extra_data + data)
```

Both integrator suites build mock runs of real shape and dtype, so they run on a
laptop or a login node in a few minutes. `tests/common/test_pinned_digests.py`
pins the values that decide whether a stored file is still valid — both
configs' field lists, their config hashes, both operator hashes and every
module's export surface — and `test_output_snapshot.py` in each suite pins a
content digest of a whole mock run's output. A change to the code that reddens
one of those has moved a stored number, whether or not it meant to.

## Style

```bash
uvx ruff format && uvx ruff check
```

`beamtime.json`, `elog.csv`, and `user_processed/` are runtime artifacts and are
git-ignored, as is all facility data.

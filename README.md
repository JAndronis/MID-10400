# MID-10400-analysis — EuXFEL proposal 10400

Analysis code for European XFEL proposal **10400** (ferritin crystallization in
acoustically levitated droplets, MID instrument, cycle 202601).

[`CLAUDE.md`](CLAUDE.md) holds the scientific context, the environment, the
data-format facts and the pitfalls; the per-task design guides live in
[`context/`](context/).

## Layout

```
src/analysis/           analysis package (<pkg>)
    saxs/               first-pass AGIPD SAXS loader and integrator
    agipd_integrate.py  earlier prototype, superseded by saxs/
src/readers/            pyBeamtime reader plugin (EuXFELMIDRawReader)
    io/readers/euxfel.py
src/amore/              DAMNIT deployment: context file + legacy helpers (not packaged)
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
stack, which the reader's `load_run` and all of `analysis.saxs` need. Versions
are pinned in `uv.lock`; the "Measured stack" table in `CLAUDE.md` is the set
every benchmark was taken against.

`pyBeamtime` is consumed as an **editable path dependency** pointing at a local
checkout (see `[tool.uv.sources]`; the path is machine-specific).

## SAXS first pass

`analysis.saxs` replaces `amore/analysis_helpers.integrate_run`. It integrates
every AGIPD frame of a run over its photon hits with a sparse full-split
operator, and stores per-q-bin sufficient statistics rather than an averaged
I(q), so any later grouping pools exactly:

```
I(q) = Σ_G S / Σ_G N          σ(q) = sqrt(Σ_G V) / Σ_G N
```

Design and phase plan: [`context/saxs-first-pass-integrator.md`](context/saxs-first-pass-integrator.md).
Implemented so far (P1): the operator, the sparse kernels, the pure per-frame
integration and the sparse-vs-pyFAI gate.

```python
from extra_geom import AGIPD_1MGeometry
from analysis.saxs import FirstPassConfig, build_operator, denominator, integrate_frame

cfg = FirstPassConfig(proposal=10400, run=423, geometry_file="…/geom_latest.geom")
op, ai = build_operator(AGIPD_1MGeometry.from_crystfel_geom(cfg.geometry_file), cfg)
result = integrate_frame(op, counts, bad, base_bad, denominator(op, base_bad))
```

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

## Tests

```bash
uv run pytest                    # unit tests, no data and no Maxwell needed
uv run pytest tests/saxs         # the SAXS integrator suite
uv run pytest -m integration     # load_run smoke test (needs extra_data + data)
```

The `analysis.saxs` tests build a real-sized AGIPD geometry from EXtra-geom's
test quad positions, so they run on a laptop or a login node in a few seconds.

## Style

```bash
uvx ruff format && uvx ruff check
```

`beamtime.json`, `elog.csv`, and `user_processed/` are runtime artifacts and are
git-ignored, as is all facility data.

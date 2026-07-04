# p010400_mid — EuXFEL MID raw reader for pyBeamtime

Analysis code for European XFEL proposal **10400** (ferritin crystallization in
acoustically levitated droplets, MID instrument, cycle 202601). See
[`.claude/CLAUDE.md`](.claude/CLAUDE.md) for the full scientific and
architectural spec.

This package provides the **Tier-1 EuXFEL raw reader** as a pyBeamtime plugin:
`EuXFELMIDRawReader`, which turns a MID run directory into a lazy, dask-backed
`xarray.Dataset` via [EXtra-data](https://github.com/European-XFEL/EXtra-data).
XPCS/XCCA/SAXS orchestration (Tier 2), masking, and normalization are **not**
implemented yet (open decisions — see CLAUDE.md §8).

## Layout

```
src/p010400_mid/
    io/readers/euxfel.py    # EuXFELMIDRawReader + module-level path helpers
```

The reader is written to be upstreamed into pyBeamtime near-verbatim: it uses
absolute `pyBeamtime.*` imports and self-registers with `ReaderRegistry` at
import time. Upstreaming = move `euxfel.py` into
`pyBeamtime/io/readers/` and add a `from . import euxfel` line to that package's
`__init__.py`.

## Setup (uv)

```bash
uv sync                 # core env: pyBeamtime (editable) + deps + pytest, on Python 3.12
uv sync --extra euxfel  # add the EXtra-data / EXtra-geom stack (needed by load_run)
```

`pyBeamtime` is consumed as an **editable path dependency** pointing at a local
checkout (see `[tool.uv.sources]` in `pyproject.toml`; the path is
machine-specific). The `euxfel` extra is optional because `load_run` runs
against real data on Maxwell (`/gpfs/exfel/exp/MID/202601/p010400`); the pure
path helpers, `can_read`, and `list_runs` work without it.

## Using the reader

Because this is a plugin (not built into pyBeamtime), **import the package once**
so it registers with `ReaderRegistry`:

```python
import p010400_mid                        # registers EuXFELMIDRawReader (slug "mid")
from pyBeamtime.beamtime import Beamtime

bt = Beamtime.from_path("/gpfs/exfel/exp/MID/202601/p010400")
ds = bt[500].load_raw()                    # lazy xr.Dataset for run r0500
```

pyBeamtime's `beamtime init --beamline mid` CLI only sees this reader in a
process where `p010400_mid` has been imported. For a one-off, write
`beamtime.json` by hand (`raw_reader: "EuXFELMIDRawReader"`) or upstream the
reader.

### Elog

Map runs to samples with the canonical elog. Auto-detect will not match the
run-table's column names, so pass them explicitly:

```bash
beamtime import-elog run_table.csv --scan-col "Run Number" --sample-col "Sample Name"
```

## Tests

```bash
uv run pytest                    # pure helpers, can_read, list_runs, registration
uv run pytest -m integration     # load_run smoke test (needs extra_data + data)
```

`beamtime.json`, `elog.csv`, and `user_processed/` are runtime artifacts and are
git-ignored.

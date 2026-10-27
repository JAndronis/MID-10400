import numpy as np
import pytest
import xarray as xr

from analysis.droplets import (
    CONDITIONS,
    DROPLETS,
    NEAR_ZERO_BLOCKS,
    REPRESENTATIVE,
    by_id,
    groups,
)


def test_ids_are_unique_and_name_the_first_run():
    ids = [d.id for d in DROPLETS]
    assert len(set(ids)) == len(ids)
    assert all(d.id == f"s{d.runs[0]:03d}" for d in DROPLETS)


def test_runs_left_out_belong_to_the_droplet_and_leave_the_folder_name():
    s289 = by_id("s289")
    assert set(s289.excluded) <= set(s289.runs)
    assert s289.kept_runs[0] == 290 and s289.kept_runs[-1] == 306
    assert s289.name == "s289_ferritin50-peg6k_r290-306"


def test_every_sample_has_a_title_and_every_representative_its_group():
    assert {d.sample for d in DROPLETS} <= set(CONDITIONS)
    for (sample, mesh), droplet_id in REPRESENTATIVE.items():
        droplet = by_id(droplet_id)
        assert (droplet.sample, droplet.mesh, droplet.figures) == (sample, mesh, True)
    found = groups()
    assert found == sorted(found, key=lambda g: (list(CONDITIONS).index(g[0]), g[1]))
    assert set(REPRESENTATIVE) <= set(found)


def test_near_zero_blocks_lie_in_figure_droplets():
    kept = {r for d in DROPLETS if d.figures for r in d.kept_runs}
    assert {run for run, _ in NEAR_ZERO_BLOCKS} <= kept


def test_ferritin_concentrates_as_the_droplet_shrinks():
    timeline = xr.Dataset(
        {"volume_mm3": ("trainId", [1.0, 0.5, 0.25])}, coords={"trainId": [1, 2, 3]}
    )
    conc = by_id("s443").ferritin_mg_per_ml(timeline)
    assert conc[0] == pytest.approx(50.0)
    assert np.all(np.diff(conc.values) > 0)

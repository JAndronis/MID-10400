"""One run end to end on the mock proc data: idempotence, resume, workers."""

from __future__ import annotations

import hashlib
import os

import h5py
import numpy as np
import pytest
from mockrun import source
from xpcs_fakes import EMPTY, FRAMES, SHORT, TRAIN_IDS, BrokenPool, InlinePool

from analysis.common.lock import OutputBusy, output_lock
from analysis.common.status import FrameStatus
from analysis.common.writer import IncompleteRun
from analysis.saxs.masks import build_static_bad
from analysis.xpcs import extract
from analysis.xpcs.config import Kernel
from analysis.xpcs.hits import HitSet
from analysis.xpcs.kernel import PairStatus
from analysis.xpcs.rings import build_envelope, build_rings, pixel_map
from analysis.xpcs.store import RingPhotonStore, block_dir_for

GOOD = (10000, 10001, 10003, 10005, 10006, 10007, 10009)
VOLATILE = frozenset(
    {
        "config_hash", "host", "input_file_sha256", "lock", "lock_takeover",
        "package_versions", "platform", "run_checks", "setup_timings", "started_at",
        "timings", "wall_s", "n_workers", "n_blocks", "history", "config_operational",
        "bits_present",
    }
)  # fmt: skip


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rows(path):
    with h5py.File(path, "r") as f:
        return {int(t): i for i, t in enumerate(f["hits/train_id"][()].tolist())}, f[
            "hits/status"
        ][()]


def read_mock_train(run_path, train_id):
    """A train straight out of the mock's module files: dense counts and mask."""
    data, mask = [], []
    for module in range(16):
        with h5py.File(run_path / f"CORR-R0423-AGIPD{module:02d}-S00000.h5", "r") as f:
            ids = f["INDEX/trainId"][()]
            index = f[f"INDEX/{source(module)}/image"]
            at = int(np.flatnonzero(ids == train_id)[0])
            first, count = int(index["first"][at]), int(index["count"][at])
            image = f[f"INSTRUMENT/{source(module)}/image"]
            data.append(image["data"][first : first + count])
            mask.append(image["mask"][first : first + count])
    # (16, P, 512, 128) -> (P, 16*512*128)
    to_frames = lambda a: np.stack(a).transpose(1, 0, 2, 3).reshape(a[0].shape[0], -1)  # noqa: E731
    return to_frames(data), to_frames(mask)


def test_sums_equal_an_independent_reading_of_the_mock_files(pipeline):
    path = pipeline.run(GOOD)
    cfg = pipeline.cfg()
    pixels = pixel_map(cfg, pipeline.geom)
    envelope = build_envelope(pixels, cfg)
    rings = build_rings(pixels, envelope, build_static_bad(cfg.saxs).bad, cfg)
    row_of, status = rows(path)
    assert (status == FrameStatus.OK).all()

    def good(mask):
        flagged = ((mask[:, envelope.pixels] & np.uint32(cfg.saxs.mask_bits)) != 0).any(
            0
        )
        return (rings.label >= 0) & ~flagged

    with h5py.File(path, "r") as f:
        for cur_id, prev_id in ((10001, 10000), (10003, 10001), (10000, None)):
            data, mask = read_mock_train(pipeline.mock.path, cur_id)
            x = data[:, envelope.pixels].astype(np.int64)
            pair = good(mask)
            if prev_id:
                pdata, pmask = read_mock_train(pipeline.mock.path, prev_id)
                y = pdata[:, envelope.pixels].astype(np.int64)
                pair = pair & good(pmask)
            row = row_of[cur_id]
            assert f["hits/partner"][row] == (prev_id or 0)
            for ring in range(rings.n_rings):
                own = good(mask) & (rings.label == ring)
                np.testing.assert_array_equal(
                    f["sums/on"][row, ring], x[:, own] @ x[:, own].T
                )
                np.testing.assert_array_equal(
                    f["sums/s_on"][row, ring], x[:, own].sum(1)
                )
                assert f["sums/cnt"][row, ring] == own.sum()
                if prev_id:
                    sel = pair & (rings.label == ring)
                    np.testing.assert_array_equal(
                        f["sums/off"][row, ring], x[:, sel] @ y[:, sel].T
                    )
                    assert f["sums/cnt_pair"][row, ring] == sel.sum()
            assert bool(f["hits/has_off"][row]) == bool(prev_id)


def test_trains_that_gave_no_photons_are_accounted_for_not_dropped(pipeline):
    path = pipeline.run(sorted((*GOOD, SHORT, EMPTY)))
    row_of, status = rows(path)
    assert status[row_of[SHORT]] == FrameStatus.MISSING_MODULES
    assert status[row_of[EMPTY]] == FrameStatus.NO_FRAMES
    assert (status[[row_of[t] for t in GOOD]] == FrameStatus.OK).all()
    with h5py.File(path, "r") as f:
        pair = f["hits/pair_status"][()]
        assert pair[row_of[SHORT]] == PairStatus.NOT_OK
        # 10003 is two trains after 10001, and the train between is not a hit
        assert f["hits/partner"][row_of[10003]] == 10001


def test_a_complete_run_is_answered_from_the_file_and_writes_nothing(
    pipeline, tmp_path
):
    path = pipeline.run(GOOD)
    cfg = pipeline.cfg()
    files = (path, cfg.store_file)
    before = [(os.stat(p).st_ino, os.stat(p).st_mtime_ns, sha(p)) for p in files]
    assert pipeline.run(GOOD) == path
    after = [(os.stat(p).st_ino, os.stat(p).st_mtime_ns, sha(p)) for p in files]
    assert before == after
    assert not list(tmp_path.rglob("*.lock"))


def test_the_same_hits_with_a_broken_pool_still_answer_from_the_file(pipeline):
    pipeline.run(GOOD)
    pipeline.run(GOOD, pool=BrokenPool)  # no pool is ever asked for work


def test_an_empty_hit_set_is_a_result_and_extracts_nothing(pipeline):
    path = pipeline.run([])
    with h5py.File(path, "r") as f:
        assert f["hits/train_id"].shape == (0,)
        assert f.attrs["finalised"] == 1
    assert not pipeline.cfg().store_file.exists()
    assert pipeline.run([]) == path


def test_a_hit_that_is_not_a_train_of_the_run_is_refused_before_anything_is_written(
    pipeline, tmp_path
):
    with pytest.raises(ValueError, match="not trains of r423"):
        pipeline.run([10000, 99999])
    assert not list(tmp_path.rglob("*.h5"))


def test_a_held_lock_refuses_the_run_and_leaves_the_files_alone(pipeline, tmp_path):
    cfg = pipeline.cfg()
    with output_lock(cfg.output_file), pytest.raises(OutputBusy):
        pipeline.run(GOOD)
    assert not list(tmp_path.rglob("*.h5"))
    pipeline.run(GOOD)


def test_a_block_that_fails_is_retried_by_the_next_run(pipeline, monkeypatch):
    real = extract.process_block
    calls = []

    def flaky(block):
        calls.append(block.train_ids)
        if 10005 in block.train_ids and len(calls) < 99 and not retry:
            raise OSError("a node went away")
        return real(block)

    retry = False
    monkeypatch.setattr(extract, "process_block", flaky)
    with pytest.raises(IncompleteRun, match="not extracted"):
        pipeline.run(GOOD)
    cfg = pipeline.cfg()
    with RingPhotonStore.open(cfg.store_file) as store:
        ids = np.array(GOOD, dtype=np.uint64)
        failed = store.unfinished(ids)
        assert 10005 in failed.tolist() and failed.size < len(GOOD)
        assert not cfg.output_file.exists()

    calls.clear()
    retry = True
    pipeline.run(GOOD)
    extracted = {t for ids in calls for t in ids}
    assert 10005 in extracted
    assert extracted == set(failed.tolist())
    _, status = rows(cfg.output_file)
    assert (status == FrameStatus.OK).all()


def test_allow_incomplete_merges_what_there_is_and_marks_the_rest(
    pipeline, monkeypatch
):
    real = extract.process_block
    monkeypatch.setattr(
        extract,
        "process_block",
        lambda b: (
            (_ for _ in ()).throw(OSError("down")) if 10005 in b.train_ids else real(b)
        ),
    )
    path = pipeline.run(GOOD, allow_incomplete=True)
    row_of, status = rows(path)
    assert status[row_of[10005]] == FrameStatus.WORKER_ERROR
    assert (status[[row_of[t] for t in (10000, 10001)]] == FrameStatus.OK).all()


def test_a_train_whose_modules_sit_in_different_sequence_files_is_not_chained(
    pipeline, monkeypatch
):
    """Kernel.ES chains by sequence file, so a split train has no single chain."""
    real = extract.train_sequences

    def split(key):
        sequences = real(key)
        if key.train_id_coordinates()[0] == 10005:
            sequences[3] = sequences[3] + 1
        return sequences

    monkeypatch.setattr(extract, "train_sequences", split)
    path = pipeline.run(GOOD)
    row_of, status = rows(path)
    assert status[row_of[10005]] == FrameStatus.LABEL_MISMATCH
    assert (status[[row_of[t] for t in GOOD if t != 10005]] == FrameStatus.OK).all()


def test_a_larger_hit_set_extracts_only_the_new_trains(pipeline, monkeypatch):
    first = pipeline.run(GOOD[:4])
    store = pipeline.cfg().store_file
    seen = []
    real = extract.process_block
    monkeypatch.setattr(
        extract, "process_block", lambda b: seen.append(b.train_ids) or real(b)
    )
    second = pipeline.run(GOOD)
    assert first == second
    assert sorted(t for ids in seen for t in ids) == sorted(set(GOOD) - set(GOOD[:4]))
    assert second.with_name("xpcs_hits.prev.h5").exists()
    with RingPhotonStore.open(store) as s:
        assert sorted(s.train_ids().tolist()) == sorted(GOOD)
    _, status = rows(second)
    assert status.size == len(GOOD) and (status == FrameStatus.OK).all()


def test_a_partner_gap_change_rebuilds_the_sums_and_never_touches_the_store(pipeline):
    path = pipeline.run(GOOD)
    store = pipeline.cfg().store_file
    stored = sha(store)
    row_of, _ = rows(path)
    with h5py.File(path, "r") as f:
        assert f["hits/partner"][row_of[10003]] == 10001
    pipeline.run(GOOD, max_partner_gap=1)
    assert sha(store) == stored
    with h5py.File(path, "r") as f:
        assert f["hits/partner"][row_of[10003]] == 0
        assert f["hits/pair_status"][row_of[10003]] == PairStatus.GAP_TOO_LARGE
        assert f["hits/partner"][row_of[10006]] == 10005
        assert f.attrs["max_partner_gap"] == 1


def test_another_kernel_rebuilds_the_sums_from_the_same_store(pipeline):
    path = pipeline.run(GOOD)
    store = pipeline.cfg().store_file
    stored = sha(store)
    pipeline.run(GOOD, kernel=Kernel.ES)
    assert sha(store) == stored
    with h5py.File(path, "r") as f:
        assert f.attrs["kernel"] == "es"
        assert f["hits/has_off"][()].all()  # extra-speckle gives every chain a baseline


def test_a_store_made_for_another_envelope_is_refused_and_kept(pipeline):
    from analysis.xpcs.store import StoreMismatch

    pipeline.run(GOOD)
    store = pipeline.cfg().store_file
    stored = sha(store)
    with pytest.raises(StoreMismatch):
        pipeline.run(GOOD, envelope_margin_mm=1.0)
    assert sha(store) == stored


def test_worker_count_and_block_size_change_no_stored_number(
    pipeline, mock_run, geom, tmp_path, h5_digest
):
    """Spawned workers: 1 worker in blocks of 4 against 3 workers in blocks of 1."""
    from dataclasses import replace

    from analysis.xpcs.run import run_xpcs_hits

    run, dc = mock_run
    digests = []
    for workers, block in ((1, 4), (3, 1)):
        cfg = replace(
            pipeline.cfg(),
            output_root=str(tmp_path / f"w{workers}"),
            n_workers=workers,
            trains_per_block=block,
        )
        out = run_xpcs_hits(
            cfg, HitSet.of(GOOD), dc=dc, geometry=geom, run_dir=run.path
        )
        digests.append((h5_digest(cfg.store_file, VOLATILE), h5_digest(out, VOLATILE)))
    assert digests[0] == digests[1]


def test_inline_and_spawned_pools_agree(pipeline, mock_run, geom, tmp_path, h5_digest):
    pipeline.run(GOOD)
    from dataclasses import replace

    from analysis.xpcs.run import run_xpcs_hits

    run, dc = mock_run
    cfg = replace(
        pipeline.cfg(), output_root=str(tmp_path / "again"), trains_per_block=2
    )
    run_xpcs_hits(
        cfg,
        HitSet.of(GOOD),
        dc=dc,
        geometry=geom,
        run_dir=run.path,
        pool_factory=InlinePool,
    )
    assert h5_digest(cfg.store_file, VOLATILE) == h5_digest(
        pipeline.cfg().store_file, VOLATILE
    )


def test_the_results_name_the_inputs_that_made_them(pipeline):
    path = pipeline.run(GOOD)
    with h5py.File(path, "r") as f:
        assert f.attrs["hits_sha256"] == HitSet.of(GOOD).sha256
        assert f.attrs["n_frames"] == FRAMES
        assert f["pattern/pulse_ids"][()].tolist() == list(range(FRAMES))
        assert f["provenance"].attrs["hit_set_sha256"] == HitSet.of(GOOD).sha256
        assert f.attrs["envelope_sha256"]
    assert len(TRAIN_IDS) == 10


def test_the_public_entry_point_returns_a_reader_over_the_run(pipeline, mock_run, geom):
    from analysis.xpcs import XpcsHits, xpcs_hits

    run, dc = mock_run
    result = xpcs_hits(
        pipeline.cfg(),
        HitSet.of(GOOD),
        dc=dc,
        geometry=geom,
        run_dir=run.path,
        pool_factory=InlinePool,
    )
    assert isinstance(result, XpcsHits)
    coverage = result.coverage()
    assert coverage.n_hits == len(GOOD) and coverage.accounted
    assert coverage.n_usable == len(GOOD)
    g2 = result.g2(binned=False)
    assert g2.tau.size == FRAMES - 1
    assert g2.tau.values[0] == pytest.approx(1 / (1.3e9 / 288))


def test_a_pool_that_dies_mid_extraction_leaves_a_store_the_next_run_completes(
    pipeline,
):
    from concurrent.futures import BrokenExecutor

    with pytest.raises(BrokenExecutor):
        pipeline.run(GOOD, pool=BrokenPool)
    assert not pipeline.cfg().output_file.exists()
    path = pipeline.run(GOOD)
    _, status = rows(path)
    assert (status == FrameStatus.OK).all()


def test_a_block_that_fails_to_sum_is_reported_and_retried(pipeline, monkeypatch):
    from analysis.xpcs import merge

    real = merge.process_block
    state = {"fail": True}

    def flaky(block):
        if state["fail"] and 10006 in block.train_ids:
            raise OSError("the store went away")
        return real(block)

    monkeypatch.setattr(merge, "process_block", flaky)
    with pytest.raises(IncompleteRun, match="could not be summed"):
        pipeline.run(GOOD)
    path = pipeline.cfg().output_file
    row_of, status = rows(path)
    assert status[row_of[10006]] == FrameStatus.WORKER_ERROR
    state["fail"] = False
    pipeline.run(GOOD)
    assert (rows(path)[1] == FrameStatus.OK).all()


def test_a_finished_run_leaves_no_block_files(pipeline):
    pipeline.run(GOOD)
    assert not block_dir_for(pipeline.cfg().store_file).exists()


def test_a_block_that_dies_before_writing_leaves_no_file(pipeline, monkeypatch):
    real = extract.process_block

    def flaky(block):
        if 10005 in block.train_ids:
            raise OSError("a node went away")
        return real(block)

    monkeypatch.setattr(extract, "process_block", flaky)
    with pytest.raises(IncompleteRun):
        pipeline.run(GOOD)
    assert not list(block_dir_for(pipeline.cfg().store_file).glob("*"))


def test_block_files_a_crashed_parent_left_are_merged_not_extracted_again(
    pipeline, monkeypatch
):
    """A worker finished and its file landed, but the parent died before merging it."""
    real = extract.process_block

    def parent_dies(block):
        real(block)
        raise OSError("parent gone")

    monkeypatch.setattr(extract, "process_block", parent_dies)
    with pytest.raises(IncompleteRun):
        pipeline.run(GOOD)
    cfg = pipeline.cfg()
    folder = block_dir_for(cfg.store_file)
    assert list(folder.glob("block_*.h5"))
    (folder / "block_999999.h5.tmp").write_bytes(b"half")
    with RingPhotonStore.open(cfg.store_file) as store:
        assert store.unfinished(np.array(GOOD, dtype=np.uint64)).size == len(GOOD)

    seen = []
    monkeypatch.setattr(
        extract, "process_block", lambda b: seen.append(b.train_ids) or real(b)
    )
    path = pipeline.run(GOOD)
    assert seen == []
    assert not folder.exists()
    _, status = rows(path)
    assert (status == FrameStatus.OK).all()


def corrupt_chunk(dataset, coord):
    """Overwrite one stored chunk of ``dataset`` on disk, bypassing HDF5."""
    dataset.file.flush()
    info = dataset.id.get_chunk_info_by_coord(coord)
    with open(dataset.file.filename, "r+b") as raw:
        raw.seek(info.byte_offset)
        raw.write(b"\xff" * info.size)


def test_a_results_file_that_does_not_read_back_is_removed_not_finalised(
    pipeline, monkeypatch
):
    from analysis.common.readback import UnreadableOutput
    from analysis.xpcs.result import ResultWriter

    real = ResultWriter.finalise

    def after_a_full_disk(self, provenance):
        corrupt_chunk(self._f["sums/s_on"], (0, 0, 0))
        return real(self, provenance)

    monkeypatch.setattr(ResultWriter, "finalise", after_a_full_disk)
    with pytest.raises(UnreadableOutput, match="sums/s_on"):
        pipeline.run(GOOD)
    cfg = pipeline.cfg()
    assert not cfg.output_file.exists()
    assert not list(cfg.output_file.parent.glob("*.lock"))

    monkeypatch.setattr(ResultWriter, "finalise", real)
    path = pipeline.run(GOOD)
    with h5py.File(path, "r") as f:
        assert f.attrs["finalised"] == 1
        assert f["sums/s_on"][()].any()

"""Lit-cell selection and the readout-noise measurement (§3 D4, D3)."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest
from waxs_mockrun import (  # noqa: E402
    CELLS,
    LIT_CELLS,
    MODULE_SHAPE,
    PHOTON_KEV,
    READ_NOISE_KEV,
)

from analysis.waxs.cells import (  # noqa: E402
    CellAccumulator,
    ImplausibleLitCells,
    UnexpectedLitCells,
    is_storage_cell_sequence,
    split_lit_dark,
)
from analysis.waxs.config import EXPECTED_LIT_CELLS  # noqa: E402


def synthetic_train(lit_cells=LIT_CELLS, seed=0, rate=0.6):
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, READ_NOISE_KEV, (CELLS, *MODULE_SHAPE)).astype(np.float32)
    for cell in lit_cells:
        data[cell] += PHOTON_KEV * rng.poisson(rate, MODULE_SHAPE)
    mask = np.zeros((CELLS, *MODULE_SHAPE), dtype=np.uint32)
    return data, mask, np.arange(CELLS, dtype=np.uint16)


def test_the_split_is_measured_not_assumed(cfg, operator):
    op, _ = operator
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(*synthetic_train())
    result = accumulator.finalise()

    assert result.lit == LIT_CELLS
    assert result.dark == tuple(c for c in range(CELLS) if c not in LIT_CELLS)
    lit_fractions = result.lit_fraction[list(LIT_CELLS)]
    dark_fractions = np.delete(result.lit_fraction, list(LIT_CELLS))
    # The real detectors show 42-53 % against <= 0.08 %; the margin is what
    # makes any threshold in 0.01-0.30 safe.
    assert lit_fractions.min() > 0.3
    assert dark_fractions.max() < 0.01


def test_the_readout_noise_comes_from_the_dark_cells(cfg, operator):
    op, _ = operator
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(*synthetic_train())
    result = accumulator.finalise()

    assert result.read_noise_kev == pytest.approx(READ_NOISE_KEV, rel=0.02)
    assert result.read_noise_samples > 0


def test_a_changed_cell_pattern_fails_loudly(cfg, operator):
    """§3 D4: a silent change would halve or double I(q) with nothing saying so.

    Only when the set is *pinned*: ``cfg.expected_lit_cells`` is ``None`` by
    default because the proposal used four readout patterns, so this is the
    reprocess-the-science-block case rather than the every-run one.
    """
    op, _ = operator
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(*synthetic_train(lit_cells=(0, 1, 2, 3)))
    result = accumulator.finalise()

    assert result.lit == (0, 1, 2, 3)
    with pytest.raises(UnexpectedLitCells, match=r"cells \[0, 1, 2, 3\] are lit"):
        result.check_expected(EXPECTED_LIT_CELLS)
    # ...and four consecutive cells are a shape a JUNGFRAU can read, so the
    # run-invariant check says nothing about it. That is the division of
    # labour: one gate knows this beamtime, the other knows the detector.
    result.check_structure(CELLS)


def test_the_failure_names_every_cell_fraction(cfg, operator):
    op, _ = operator
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(*synthetic_train(lit_cells=(0,)))
    with pytest.raises(UnexpectedLitCells) as raised:
        accumulator.finalise().check_expected(EXPECTED_LIT_CELLS)
    # The evidence has to be in the message: the next question is always
    # "by how much did it miss the threshold?"
    assert "per-cell fractions" in str(raised.value)
    assert str(raised.value).count(":") >= CELLS


# ── the run-invariant shape check ────────────────────────────────────────────
@pytest.mark.parametrize(
    ("lit", "expected"),
    [
        ((0, 1, 2, 3, 4, 5, 6, 15), True),  # eight cells starting at 15
        (tuple(range(16)), True),  # all sixteen
        ((15,), True),  # one cell
        ((), True),  # no beam
        ((0, 2, 4), False),  # no start produces this
        ((1, 12), False),  # one of the 67 the old floor produced
        ((5, 7, 9, 10), False),  # another
    ],
)
def test_only_a_storage_cell_sequence_is_plausible(lit, expected):
    """A JUNGFRAU reads consecutive cells mod 16 — every valid pattern is one."""
    assert is_storage_cell_sequence(lit, 16) is expected


def test_an_impossible_set_names_itself_and_the_knobs(cfg, operator):
    """The tripwire raises rather than logging: one that returns is not one."""
    op, _ = operator
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(*synthetic_train(lit_cells=(0, 2, 4)))
    result = accumulator.finalise()

    assert result.lit == (0, 2, 4)
    with pytest.raises(ImplausibleLitCells) as raised:
        result.check_structure(CELLS)
    message = str(raised.value)
    assert "lit_fraction_min" in message and "lit_gap_ratio" in message


# ── where the floor and the gap each bite ────────────────────────────────────
def test_the_floor_alone_splits_a_normal_run():
    """Lit cells well above the floor, dark ones far below it, no gap needed."""
    fraction = np.array([0.4, 0.4, 0.4, 1e-6, 1e-6, 1e-6])
    lit, gap = split_lit_dark(fraction, floor=4.805e-4, gap_ratio=30.0)
    assert lit.tolist() == [True, True, True, False, False, False]
    assert gap is None  # every cell above the floor was kept


def test_the_gap_splits_an_attenuated_run():
    """All six clear the floor; only the step between them says where to cut."""
    fraction = np.array([0.05, 0.05, 0.05, 1e-3, 1e-3, 1e-3])
    lit, gap = split_lit_dark(fraction, floor=4.805e-4, gap_ratio=30.0)
    assert lit.tolist() == [True, True, True, False, False, False]
    assert gap == pytest.approx(50.0)


def test_nothing_above_the_floor_is_a_dark_run():
    fraction = np.zeros(16)
    lit, gap = split_lit_dark(fraction, floor=4.805e-4, gap_ratio=30.0)
    assert not lit.any()
    assert gap is None


def test_the_split_does_not_reorder_the_cells():
    """``split_lit_dark`` sorts internally; the mask it returns must not."""
    fraction = np.array([1e-6, 0.4, 1e-6, 0.4])
    lit, _ = split_lit_dark(fraction, floor=4.805e-4, gap_ratio=30.0)
    assert lit.tolist() == [False, True, False, True]


def test_no_dark_cell_leaves_the_noise_unmeasured(cfg, operator):
    op, _ = operator
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(*synthetic_train(lit_cells=tuple(range(CELLS))))
    result = accumulator.finalise()

    assert result.dark == ()
    assert result.read_noise_kev is None


def test_the_static_mask_is_excluded_from_the_statistics(cfg, operator):
    """A masked pixel must not vote on whether a cell is lit."""
    op, _ = operator
    data, mask, cells = synthetic_train(lit_cells=())
    # Put a huge value everywhere the static mask excludes: if those counted,
    # every cell would look lit.
    data.reshape(CELLS, -1)[:, op.static_bad] = 1e4
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(data, mask, cells)
    assert accumulator.finalise().lit == ()


def test_the_dynamic_mask_is_excluded_too(cfg, operator):
    op, _ = operator
    data, mask, cells = synthetic_train(lit_cells=())
    keep = np.flatnonzero(~op.static_bad)[:1000]
    data.reshape(CELLS, -1)[:, keep] = 1e4
    mask.reshape(CELLS, -1)[:, keep] = np.uint32(1 << 21)
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(data, mask, cells)
    result = accumulator.finalise()
    assert result.lit == ()
    assert accumulator.bits_present == 1 << 21


def test_cell_ids_are_taken_from_the_reader_not_from_position(cfg, operator):
    """CLAUDE.md pitfall 4, on the cell axis."""
    op, _ = operator
    data, mask, _ = synthetic_train(lit_cells=(0, 1))
    shifted = np.arange(CELLS, dtype=np.uint16) + 100
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(data, mask, shifted)
    assert accumulator.finalise().lit == (100, 101)


def test_a_threshold_between_the_two_populations_is_what_matters(cfg, operator):
    op, _ = operator
    data, mask, cells = synthetic_train()
    for fraction in (0.01, 0.10, 0.30):
        accumulator = CellAccumulator(
            dataclasses.replace(cfg, lit_fraction_min=fraction), op.static_bad
        )
        accumulator.update(data, mask, cells)
        assert accumulator.finalise().lit == LIT_CELLS


def test_mismatched_inputs_are_refused(cfg, operator):
    op, _ = operator
    data, mask, cells = synthetic_train()
    accumulator = CellAccumulator(cfg, op.static_bad)
    with pytest.raises(ValueError, match="cell_ids has shape"):
        accumulator.update(data, mask, cells[:4])
    with pytest.raises(TypeError, match="uint32"):
        accumulator.update(data, mask.astype(np.uint16), cells)
    with pytest.raises(ValueError, match=r"\(n_cells, 512, 1024\)"):
        accumulator.update(data[0], mask[0], cells[:1])

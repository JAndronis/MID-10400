"""Lit-cell selection and the readout-noise measurement."""

from __future__ import annotations

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
    # The two populations sit far apart, as on the real detectors, so any
    # threshold between 0.01 and 0.30 splits them.
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
    """A pinned set the run does not match raises, and so does an impossible one.

    A silent change would halve or double I(q). ``cfg.expected_lit_cells`` is
    ``None`` by default because the proposal used several readout patterns, so
    the pin applies only when a reprocess sets it. Four consecutive cells are a
    pattern a JUNGFRAU can read, so only the pin catches them; ``(0, 2, 4)`` is
    one no ``storageCellStart`` produces, so the shape check refuses it.
    """
    op, _ = operator

    def measured(lit_cells):
        accumulator = CellAccumulator(cfg, op.static_bad)
        accumulator.update(*synthetic_train(lit_cells=lit_cells))
        return accumulator.finalise()

    four = measured((0, 1, 2, 3))
    assert four.lit == (0, 1, 2, 3)
    with pytest.raises(UnexpectedLitCells, match=r"cells \[0, 1, 2, 3\] are lit"):
        four.check_expected(EXPECTED_LIT_CELLS)
    four.check_structure(CELLS)

    impossible = measured((0, 2, 4))
    assert impossible.lit == (0, 2, 4)
    with pytest.raises(ImplausibleLitCells):
        impossible.check_structure(CELLS)


# ── the run-invariant shape check ────────────────────────────────────────────
@pytest.mark.parametrize(
    ("lit", "expected"),
    [
        ((0, 1, 2, 3, 4, 5, 6, 15), True),  # eight cells starting at 15
        (tuple(range(16)), True),  # all sixteen
        ((15,), True),  # one cell
        ((), True),  # no beam
        ((0, 2, 4), False),  # no start produces this
        ((1, 12), False),  # a gap no start produces
        ((5, 7, 9, 10), False),  # another
    ],
)
def test_only_a_storage_cell_sequence_is_plausible(lit, expected):
    """A JUNGFRAU reads consecutive cells mod 16, so every valid pattern is one."""
    assert is_storage_cell_sequence(lit, 16) is expected


@pytest.mark.parametrize(
    ("fraction", "lit", "gap"),
    [
        # Lit cells well above the floor, dark ones far below it.
        ([0.4, 0.4, 0.4, 1e-6, 1e-6, 1e-6], [1, 1, 1, 0, 0, 0], None),
        # All clear the floor (an attenuated run); only the step says where.
        ([0.05, 0.05, 0.05, 1e-3, 1e-3, 1e-3], [1, 1, 1, 0, 0, 0], 50.0),
        # Nothing above the floor: a dark run.
        ([0.0] * 16, [0] * 16, None),
        # The split sorts internally; the mask it returns must not.
        ([1e-6, 0.4, 1e-6, 0.4], [0, 1, 0, 1], None),
    ],
)
def test_the_floor_and_the_gap_split_the_cells(fraction, lit, gap):
    got, step = split_lit_dark(np.array(fraction), floor=4.805e-4, gap_ratio=30.0)
    assert got.tolist() == [bool(x) for x in lit]
    assert step == (None if gap is None else pytest.approx(gap))


def test_no_dark_cell_leaves_the_noise_unmeasured(cfg, operator):
    op, _ = operator
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(*synthetic_train(lit_cells=tuple(range(CELLS))))
    result = accumulator.finalise()

    assert result.dark == ()
    assert result.read_noise_kev is None


def test_masked_pixels_do_not_vote_on_whether_a_cell_is_lit(cfg, operator):
    """Huge values under the static mask, or under ``data.mask``, light nothing."""
    op, _ = operator
    data, mask, cells = synthetic_train(lit_cells=())
    data.reshape(CELLS, -1)[:, op.static_bad] = 1e4
    keep = np.flatnonzero(~op.static_bad)[:1000]
    data.reshape(CELLS, -1)[:, keep] = 1e4
    mask.reshape(CELLS, -1)[:, keep] = np.uint32(1 << 21)
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(data, mask, cells)
    assert accumulator.finalise().lit == ()
    assert accumulator.bits_present == 1 << 21


def test_cell_ids_are_taken_from_the_reader_not_from_position(cfg, operator):
    """A cell's id comes from ``data.memoryCell``, never from its array position."""
    op, _ = operator
    data, mask, _ = synthetic_train(lit_cells=(0, 1))
    shifted = np.arange(CELLS, dtype=np.uint16) + 100
    accumulator = CellAccumulator(cfg, op.static_bad)
    accumulator.update(data, mask, shifted)
    assert accumulator.finalise().lit == (100, 101)

"""The JUNGFRAU static mask.

Unlike AGIPD there is no seam mask, because proc sets bit 22
``NON_STANDARD_SIZE`` in ``data.mask`` for the wide edge pixels. There is no
per-cell base mask either. JUNGFRAU frames are ``float32``, so the dynamic mask
is carried as NaN into pyFAI, which drops those pixels from numerator and
denominator alike (:mod:`analysis.waxs.operator`).

So ``static_bad`` is the ``.edf`` file alone. It is a native pyFAI mask written
from silx view, with pyFAI's polarity (non-zero = excluded), and is used with no
inversion.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

from analysis.common.cpu import file_sha256
from analysis.common.masks import MaskSource, StaticMask
from analysis.waxs.config import MODULE_SHAPE, NPIX, JungfrauWaxsConfig

__all__ = [
    "build_static_bad",
    "load_static_mask",
]


def load_static_mask(path: str | Path) -> np.ndarray:
    """Load a ``.edf`` static mask as a flat boolean array, non-zero = excluded.

    :raises ValueError: unexpected shape, or a mask that excludes every pixel,
        which would leave nothing to integrate.
    :raises TypeError: a non-boolean, non-integer dtype, which would make the
        non-zero convention ambiguous.
    """
    import fabio

    with fabio.open(str(path)) as image:
        array = np.asarray(image.data)
    if array.shape != MODULE_SHAPE:
        raise ValueError(
            f"static mask {path} has shape {array.shape}; expected {MODULE_SHAPE}"
        )
    if array.dtype != np.bool_ and not np.issubdtype(array.dtype, np.integer):
        raise TypeError(
            f"static mask {path} has dtype {array.dtype}; expected bool or an "
            "integer type, so that 'non-zero = excluded' is unambiguous"
        )
    bad = array.reshape(-1) != 0
    if bad.all():
        raise ValueError(f"static mask {path} excludes every pixel")
    return bad


def build_static_bad(cfg: JungfrauWaxsConfig) -> StaticMask:
    """The run's static mask: the ``.edf`` file, and nothing else."""
    bad = np.zeros(NPIX, dtype=bool)
    sources: list[MaskSource] = []

    if cfg.static_mask_file is not None:
        path = cfg.static_mask_file
        contribution = load_static_mask(path)
        sources.append(
            MaskSource(
                "static_mask", str(path), file_sha256(path), int(contribution.sum())
            )
        )
        bad |= contribution

    bad.flags.writeable = False
    return StaticMask(
        bad=bad,
        sha256=hashlib.sha256(np.packbits(bad).tobytes()).hexdigest(),
        sources=tuple(sources),
    )

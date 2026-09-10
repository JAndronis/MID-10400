"""EuXFEL MID analysis package for proposal 10400.

Importing this package registers :class:`EuXFELMIDRawReader` (slug ``"mid"``)
with pyBeamtime's ``ReaderRegistry``, so ``Beamtime.from_config`` /
``beamtime init --beamline mid`` can resolve it.
"""

from __future__ import annotations

from readers.io.readers.euxfel import EuXFELMIDRawReader

__version__ = "0.1.0"
__all__ = ["EuXFELMIDRawReader", "__version__"]

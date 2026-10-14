"""A mock CAM5: EXtra-data's mock Basler camera under the droplet camera's name.

Shared by ``tests/test_camera.py`` and the DAMNIT cells' tests in
``tests/diagnostics``, whose conftest puts this directory on ``sys.path``.
"""

from __future__ import annotations

import h5py
import numpy as np
from extra_data import H5File
from extra_data.tests.mockdata import write_file
from extra_data.tests.mockdata.basler_camera import BaslerCamera

from analysis.camera import CAMERA_SOURCE

SHAPE = (60, 81)  # odd width, as CAM5's 659
FIRST_TRAIN = 10000


def mock_camera(path, n_trains=24, n_images=None):
    """A raw file with CAM5 images; ``n_images`` < ``n_trains`` leaves gaps."""
    device = BaslerCamera(
        CAMERA_SOURCE.split(":")[0], nsamples=n_images, sensor_size=SHAPE
    )
    write_file(str(path), [device], n_trains, firsttrain=FIRST_TRAIN)
    with h5py.File(path, "r+") as f:
        pixels = f[f"INSTRUMENT/{CAMERA_SOURCE}/data/image/pixels"]
        n = pixels.shape[0]
        ramp = np.linspace(100, 4000, SHAPE[1])[None, :] * np.ones(SHAPE)
        pixels[:] = (ramp[None] + 10 * np.arange(n)[:, None, None]).astype("u2")
    return H5File(str(path))

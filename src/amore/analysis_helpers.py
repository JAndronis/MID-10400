import logging
import multiprocessing as mp
import queue
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Thread, local

import matplotlib.pyplot as plt
import numpy as np
import pasha
import xarray as xr
from extra.components import XGM
from extra.utils import imshow2
from extra_data import by_id
from extra_data.components import AGIPD1M
from extra_geom import agipd_asic_seams

log = logging.getLogger("azimuthal-integration")


def integrate_frame(ai, frame, mask=None, npt=300, method="csc"):
    if frame.ndim == 3:
        frame = frame.reshape(-1, frame.shape[-1])
    if mask is not None and mask.shape != frame.shape:
        mask = mask.reshape(frame.shape)

    return ai.integrate1d(frame, npt=npt, mask=mask, method=method, unit="q_A^-1")


def get_integrator(geom, wavelength, sdd=None, beamcenter=None, poni_file=None):
    from pyFAI.integrator.azimuthal import AzimuthalIntegrator

    det = geom.to_pyfai_detector()
    det.set_pixel_corners(geom.to_distortion_array())
    ai = AzimuthalIntegrator(detector=det, wavelength=wavelength, dist=sdd)

    geom_y, geom_x = geom._snapped().centre
    center_x = geom_x
    center_y = geom_y

    if beamcenter is not None:
        center_x, center_y = beamcenter

    if center_x is not None and center_y is not None and sdd is not None:
        ai.setFit2D(sdd * 1e3, center_x, center_y)

    if poni_file is not None:
        print(f"Loading from {poni_file}")
        ai = ai.load(str(poni_file))

    return ai


def integrate_run(
    run,
    sdd: float,
    geom,
    beamcenter: "tuple[float, float] | None" = None,
    i0=None,
    mask: str | Path | None = None,
    poni_file=None,
    use_gpu_target: "tuple[int, int] | None" = None,
    n_workers: int = 25,
    trains_per_part: int = 2,
    npt: int = 500,
    n_consumers: int = 2,
) -> xr.DataArray:
    """Azimuthally integrate a run's AGIPD frames into I(q) per frame.

    Reads calibrated multi-module detector frames in parallel loader processes,
    converts them to float32, sets calibration-flagged pixels to NaN, optionally
    normalizes by an incident-intensity monitor, and azimuthally integrates each
    frame with pyFAI. The static ASIC-seam mask is always applied; an optional
    user mask is combined with it.

    Parameters
    ----------
    run : extra_data.DataCollection
        Collection holding the corrected AGIPD modules and control sources.
    sdd : float
        Sample-detector distance in metres.
    geom : extra_geom geometry
        Detector geometry used to build the pyFAI integrator.
    beamcenter : (float, float), optional
        Beam center in Fit2D pixel coordinates; defaults to the geometry centre.
    i0 : xarray.DataArray, optional
        Per-(train, pulse) incident intensity; frames are divided by it in place.
    mask : str | pathlib.Path | numpy.ndarray, optional
        User mask combined with the always-applied static ASIC-seam mask.
    poni_file : str | pathlib.Path, optional
        pyFAI PONI calibration file used by the CPU path.
    use_gpu_target : (int, int), optional
        OpenCL (platform, device) index selecting the GPU path. When None the
        CPU (Cython) path is used instead.
    n_workers : int
        Number of parallel loader processes. Throughput is best near 25 and
        degrades beyond it; per-worker peak memory scales with trains_per_part.
    trains_per_part : int
        Trains per loader chunk. The per-worker staging buffer and peak memory
        scale linearly with it; 2 is the recommended default.
    npt : int
        Number of radial bins.
    n_consumers : int
        Number of GPU integration consumer threads. Each consumer owns a private
        pyFAI integrator, so no integrator instance is shared across threads.

    Returns
    -------
    xarray.DataArray
        Integrated intensities with dims (trainId, pulseId, q) and matching
        coordinates.
    """
    if trains_per_part < 1:
        raise ValueError("trains_per_part must be >= 1")
    if n_workers < 1 or n_consumers < 1:
        raise ValueError("n_workers and n_consumers must be >= 1")

    user_mask = mask
    if isinstance(user_mask, (str, Path)):
        user_mask = np.load(mask)  # ty: ignore[invalid-argument-type]

    # Split the run
    chunks = run.split_trains(trains_per_part=trains_per_part)
    chunk_train_ids = [chunk.train_ids for chunk in chunks]
    print(f"{len(chunk_train_ids)} chunks over {n_workers} workers", flush=True)

    # Load metadata
    agipd = AGIPD1M(run, min_modules=16)
    n_pulses = int(agipd.frames_per_train)
    unstacked_shape = (agipd.n_modules, -1, n_pulses, *agipd.module_shape)

    xgm = XGM(run)
    wavelength = xgm.wavelength(with_units=False) / 1e9

    # Load mask
    static_mask = np.repeat(agipd_asic_seams()[np.newaxis], 16, axis=0)
    if user_mask is not None:
        static_mask |= user_mask

    # Create integrator. This instance is used only for single-threaded work
    # (deriving the q axis below); GPU consumer threads use their own instances.
    ai = get_integrator(geom, wavelength, sdd, beamcenter)

    # A pyFAI OpenCL integrator holds device context and buffers and is not safe
    # to use concurrently. Give each GPU consumer thread its own integrator via
    # thread-local storage, built lazily on the thread's first integration.
    consumer_integrators = local()

    def get_consumer_integrator():
        ai_local = getattr(consumer_integrators, "ai", None)
        if ai_local is None:
            ai_local = get_integrator(geom, wavelength, sdd, beamcenter)
            consumer_integrators.ai = ai_local
        return ai_local

    # Allocate output array
    I_out = pasha.alloc(
        (len(run.train_ids), n_pulses, npt), dtype=np.float32, fill=np.nan
    )
    data_out = pasha.alloc(
        (n_workers, agipd.n_modules, trains_per_part, n_pulses, *agipd.module_shape),
        dtype=np.float32,
    )
    load_queue = mp.Queue(maxsize=n_workers)
    process_queue = mp.Queue(maxsize=n_workers)

    timeout = 3600
    for i in range(n_workers):
        load_queue.put(i, timeout=timeout)

    def load_chunk(agipd_sel):
        data = agipd_sel.get_array("image.data", unstack_pulses=False).data
        data = data.reshape(unstacked_shape)
        mask = agipd_sel.get_array("image.mask", unstack_pulses=False).data
        mask = mask.reshape(unstacked_shape)

        # Convert to float32 if necessary
        data = data.astype(np.float32, copy=False)

        # Mask bad pixels from the calibration pipeline
        data[mask > 0] = np.nan

        # Normalize
        if i0 is not None:
            i0_sel = i0.sel(trainId=agipd_sel.train_ids).data
            data /= i0_sel[np.newaxis, :, :, np.newaxis, np.newaxis]

        return data

    def integrate_chunk(ai, data, I_out, chunk_idx, method):
        trains_in_chunk = len(chunk_train_ids[chunk_idx])
        offset = sum(len(x) for x in chunk_train_ids[:chunk_idx])

        for train in range(trains_in_chunk):
            for pulse in range(data.shape[2]):
                frame = data[:, train, pulse]
                res = integrate_frame(
                    ai, frame, mask=static_mask, method=method, npt=npt
                )
                I_out[int(offset + train), pulse][...] = res.intensity

    def process_chunk_cpu(worker_id, index, tids):
        ai = get_integrator(geom, wavelength, sdd, poni_file=poni_file)
        run_sel = run.select_trains(by_id[tids])
        agipd = AGIPD1M(run_sel)
        print(f"Loading {len(agipd.train_ids)} trains", flush=True)
        data = load_chunk(agipd)
        integrate_chunk(ai, data, I_out, index, "csc")
        print(f"Processed {index}", flush=True)

    def process_chunk_gpu(worker_id, index, tids):
        run_sel = run.select_trains(by_id[tids])
        agipd = AGIPD1M(run_sel, raw=False)

        # print(f"Loading {len(agipd.train_ids)} trains for {index}", flush=True)
        data = load_chunk(agipd)
        out_idx = load_queue.get(timeout=timeout)
        n = data.shape[1]
        # data_out[out_idx][...] = data
        data_out[out_idx][:, :n] = data
        process_queue.put((index, out_idx))

    def gpu_worker(chunk_idx, out_idx):
        ai_local = get_consumer_integrator()
        data = data_out[out_idx]
        integrate_chunk(
            ai_local, data, I_out, chunk_idx, ("full", "csr", "opencl", use_gpu_target)
        )
        load_queue.put(out_idx, timeout=timeout)

    # Do the azimuthal integration in parallel
    pasha.set_default_context("processes", num_workers=n_workers)
    func = process_chunk_cpu if use_gpu_target is None else process_chunk_gpu
    map_thread = Thread(target=lambda: pasha.map(func, chunk_train_ids))
    map_thread.start()

    if use_gpu_target is not None:
        with ThreadPoolExecutor(max_workers=n_consumers) as pool:
            while map_thread.is_alive() or not process_queue.empty():
                try:
                    chunk_idx, out_idx = process_queue.get(timeout=0.5)
                except (queue.Empty, TimeoutError):
                    continue
                else:
                    pool.submit(gpu_worker, chunk_idx, out_idx)

    map_thread.join(timeout=timeout)

    # Get the q array
    q = integrate_frame(ai, np.random.rand(16, 512, 128), npt=npt).radial

    # Create DataArray
    result = xr.DataArray(
        I_out,
        dims=("trainId", "pulseId", "q"),
        coords={
            "trainId": np.array(run.train_ids),
            "pulseId": np.arange(n_pulses),
            "q": q,
        },
    )
    return result


def plot_pulse_resolved(data, name, xlabel):
    dim = data.dims[-1]

    from matplotlib.gridspec import GridSpec

    fig = plt.figure(figsize=(9, 7))
    gs = GridSpec(2, 2, figure=fig)
    ax1 = fig.add_subplot(gs[0, :])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    ax1.plot(data[dim], data.mean(dim=("trainId", "pulseId")))
    ax1.set_yscale("log")
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("Intensity [arb. u]")
    ax1.set_title(f"Mean {name} over all trains and pulses")
    ax1.grid()

    pulse_mean = data.mean("trainId")
    im = imshow2(
        pulse_mean,
        ax=ax2,
        lognorm=True,
        aspect="auto",
        extent=[data[dim].min(), data[dim].max(), len(data.pulseId), 0],
    )
    ax2.set_xlabel(xlabel)
    ax2.set_ylabel("Pulse ID")
    ax2.set_title(f"{name} per-pulse (averaged over trains)")
    fig.colorbar(im, ax=ax2)

    train_mean = data.mean("pulseId")
    im = imshow2(
        train_mean,
        ax=ax3,
        lognorm=True,
        aspect="auto",
        extent=[data[dim].min(), data[dim].max(), len(data.trainId), 0],
    )
    ax3.set_xlabel(xlabel)
    ax3.set_ylabel("Train")
    ax3.set_title(f"{name} per-train (averaged over pulses)")
    fig.colorbar(im, ax=ax3)

    fig.tight_layout()

    return fig


def volume_integration(image, threshold=100, sigma=2):
    """
    Detects the first and last significant intensity changes in pixel values
    for each column of the image using a Gaussian filter and a squared
    difference threshold.

    Parameters:
    - image (numpy.ndarray): Input 2D image array.
    - threshold (float): Threshold for detecting significant intensity changes.
    - sigma (float): Standard deviation for Gaussian filtering.

    Returns:
    - X (numpy.ndarray): Row indices of detected edges.
    - Y (numpy.ndarray): Column indices of detected edges.
    """
    from scipy import ndimage

    # Apply Gaussian filtering
    image_filtered = ndimage.gaussian_filter(image, sigma)

    # Compute squared intensity differences along each column
    gap = np.diff(image_filtered, axis=0) ** 2  # Equivalent to manual differencing

    # Find first significant change (top-down) per column
    top_edges = np.argmax(
        gap > threshold, axis=0
    )  # Returns first index where condition is True

    # Find first significant change (bottom-up) per column
    bottom_edges = gap.shape[0] - np.argmax(gap[::-1, :] > threshold, axis=0) - 1

    # Mask out columns where no threshold crossing was found
    valid_top = gap[top_edges, np.arange(gap.shape[1])] > threshold
    valid_bottom = gap[bottom_edges, np.arange(gap.shape[1])] > threshold

    # Filter out invalid results
    X = np.concatenate((top_edges[valid_top], bottom_edges[valid_bottom]))
    Y = np.concatenate(
        (np.arange(gap.shape[1])[valid_top], np.arange(gap.shape[1])[valid_bottom])
    )

    return X, Y


def fit_ellipse(x, y):
    r"""
    Fits an ellipse to a set of 2D points (x, y) using least squares minimization.

    The general quadratic equation of an ellipse is:

    $$
    A x^2 + B x y + C y^2 + D x + E y + F = 0
    $$

    This function finds the best-fitting ellipse by solving the eigenvalue problem:

    $$
    S^{-1} C v = \lambda v
    $$

    where:
    - \( S = D^T D \) is the scatter matrix,
    - \( C \) is the constraint matrix that enforces the ellipse condition.

    Parameters:
    - x, y (numpy.ndarray): Arrays of x and y coordinates.

    Returns:
    - a (numpy.ndarray): Coefficients \([A, B, C, D, E, F]\) of the fitted
      ellipse equation.
    """
    from scipy.linalg import eig

    x, y = x[:, np.newaxis], y[:, np.newaxis]

    # Construct design matrix D
    D = np.hstack((x * x, x * y, y * y, x, y, np.ones_like(x)))

    # Compute scatter matrix S
    S = np.dot(D.T, D)

    # Constraint matrix to enforce ellipse shape
    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2  # Enforce conic constraints
    C[1, 1] = -1

    # Solve generalized eigenvalue problem
    E, V = eig(S, C)
    mask = np.isfinite(E) & (
        E.real > 0
    )  # exactly one such eigenvalue for a real ellipse fit
    a = V[:, mask][:, 0]

    return a.real  # Ensure real output


def ellipse_center(a):
    r"""
    Computes the center \((x_0, y_0)\) of the ellipse given its quadratic
    equation coefficients.

    The center of the ellipse is computed using:

    $$
    x_0 = \frac{C D - B E}{B^2 - A C}
    $$

    $$
    y_0 = \frac{A E - B D}{B^2 - A C}
    $$

    where \( A, B, C, D, E, F \) are the coefficients of the ellipse equation.

    Parameters:
    - a (numpy.ndarray): Coefficients \([A, B, C, D, E, F]\) of the fitted ellipse.

    Returns:
    - center (numpy.ndarray): \((x_0, y_0)\) coordinates of the ellipse center.
    """
    A, B, C, D, E, _ = a
    B /= 2
    D /= 2
    E /= 2

    denominator = B * B - A * C
    x0 = (C * D - B * E) / denominator
    y0 = (A * E - B * D) / denominator

    return np.array([x0, y0])


def ellipse_angle_of_rotation(a):
    r"""
    Computes the rotation angle \(\phi\) of the ellipse with respect to the x-axis.

    The angle is given by:

    $$
    \phi = \frac{1}{2} \tan^{-1} \left(\frac{2B}{A - C}\right)
    $$

    where:
    - \( A, B, C \) are the quadratic terms of the ellipse equation.

    Parameters:
    - a (numpy.ndarray): Coefficients \([A, B, C, D, E, F]\) of the fitted ellipse.

    Returns:
    - phi (float): Rotation angle in radians.
    """
    A, B, C, _, _, _ = a
    B /= 2

    return 0.5 * np.arctan2(2 * B, (A - C))


def ellipse_axis_length(a):
    r"""
    Computes the lengths of the major and minor axes of the ellipse.

    The formula for the axis lengths is:

    $$
    \text{numerator} = 2 \left(A E^2 + C D^2 + F B^2 - 2 B D E - A C F \right)
    $$

    $$
    \text{denominator}_1 = (B^2 - A C)
        \left( (C - A) \sqrt{1 + \frac{4B^2}{(A - C)^2}} - (C + A) \right)
    $$

    $$
    \text{denominator}_2 = (B^2 - A C)
        \left( (A - C) \sqrt{1 + \frac{4B^2}{(A - C)^2}} - (C + A) \right)
    $$

    The major and minor axis lengths are then computed as:

    $$
    \text{major\_axis} = \sqrt{\frac{\text{numerator}}{\text{denominator}_1}}
    $$

    $$
    \text{minor\_axis} = \sqrt{\frac{\text{numerator}}{\text{denominator}_2}}
    $$

    Parameters:
    - a (numpy.ndarray): Coefficients \([A, B, C, D, E, F]\) of the fitted ellipse.

    Returns:
    - axes (numpy.ndarray): \([\text{major\_axis}, \text{minor\_axis}]\) lengths.
    """
    A, B, C, D, E, F = a
    B /= 2
    D /= 2
    E /= 2

    denominator = B * B - A * C

    # Compute the numerator term
    num = 2 * (A * E**2 + C * D**2 + F * B**2 - 2 * B * D * E - A * C * F)

    # Compute denominator terms for axis length
    term = np.sqrt(1 + (4 * B**2) / ((A - C) ** 2))
    denom1 = denominator * ((C - A) * term - (C + A))
    denom2 = denominator * ((A - C) * term - (C + A))

    # Compute axis lengths
    major_axis = np.sqrt(num / denom1)
    minor_axis = np.sqrt(num / denom2)

    return np.array([major_axis, minor_axis])


# TODO:replace this function with a more robust one that uses skimage
# def fit_droplet(x, y):
#     """
#     Fits an ellipse to a set of 2D points and extracts its center, rotation
#     angle, and axes lengths.

#     Given a set of points \( (x, y) \), this function:
#     1. Fits an ellipse using the least squares method.
#     2. Computes the ellipse parameters:
#         - Center \( (x_0, y_0) \)
#         - Rotation angle \( \phi \)
#         - Major and minor axis lengths

#     Parameters:
#     - x, y (numpy.ndarray): Arrays of x and y coordinates.

#     Returns:
#     - center (numpy.ndarray): \( (x_0, y_0) \) coordinates of the ellipse center.
#     - phi (float): Rotation angle in radians.
#     - axes (numpy.ndarray): \([\text{major\_axis}, \text{minor\_axis}]\) lengths.
#     """
#     # Fit ellipse without shifting data
#     a = fit_ellipse(x, y)

#     # Compute parameters in the original coordinate system
#     center = ellipse_center(a)
#     phi = ellipse_angle_of_rotation(a)
#     radius = ellipse_axis_length(a)

#     return center, phi, radius


def fit_droplet(x, y):
    """
    Fit an ellipse to edge points.
    x = horizontal (columns), y = vertical (rows)

    Returns
    -------
    center : np.ndarray  (x0, y0)
    phi    : float        rotation of the radius[0] semi-axis from +x, in radians
    radius : np.ndarray  (a, b) semi-axes in pixels; a lies along phi, b perpendicular
    """
    from skimage.measure import EllipseModel

    pts = np.column_stack([np.asarray(x, float), np.asarray(y, float)])
    model = EllipseModel()
    if not model.estimate(pts) or model.params is None:
        raise ValueError(
            "ellipse fit did not converge"
        )  # batch worker already catches ValueError
    xc, yc, a, b, theta = model.params
    return np.array([xc, yc]), float(theta), np.array([a, b])


def volume_estimate(radius, px):
    r"""
    Estimates the volume of an ellipsoid given its radii in pixels and the
    pixel size in micrometers.

    The volume of an ellipsoid is given by:

    $$
    V = \frac{4}{3} \pi a b c
    $$

    where:
    - \( a, b, c \) are the semi-axes (radii).
    - The radii are first converted from pixels to micrometers using the
      given pixel size \( px \).

    Parameters:
    - radius (tuple or list of floats): The two measured radii in pixels.
    - px (float): The pixel size in micrometers (um).

    Returns:
    - volume_est (float): Estimated volume in cubic millimeters (mm³).
    """

    # Convert radii from pixels to micrometers
    r0_um = radius[0] * px  # Convert first radius
    r1_um = radius[1] * px  # Convert second radius

    # Compute volume based on the larger radius being the major axis
    if r0_um > r1_um:
        volume_est = (4 / 3) * np.pi * r1_um * (r0_um**2) * 1e-9  # Convert um^3 to mm^3
    else:
        volume_est = (4 / 3) * np.pi * (r1_um**2) * r0_um * 1e-9  # Convert um^3 to mm^3

    return volume_est


def process_single_droplet(
    image, px, threshold=100, sigma=2, plot=False, all_messages=False
):
    """
    Combined ellipsoid fitting and volume estimation levitated droplet tracking
    """

    ### can add different volume integration methods in the future
    Y, X = volume_integration(image, threshold, sigma)

    center, phi, radius = fit_droplet(X, Y)

    if plot:
        plot_ellipse(image, center, phi, radius)

    droplet_volume = volume_estimate(radius, px)

    if all_messages:
        print(rf"Droplet Volume: {droplet_volume:.2f} $mm^3$")

    return center, phi, radius, droplet_volume


def process_droplet_batch(kd, roi=None, px=1, threshold=100, sigma=2, save=True):
    """
    Batch process of droplet fitting for 3D image data

    Parameters:
    - images (numpy.ndarray): 3D array of greyscale droplet images [t, y, x]-
    - threshold (float): Edge detection threshold.
    - sigma (float): Standard deviation for Gaussian filtering.
    """
    kd = kd.drop_empty_trains()
    tids = np.array(kd.train_ids)
    nt = len(tids)
    ctx = pasha.ProcessContext(num_workers=4)

    centers = ctx.alloc((nt, 2), dtype=np.float32, fill=np.nan)
    phis = ctx.alloc((nt,), dtype=np.float32, fill=np.nan)
    radii = ctx.alloc((nt, 2), dtype=np.float32, fill=np.nan)
    volumes = ctx.alloc((nt,), dtype=np.float32, fill=np.nan)

    def worker(worker_id, index, tid):
        image = kd[index]["data.image.pixels"].ndarray(roi=roi).squeeze()

        try:
            center, phi, radius, volume = process_single_droplet(
                image,
                px=px,
                threshold=threshold,
                sigma=sigma,
                plot=False,
                all_messages=False,
            )
        except ValueError:
            return

        # if volume > 100:
        #     return

        centers[index] = center
        phis[index] = phi
        radii[index] = radius
        volumes[index] = volume

    # Big hammer to hide lots of LinAlgWarnings
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ctx.map(worker, tids)

    centers = xr.DataArray(
        centers, dims=("trainId", "dim"), coords=dict(trainId=tids, dim=["x", "y"])
    )
    phis = xr.DataArray(phis, dims=("trainId",), coords=dict(trainId=tids))
    radii = xr.DataArray(
        radii, dims=("trainId", "dim"), coords=dict(trainId=tids, dim=["x", "y"])
    )
    volumes = xr.DataArray(volumes, dims=("trainId",), coords=dict(trainId=tids))

    result = xr.Dataset(dict(center=centers, phi=phis, radius=radii, volume=volumes))
    result.attrs["pixel_size_um"] = px

    # saving
    if save:
        run_no = kd.run_metadata()["runNumber"]
        out_dir = f"/gpfs/exfel/exp/MID/202601/p010400/scratch/xpcs/r{run_no:04d}"
        filename = out_dir + "/droplet_tracking.nc"
        result.to_netcdf(filename)

    return result


def plot_ellipse(image, center, phi, radius, ax=None, **kwargs):
    r"""
    Plots an ellipse overlaying an image using Matplotlib's Ellipse patch.

    The ellipse equation follows:

    - Center: \( (x_0, y_0) \)
    - Major axis length: \( a \)
    - Minor axis length: \( b \)
    - Rotation angle: \( \phi \) (in degrees)

    Parameters:
    - image (numpy.ndarray): The grayscale image on which the ellipse is overlaid.
    - center (tuple or numpy.ndarray): The (x, y) coordinates of the ellipse center.
    - axes (tuple or numpy.ndarray): The lengths of the major and minor axes (a, b).
    - phi (float): Rotation angle of the ellipse in radians.

    Returns:
    - None (displays the image with the overlaid ellipse).
    """
    from matplotlib import patches

    if ax is None:
        _, ax = plt.subplots(figsize=(6, 6))
    if image is not None:
        ax.imshow(image, cmap="gray", origin="upper", **kwargs)
    ell = patches.Ellipse(
        xy=center,
        width=2 * radius[0],
        height=2 * radius[1],
        angle=np.degrees(phi),
        edgecolor="red",
        facecolor="none",
        lw=2,
    )
    ax.add_patch(ell)
    ax.scatter(*center, color="blue", marker="x")
    ax.set(title="Fitted Ellipse Overlay", xlabel="X-axis", ylabel="Y-axis")
    return ax

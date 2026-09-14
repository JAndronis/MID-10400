import logging

import matplotlib.pyplot as plt
import numpy as np
import pasha
import xarray as xr

log = logging.getLogger("azimuthal-integration")


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

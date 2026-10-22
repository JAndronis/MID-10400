import logging

import matplotlib.pyplot as plt
import numpy as np
import pasha
import xarray as xr

log = logging.getLogger("azimuthal-integration")


def volume_integration(image, threshold=100, sigma=2):
    """
    Find the top and bottom droplet edge in each column of a 2D image.

    An edge is the first row, from either end, where the squared row-to-row
    difference of the Gaussian-filtered image exceeds ``threshold``. Columns
    with no such row contribute no point.

    Parameters:
    - image (numpy.ndarray): 2D image.
    - threshold (float): Bound on the squared difference.
    - sigma (float): Gaussian filter width in pixels.

    Returns:
    - X (numpy.ndarray): Row indices of the edges.
    - Y (numpy.ndarray): Column indices of the edges.
    """
    from scipy import ndimage

    image_filtered = ndimage.gaussian_filter(image, sigma)

    gap = np.diff(image_filtered, axis=0) ** 2

    top_edges = np.argmax(gap > threshold, axis=0)

    bottom_edges = gap.shape[0] - np.argmax(gap[::-1, :] > threshold, axis=0) - 1

    # argmax returns 0 where nothing crosses, so check the crossing itself.
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
    Fit an ellipse to edge points, x horizontal (columns) and y vertical (rows).

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
        # process_droplet_batch's worker skips a train on ValueError.
        raise ValueError("ellipse fit did not converge")
    xc, yc, a, b, theta = model.params
    return np.array([xc, yc]), float(theta), np.array([a, b])


def volume_estimate(radius, px):
    r"""
    Volume of the oblate spheroid with the two measured semi-axes.

    The unseen third semi-axis is taken equal to the larger one, so
    \( V = \frac{4}{3} \pi a^2 b \) with \( a \ge b \).

    Parameters:
    - radius (tuple or list of floats): The two measured semi-axes in pixels.
    - px (float): The pixel size in micrometers (um).

    Returns:
    - volume_est (float): Volume in cubic millimeters (mm³).
    """

    r0_um = radius[0] * px
    r1_um = radius[1] * px

    # 1e-9 converts um^3 to mm^3.
    if r0_um > r1_um:
        volume_est = (4 / 3) * np.pi * r1_um * (r0_um**2) * 1e-9
    else:
        volume_est = (4 / 3) * np.pi * (r1_um**2) * r0_um * 1e-9

    return volume_est


def process_single_droplet(
    image, px, threshold=100, sigma=2, plot=False, all_messages=False
):
    """
    Fit an ellipse to one droplet image and estimate its volume in mm³.
    """

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
    Fit the droplet in every camera image of a run, one train per image.

    A train whose fit fails keeps NaN in every output. With ``save`` the result
    is also written to scratch/xpcs/r{run:04d}/droplet_tracking.nc, which must
    already exist.

    Parameters:
    - kd (KeyData): The camera's image key; empty trains are dropped.
    - roi (tuple of slices): Region of each image to fit.
    - px (float): Pixel size in micrometers.
    - threshold (float): Edge detection threshold.
    - sigma (float): Gaussian filter width in pixels.

    Returns:
    - xarray.Dataset with center, phi, radius and volume per trainId.
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

        centers[index] = center
        phis[index] = phi
        radii[index] = radius
        volumes[index] = volume

    # The ellipse fit emits many LinAlgWarnings.
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

    if save:
        run_no = kd.run_metadata()["runNumber"]
        out_dir = f"/gpfs/exfel/exp/MID/202601/p010400/scratch/xpcs/r{run_no:04d}"
        filename = out_dir + "/droplet_tracking.nc"
        result.to_netcdf(filename)

    return result


def plot_ellipse(image, center, phi, radius, ax=None, **kwargs):
    r"""
    Draw a fitted ellipse and its centre over an image.

    Parameters:
    - image (numpy.ndarray or None): Grayscale image to draw under the ellipse.
    - center (tuple or numpy.ndarray): The (x, y) centre in pixels.
    - phi (float): Rotation of the ``radius[0]`` semi-axis from +x, in radians.
    - radius (tuple or numpy.ndarray): The two semi-axes in pixels.
    - ax (matplotlib Axes or None): Axes to draw on; a new figure if None.
    - kwargs: Passed to ``imshow``.

    Returns:
    - The Axes.
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

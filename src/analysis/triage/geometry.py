"""The figure of one run's hit recognition (run-triage §4.4).

Takes the dict `analysis.triage.classify.hits` returns. A mesh run carries the
overlap fraction f per train (the share of a full hit's droplet signal) and gets
the scan map beside f against time; any other run has no f and gets the class
timeline alone.

The droplet outline on the map is an approximation: an ellipse with the
semi-axes of ``droplet_fit``, its bottom edge at the lowest Z of an "on" train
and its X centre at the middle of the "on" trains' X range. It is a guide for
the eye, not a fit.
"""

from __future__ import annotations

import numpy as np

__all__ = ["CLASS_COLORS", "droplet_outline", "hit_geometry_figure"]

CLASS_COLORS = {"on": "#d62728", "edge": "#ff9f1c", "off": "#4c78a8", "none": "#bbbbbb"}
#: Trains are 10 Hz.
TRAIN_RATE_HZ = 10.0


def droplet_outline(x, z, classes, semi_axes_mm):
    """(xc, zc, a, b) of the approximate droplet, or None without "on" trains.

    :param x, z: hexapod position per train, same order as ``classes``.
    :param semi_axes_mm: (a, b), the horizontal and vertical semi-axes.
    """
    on = (np.asarray(classes) == "on") & np.isfinite(x) & np.isfinite(z)
    if semi_axes_mm is None or not on.any():
        return None
    a, b = (float(v) for v in semi_axes_mm)
    xc = 0.5 * (float(np.min(x[on])) + float(np.max(x[on])))
    return xc, float(np.min(z[on])) + b, a, b


def hit_geometry_figure(found: dict, positions=None, semi_axes_mm=None, *, title=None):
    """The figure for one run.

    :param found: the result of ``classify.hits``.
    :param positions: ``(X, Z)`` hexapod positions [mm] per train, in the order
        of ``found["classes"]``; without them a mesh run has no map.
    :param semi_axes_mm: ``(a, b)`` of the droplet, to draw its outline.
    """
    import matplotlib.pyplot as plt
    from matplotlib.patches import Ellipse

    from analysis.triage.classify import ON_FRACTION

    classes = found["classes"].values.astype(str)
    tids = found["classes"].trainId.values
    t = (tids.astype(np.int64) - int(tids[0])) / TRAIN_RATE_HZ if tids.size else tids
    f = found["overlap"]
    has_map = found["kind"] == "mesh" and positions is not None
    n = 3 if has_map else 2
    fig, axes = plt.subplots(1, n, figsize=(5.2 * n, 4.2))
    axes = np.atleast_1d(axes)
    names = [k for k in CLASS_COLORS if (classes == k).any()]
    colors = np.array([CLASS_COLORS[c] for c in classes]) if classes.size else []

    i = 0
    if has_map:
        x, z = (np.asarray(p, float) for p in positions)
        ax = axes[i]
        i += 1
        for k in names:
            m = classes == k
            ax.scatter(x[m], z[m], s=8, c=CLASS_COLORS[k], label=f"{k} {int(m.sum())}")
        outline = droplet_outline(x, z, classes, semi_axes_mm)
        if outline is not None:
            xc, zc, a, b = outline
            ax.add_patch(
                Ellipse((xc, zc), 2 * a, 2 * b, fill=False, ls="--", color="k")
            )
        ax.set_xlabel("hexapod X [mm]")
        ax.set_ylabel("hexapod Z [mm]")
        ax.set_title("scan map (dashed: droplet approximation)", fontsize=9)
        ax.legend(fontsize=7, markerscale=1.5)

    ax = axes[i]
    i += 1
    if f is not None:
        ax.scatter(t, f.values, s=5, c=colors)
        ax.axhline(ON_FRACTION, color="k", lw=0.8, ls="--")
        ax.annotate(
            f"{ON_FRACTION:g}", (0.01, ON_FRACTION), xycoords=ax.get_yaxis_transform()
        )
        ax.set_ylabel("overlap fraction f")
        ax.set_title("f per train", fontsize=9)
    else:
        level = {"off": 0, "on": 1}
        y = np.array([level.get(c, np.nan) for c in classes], float)
        ax.scatter(t, y, s=5, c=colors)
        ax.set_yticks([0, 1], ["off", "on"])
        ax.set_title("class per train (static run: no f)", fontsize=9)
    ax.set_xlabel("time since first train [s]")

    ax = axes[i]
    counts = [int((classes == k).sum()) for k in names]
    ax.bar(names, counts, color=[CLASS_COLORS[k] for k in names])
    for j, c in enumerate(counts):
        ax.text(j, c, str(c), ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("trains")
    ax.set_title("classes", fontsize=9)

    block = found["block"] or ""
    fig.suptitle(
        title or f"r{found['run']:04d}, {found['kind']} run {block}".strip(),
                 fontsize=10)  # fmt: skip
    fig.tight_layout()
    return fig

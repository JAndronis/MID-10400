import sqlite3
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from analysis_helpers import plot_ellipse, process_droplet_batch
from damnit.context import Cell, Variable
from extra.components import XGM, Scantool, XrayPulses
from extra.data import by_id
from extra_speckle.pipeline import xpcs_offline

XPCS_RESULTS = {}
MASK_PATH = "/gpfs/exfel/u/usr/MID/202601/p010400/masks/mask_2026-09-08_AGIPD_SAXS.npy"


def last_valid_value(var_name, run_number):
    with sqlite3.connect("runs.sqlite") as db:
        df = pd.read_sql_query("SELECT * FROM runs", db)

    df_sel = df[["run", var_name]]
    df_sel = df_sel.loc[df_sel["run"] <= run_number]
    index = df_sel[var_name].last_valid_index()
    return df_sel[var_name].iloc[index] if index is not None else None


# Run metadata


@Variable("Trains", tags=["Metadata"])
def n_trains(run):
    return len(run.train_ids)


@Variable("Run length", tags=["Metadata"])
def run_length(run):
    ts = run.train_timestamps()
    delta = ts[-1] - ts[0]
    delta_s = int(delta / np.timedelta64(1, "s"))
    return str(timedelta(seconds=delta_s))


def run_size_tb(path):
    run_size_bytes = sum(f.stat().st_size for f in path.rglob("*"))
    return run_size_bytes / 1e12


@Variable("Raw size [TB]", tags=["Metadata", "data_vol"])
def raw_size(run, proposal_path: "meta#proposal_path", run_no: "meta#run_number"):
    run_path = proposal_path / "raw" / f"r{run_no:04}"
    return run_size_tb(run_path)


@Variable("Proc size [TB]", data="proc", tags=["data_vol"])
def proc_size(run, proposal_path: "meta#proposal_path", run_no: "meta#run_number"):
    run_path = proposal_path / "proc" / f"r{run_no:04}"
    return run_size_tb(run_path)


@Variable("Run type", tags=["Metadata"])
def run_type(run, run_type: "mymdc#run_type"):
    return run_type


@Variable("Sample", tags=["Metadata"])
def sample(run, sample: "mymdc#sample_name"):
    return sample


@Variable("Scan type", tags=["Metadata"])
def scan_type(run):
    sc = Scantool(run)
    if sc.active:
        return sc.format(compact=True)


# Beam properties


@Variable("XGM intensity [uJ]", summary="mean")
def xgm_intensity(run):
    return XGM(run).pulse_energy().mean("pulseIndex")


@Variable("XGM overview")
def xgm_overview_plot(run):
    return XGM(run).plot()


@Variable("Pulses", summary="mean")
def pulses(run):
    return XrayPulses(run).pulse_counts().to_xarray()


@Variable("Rep rate [MHz]")
def rep_rate(run):
    pulses = XrayPulses(run)
    rep_rates = pulses.pulse_repetition_rates()
    rep_rates = rep_rates[rep_rates > 0]
    if len(rep_rates) > 0:
        return rep_rates.mean() / 1e6


@Variable("XTD1 trans.", summary="nanmean")
def xtd1_transmission(run):
    return run.alias["xtd1-transmission"].xarray()


@Variable("XTD6 trans.", summary="nanmean")
def xtd6_transmission(run):
    return run.alias["xtd6-transmission"].xarray()


@Variable("Opt. trans.", summary="nanmean")
def opt_transmission(run):
    return run.alias["opt-transmission"].xarray()


@Variable("Total trans.", summary="nanmean")
def total_transmission(
    run,
    xtd1: "var#xtd1_transmission",
    xtd6: "var#xtd6_transmission",
    opt: "var#opt_transmission",
):
    return xtd1 * xtd6 * opt


### clause to pick selected trains from droplet filter, else

selected_trains = None


@Variable("Droplet fit", data="proc", cluster=True, tags=["offline"])
def droplet_fit(
    run,
    proposal: "meta#proposal",
    run_no: "meta#run_number",
    run_type: "mymdc#run_type",
    proposal_path: "meta#proposal_path",
):

    if run_type in ["XPCS", "Droplet Tracking", "Levitation"]:
        droplet_roi = np.s_[80:230, 200:400]
        droplet = process_droplet_batch(
            run["MID_EXP_SAM/CAM/CAM5:daqOutput"],
            droplet_roi,
            px=13.9,
            threshold=1e4,
            sigma=2,
        )

        idx = len(run.train_ids) // 2
        preview_tid = droplet.trainId.data[idx]
        image = (
            run["MID_EXP_SAM/CAM/CAM5:daqOutput"]
            .select_trains(by_id[[preview_tid]])["data.image.pixels"]
            .ndarray()
            .squeeze()
        )

        from matplotlib.gridspec import GridSpec

        fig = plt.figure(figsize=(9, 7))
        gs = GridSpec(2, 2, figure=fig)
        ax1 = fig.add_subplot(gs[0, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, :])

        plot_ellipse(
            image[droplet_roi],
            droplet.center[idx],
            droplet.phi[idx],
            droplet.radius[idx],
            ax=ax1,
        )
        ax1.scatter(
            droplet.center.sel(dim="x"),
            droplet.center.sel(dim="y"),
            alpha=0.1,
            label="Centers",
        )
        ax1.legend()

        ax2.scatter(droplet.radius.sel(dim="x"), droplet.radius.sel(dim="y"), alpha=0.1)
        ax2.set_title("Droplet radius scatter plot")
        ax2.set_xlabel("X-axis")
        ax2.set_ylabel("Y-axis")
        ax2.grid()

        droplet.volume.plot(ax=ax3)
        ax3.axvline(
            preview_tid, color="red", ls="--", lw=2, label="Preview image timestep"
        )
        ax3.set_ylabel("Volume [mm³]")
        ax3.set_title("Droplet volume")
        ax3.grid()

        fig.tight_layout()

        return Cell(droplet, preview=fig)
    else:
        return None


@Variable("XPCS PIPELINE", data="proc", cluster=True, tags=["offline"])
def xpcs_pipeline(
    run,
    proposal: "meta#proposal",
    run_no: "meta#run_number",
    sample: "mymdc#sample_name",
    rep_rate: "var#rep_rate",
    run_type: "mymdc#run_type",
    pulses: "var#pulses",
    proposal_path: "meta#proposal_path",
):
    # proposal_path / "usr/geometry" / "geom_latest.geom"
    # last_valid_value("agipd_geom", run_no)
    geom_path = None

    if run_type in ["XPCS", "Levitation"]:
        # setting up the parameters
        args = dict()

        args["calc_stats"] = True
        # args['q_range'] = np.linspace(0.0225,0.0625,5)
        # args['q_range'] = np.unique(np.concatenate([
        #     np.linspace(0.0225,0.05,3), np.linspace(0.05,0.056,2),
        #     np.linspace(0.063,0.067,2), np.linspace(0.073,0.08,2)]))
        args["q_range"] = np.array(
            [0.0225, 0.035, 0.045, 0.056, 0.065, 0.069, 0.075, 0.086]
        )
        args["geom"] = geom_path
        args["mask"] = MASK_PATH

        if run_type == "XPCS":
            # args['motor_moving'] = 'MID_EXP_SAM/MDL/DATA_SELECTOR_2'
            # args['motor_key'] = 'MID_SAE_FSSS/MOTOR/SCANNERX.actualPosition'
            # args['motor_range'] = (-24, 24)  # should be revisited !

            args["remove_outer_points"] = (
                False  # change to True after revisiting motor_range
            )
        else:
            args["remove_outer_points"] = (
                False  # change to True after revisiting motor_range
            )

        args["refine_beamcenter"] = False
        args["px"] = 607.4598
        args["py"] = 672.0767

        args["sdd"] = 7531.5962  # 7532

        if run_type == "XPCS":
            args["intensity_masking"] = False
            args["outlier_masking"] = True
        else:
            args["intensity_masking"] = False
            args["outlier_masking"] = True

        args["xgm_source"] = "SA2_XTD1_XGM/XGM/DOOCS:output"

        args["run_xpcs_analysis"] = True
        args["run_saxs_analysis"] = True
        args["subtract_TTCF_off"] = True

        args["fit_result"] = True
        args["show_plots"] = False

        # ---------------------------------

        global XPCS_RESULTS
        XPCS_RESULTS = xpcs_offline(proposal, run_no, **args)

        if (XPCS_RESULTS["TTCF_raw"] is not None) and (
            XPCS_RESULTS["TTCF_off"] is not None
        ):
            res = "Done"
        else:
            res = "Incomplete"
    else:
        res = "Incomplete"

    return res


@Variable(title="SDD", data="proc", cluster=True, tags=["offline"])
def xpcs_sdd(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["setup"].config.sdd


@Variable(title="XPCS TTCF", data="proc", cluster=True, tags=["offline"])
def xpcs_ttcf(run, ds: "var#xpcs_pipeline"):
    ttcf_raw = XPCS_RESULTS["TTCF_raw"]
    ttcf_off = XPCS_RESULTS["TTCF_off"]

    if (ttcf_raw is not None) and (ttcf_off is not None):
        ttcf = ttcf_raw - ttcf_off
    else:
        ttcf = None

    return ttcf


@Variable("XPCS mean dataset", data="proc", cluster=True, tags=["offline"])
def xpcs_mean_dataset(run, ds: "var#xpcs_pipeline"):
    return xr.Dataset({"ttcf_mean": XPCS_RESULTS["TTCF"], "g2": XPCS_RESULTS["g2"]})


@Variable("XPCS SAXS", data="proc", cluster=True, tags=["offline"])
def xpcs_saxs(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["SAXS"]


@Variable("XPCS q-rings", data="proc", cluster=True, tags=["offline"])
def xpcs_q_rings_plot(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["figure_2d"]


@Variable("XPCS beamcenter", data="proc", cluster=True, tags=["offline"])
def xpcs_beamcenter_plot(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["fig_beamcenter"]


@Variable("XPCS outliers", data="proc", cluster=True, tags=["offline"])
def xpcs_outliers_plot(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["fig_outlier"]


@Variable("XPCS intensity outliers", data="proc", cluster=True, tags=["offline"])
def xpcs_intensity_outliers_plot(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["fig_intens_outlier"]


@Variable("XPCS kbar", data="proc", cluster=True, tags=["offline"])
def xpcs_kbar_plot(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["figure_kbar"]


@Variable("XPCS correction", data="proc", cluster=True, tags=["offline"])
def xpcs_correction_plot(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["figure_xpcs_corr"]


@Variable("XPCS TTCF's", data="proc", cluster=True, tags=["offline"])
def xpcs_ttcfs_plot(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["figure_ttcf"]


@Variable("XPCS g2", data="proc", cluster=True, tags=["offline"])
def xpcs_g2_plot(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["figure_g2"]


@Variable("XPCS g2 fit", data="proc", cluster=True, tags=["offline"])
def xpcs_g2_fit_plot(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["figure_fit"]


@Variable("XPCS SAXS overview", data="proc", cluster=True, tags=["offline"])
def xpcs_saxs_plot(run, ds: "var#xpcs_pipeline"):
    return XPCS_RESULTS["figure_saxs"]


# ── AGIPD SAXS integration (analysis.saxs; context file §10, P5) ─────────────
# `agipd_saxs` keeps its name and column: this is the same quantity, computed
# by `analysis.saxs` instead of `analysis_helpers.integrate_run` — over an hour
# per run against 6.3 min here, and without dividing by I0 in place, which
# could not be undone afterwards.
#
# What the column holds therefore changes at this commit: I(q) in nm^-1 and not
# I0-divided, where before it was A^-1 and divided. Runs processed earlier hold
# the old quantity, so clear the column for them rather than plotting across
# the boundary.
#
# Both variables are thin on purpose: DAMNIT execs this file into a dict, so a
# function defined here cannot be pickled to the workers the pass spawns.


@Variable("AGIPD I(q)", data="proc", cluster=True, tags=["offline"])
def agipd_saxs(run, proposal: "meta#proposal", run_no: "meta#run_number"):
    """I(q) per (trainId, pulseId). Raises if any frame did not reach OK.

    `run` is unused: a data="proc" variable is handed a proc-only collection,
    which holds no XGM, timeserver or motors, so the pass opens proc for the
    frames and raw for its own run checks. The per-frame sums land in
    scratch/agipd_saxs/r{run:04d}/; what is returned is the intensity grid.
    """
    from analysis.saxs import damnit

    return damnit.agipd_saxs(proposal, run_no)


@Variable("AGIPD I(q) overview", data="proc", cluster=True, tags=["offline"])
def agipd_iq_overview(run, grid: "var#agipd_saxs"):
    from analysis.saxs import damnit

    return damnit.overview_figure(grid)

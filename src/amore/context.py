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

        # Set True once motor_range is settled; the XPCS motor sources
        # (MID_EXP_SAM/MDL/DATA_SELECTOR_2, MID_SAE_FSSS/MOTOR/SCANNERX) and
        # their range are still unresolved, so it is False for every run_type.
        args["remove_outer_points"] = False

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


# ── AGIPD preprocessing (analysis.saxs) ─────────────────────────────────────
# One variable reads AGIPD proc, once, and writes three files: the frame table
# (scratch/agipd_saxs/r####/agipd_saxs.h5), the per-pixel window sums and the
# per-frame photon statistics (usr/cached_files/agipd_pixel_sums, ..._photon_stats).
# Every other AGIPD column reads one of those files, so the table shows which
# variable computes and which only presents (user, 2026-10-14). A complete run
# is answered from its files without writing (CLAUDE.md pitfall 26).
#
# agipd_saxs holds I(q) in nm^-1, undivided. Runs processed before this pass
# existed hold A^-1 divided by I0, so clear the column for them rather than
# plotting across the boundary.
#
# Thin on purpose: DAMNIT execs this file into a dict, so a function defined
# here cannot be pickled to the workers the pass spawns.


@Variable("AGIPD preprocessing", data="proc", cluster=True, tags=["offline"])
def agipd_preprocessing(run, proposal: "meta#proposal", run_no: "meta#run_number"):
    """The run's one read of AGIPD proc: frame table, window sums, photon stats.

    `run` is unused: the pass opens proc for the frames and raw for its own run
    checks. The cell is a status per product, green when all are complete and
    OK; a failure is a red cell, and each reader then says what its file lacks.
    """
    from analysis.saxs import damnit

    return damnit.preprocess(proposal, run_no)


@Variable("AGIPD I(q)", data="proc", cluster=True, tags=["offline"])
def agipd_saxs(
    run,
    prep: "var#agipd_preprocessing",
    proposal: "meta#proposal",
    run_no: "meta#run_number",
):
    """I(q) per (trainId, pulseId), read from the frame table.

    Raises unless every frame is OK.
    """
    from analysis.saxs import damnit

    return damnit.agipd_saxs_from_file(proposal, run_no)


@Variable("AGIPD I(q) overview", data="proc", cluster=True, tags=["offline"])
def agipd_iq_overview(run, grid: "var#agipd_saxs"):
    from analysis.saxs import damnit

    return damnit.overview_figure(grid)


# ── JUNGFRAU preprocessing (analysis.waxs; context/jungfrau-waxs-integrator) ──
# One variable runs the pass for both detectors, one after the other: the two
# JUNGFRAU-500Ks have different geometries, different static masks and
# different q ranges, so each gets its own PONI and its own output file under
# scratch/jungfrau_waxs/r{run:04d}/. The per-detector columns read those files.
# They are combined only in the overview, and there as two traces with their
# overlap visible — never concatenated. A run with no lit cell writes no file:
# its column is skipped, and the preprocessing cell says "dark".
#
# Thin on purpose, like the AGIPD ones above.


@Variable("JUNGFRAU preprocessing", data="proc", cluster=True, tags=["offline"])
def jungfrau_preprocessing(run, proposal: "meta#proposal", run_no: "meta#run_number"):
    """Both detectors' passes: a status per detector (OK counts, dark, failed)."""
    from analysis.waxs import damnit

    return damnit.preprocess(proposal, run_no)


@Variable("JF1 I(q)", data="proc", cluster=True, tags=["offline"])
def jungfrau_waxs_jf1(
    run,
    prep: "var#jungfrau_preprocessing",
    proposal: "meta#proposal",
    run_no: "meta#run_number",
):
    """I(q) per (trainId, cellId) for JUNGFRAU 1, read from its file.

    There is no pulse axis: JUNGFRAU's reader has no pulse ids, and which of a
    train's X-ray pulses each memory cell sampled is unresolved, so the cell id
    is stored and no pulse id is invented.
    """
    from analysis.waxs import damnit

    return damnit.jungfrau_waxs_from_file(proposal, run_no, "jf1")


@Variable("JF2 I(q)", data="proc", cluster=True, tags=["offline"])
def jungfrau_waxs_jf2(
    run,
    prep: "var#jungfrau_preprocessing",
    proposal: "meta#proposal",
    run_no: "meta#run_number",
):
    """I(q) per (trainId, cellId) for JUNGFRAU 2, read from its file."""
    from analysis.waxs import damnit

    return damnit.jungfrau_waxs_from_file(proposal, run_no, "jf2")


@Variable("JUNGFRAU I(q) overview", data="proc", cluster=True, tags=["offline"])
def jungfrau_waxs_overview(
    run, jf1: "var#jungfrau_waxs_jf1" = None, jf2: "var#jungfrau_waxs_jf2" = None
):
    from analysis.waxs import damnit

    return damnit.overview_figure(jf1, jf2)


@Variable("JUNGFRAU I(q) combined", data="proc", cluster=True, tags=["offline"])
def jungfrau_waxs_combined(
    run, jf1: "var#jungfrau_waxs_jf1" = None, jf2: "var#jungfrau_waxs_jf2" = None
):
    """Both detectors on one scale, jf2 fitted onto jf1 over their overlap.

    `attrs` carry the factor, the overlap it was fitted over and the reduced
    chi-squared of the residual. That chi-squared is a cross-check on the two
    PONIs only when the overlap carries a feature: on a run with no Bragg
    peaks a q error is absorbed by the factor instead of showing up here.
    Either detector may be missing (a dark run): then there is nothing to
    combine, and the cell is empty.
    """
    from analysis.waxs import damnit

    return damnit.combined_curve(jf1, jf2)


# ── Diagnostics (analysis.diagnostics; context/diagnostics.md) ──────────────────
# Frames used per detector, read straight from each pass's frame table. There is
# deliberately no var# dependency on the pass: asking for one of these never
# relaunches the cluster job (CLAUDE.md pitfall 26). The price is that a cell
# shows the file as it was when the cell ran, so refresh after a reprocess with
# `damnit reprocess --match "frames used"` (--match compares titles).


@Variable("AGIPD frames used", data="proc", tags=["diagnostics"])
def agipd_frames_used(run, proposal: "meta#proposal", run_no: "meta#run_number"):
    """OK frames over all rows of the agipd_saxs frame table, coloured by status.

    Green: all OK. Red: a WORKER_ERROR or LABEL_MISMATCH, or an unreadable file.
    Amber: anything else, e.g. an incomplete run or a file still being written.
    Skipped when the pass has not written the run.
    """
    from analysis.diagnostics import damnit

    return damnit.frames_used(proposal, run_no, "agipd")


@Variable("JF1 frames used", data="proc", tags=["diagnostics"])
def jf1_frames_used(run, proposal: "meta#proposal", run_no: "meta#run_number"):
    """As agipd_frames_used, for jungfrau_waxs_jf1.h5 (one row per lit cell)."""
    from analysis.diagnostics import damnit

    return damnit.frames_used(proposal, run_no, "jf1")


@Variable("JF2 frames used", data="proc", tags=["diagnostics"])
def jf2_frames_used(run, proposal: "meta#proposal", run_no: "meta#run_number"):
    """As agipd_frames_used, for jungfrau_waxs_jf2.h5 (one row per lit cell)."""
    from analysis.diagnostics import damnit

    return damnit.frames_used(proposal, run_no, "jf2")


# Photon statistics, read from agipd_preprocessing's file (context/xpcs-quality.md).
# The droplet hits come from DAMNIT's values in memory, because inside the cluster
# job this run's agipd_saxs is not in the per-run file until the job ends.


@Variable("Photon statistics", data="proc", cluster=True, tags=["diagnostics"])
def photon_statistics(
    run,
    prep: "var#agipd_preprocessing",
    saxs: "var#agipd_saxs",
    xgm: "var#xgm_intensity",
    transmission: "var#total_transmission",
    proposal: "meta#proposal",
    run_no: "meta#run_number",
):
    """k̄ and single-frame contrast per XPCS ROI and ring, over the droplet hits."""
    from analysis.diagnostics import damnit

    return damnit.photon_statistics(proposal, run_no, saxs, xgm, transmission)


@Variable("Photon statistics overview", data="proc", cluster=True,
          tags=["diagnostics"])  # fmt: skip
def photon_statistics_overview(run, stats: "var#photon_statistics"):
    from analysis.diagnostics import damnit

    return damnit.photon_statistics_overview(stats)


@Variable("Diagnostics overview", data="proc", cluster=True, tags=["diagnostics"])
def diagnostics_overview(
    run,
    saxs: "var#agipd_saxs",
    xgm: "var#xgm_intensity",
    transmission: "var#total_transmission",
    proposal: "meta#proposal",
    run_no: "meta#run_number",
):
    """SVD, χ² against the median and XGM scatter of the AGIPD curves, over hits."""
    from analysis.diagnostics import damnit

    return damnit.diagnostics_overview(proposal, run_no, saxs, xgm, transmission)


# The XPCS keep criterion in one cell (context/run-triage.md §4.6). It reads what
# DAMNIT and extra-speckle stored (TTCF, kbar.nc), without a var# dependency on
# the XPCS variables, so it shows the run as of its last processing: refresh with
# `damnit reprocess --match "XPCS score"`.


@Variable("XPCS score", data="proc", tags=["diagnostics"])
def xpcs_score(run, proposal: "meta#proposal", run_no: "meta#run_number"):
    """Hits, ROIs at k̄ ≥ 3e-3, best 30 s SNR with the fixed contrast."""
    from analysis.diagnostics import damnit

    return damnit.xpcs_score_cell(proposal, run_no)


# ── Droplet camera (analysis.camera; context/diagnostics.md §6) ──────────────────
# CAM5's images go to tape with raw, so each run keeps a labelled video in
# usr/cached_files/droplet_videos/. Both are raw variables: the camera is a raw
# source.


@Variable("Droplet video", cluster=True, tags=["diagnostics"])
def droplet_video(run, run_no: "meta#run_number"):
    """Every CAM5 image as r{run:04d}_cam5.mp4, 3× real time, labelled per frame.

    The cell holds the path and "N frames, G gaps"; N is counted by decoding
    the written file and must equal the number of trains with an image.
    """
    from analysis.diagnostics import damnit

    return damnit.droplet_video(run, run_no)


@Variable("Droplet first image", tags=["diagnostics"])
def droplet_first_image(run):
    """The run's first CAM5 image, reading only that one."""
    from analysis.diagnostics import damnit

    return damnit.droplet_first_image(run)

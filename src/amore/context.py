import sqlite3
from datetime import timedelta

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from analysis_helpers import plot_ellipse, process_droplet_batch
from damnit.context import Cell, Group, Variable
from extra.components import XGM, Scantool, XrayPulses
from extra.data import by_id


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


# ── Analysis columns, one Group per technique or diagnostic ──────────────────────
# A Group renames its variables "<group>.<var>" (agipd.saxs, jungfrau.jf1, ...) and
# titles them "<group title> <var title>", so a title match such as
# `damnit reprocess --match "frames used"` still finds them. Dependencies inside a
# group are "self#<var>", across groups "var#<group>.<var>". The names are read by
# name elsewhere: `analysis.damnit_names` is the one table of them.
#
# Every wrapper is thin on purpose: DAMNIT execs this file into a dict, so a
# function defined here cannot be pickled to the workers a pass spawns, and the
# physics belongs under `analysis` where it is tested.


@Group(tags=["AGIPD", "SAXS"])
class Agipd:
    """AGIPD SAXS: one read of proc (frame table, window sums, photon histograms).

    One variable reads AGIPD proc, once, and writes three files: the frame table
    (scratch/agipd_saxs/r####/agipd_saxs.h5), the per-pixel window sums and the
    per-frame photon statistics (usr/cached_files/agipd_pixel_sums, ..._photon_stats).
    Every other AGIPD column reads one of those files, so the table shows which
    variable computes and which only presents (user, 2026-10-14). A complete run
    is answered from its files without writing (CLAUDE.md pitfall 26).

    `saxs` holds I(q) in nm^-1, undivided. Runs processed before this pass
    existed hold A^-1 divided by I0, so clear the column for them rather than
    plotting across the boundary.
    """

    @Variable("preprocessing", data="proc", cluster=True, tags=["offline"])
    def preprocessing(self, run, proposal: "meta#proposal", run_no: "meta#run_number"):
        """The run's one read of AGIPD proc: frame table, window sums, photon stats.

        `run` is unused: the pass opens proc for the frames and raw for its own
        run checks. The cell is a status per product, green when all are
        complete and OK; a failure is a red cell, and each reader then says what
        its file lacks.
        """
        from analysis.saxs import damnit

        return damnit.preprocess(proposal, run_no)

    @Variable("I(q)", data="proc", cluster=True, tags=["offline"])
    def saxs(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        """I(q) per (trainId, pulseId), read from the frame table.

        Raises unless every frame is OK.
        """
        from analysis.saxs import damnit

        return damnit.agipd_saxs_from_file(proposal, run_no)

    @Variable("I(q) overview", data="proc", cluster=True, tags=["offline"])
    def iq_overview(self, run, grid: "self#saxs"):
        from analysis.saxs import damnit

        return damnit.overview_figure(grid)

    @Variable("frames used", data="proc", tags=["diagnostics"])
    def frames_used(self, run, proposal: "meta#proposal", run_no: "meta#run_number"):
        """OK frames over all rows of the frame table, coloured by status.

        There is deliberately no var# dependency on the pass: asking for this
        never relaunches the cluster job (CLAUDE.md pitfall 26). The price is
        that the cell shows the file as it was when it ran, so refresh after a
        reprocess with `damnit reprocess --match "frames used"` (--match
        compares titles).

        Green: all OK. Red: a WORKER_ERROR or LABEL_MISMATCH, or an unreadable
        file. Amber: anything else, e.g. an incomplete run or a file still being
        written. Skipped when the pass has not written the run.
        """
        from analysis.diagnostics import damnit

        return damnit.frames_used(proposal, run_no, "agipd")

    @Variable(
        "SVD and chi2",
        data="proc",
        cluster=True,
        tags=["diagnostics"],
    )
    def diagnostics(
        self,
        run,
        saxs: "self#saxs",
        xgm: "var#xgm_intensity",
        transmission: "var#total_transmission",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        """SVD, χ² against the median and XGM scatter of the AGIPD curves, over hits."""
        from analysis.diagnostics import damnit

        return damnit.diagnostics_overview(proposal, run_no, saxs, xgm, transmission)


agipd = Agipd(title="AGIPD", sep=" ")


@Group(tags=["JUNGFRAU", "WAXS"])
class Jungfrau:
    """JUNGFRAU WAXS (analysis.waxs; context/jungfrau-waxs-integrator).

    One variable runs the pass for both detectors, one after the other: the two
    JUNGFRAU-500Ks have different geometries, different static masks and
    different q ranges, so each gets its own PONI and its own output file under
    scratch/jungfrau_waxs/r{run:04d}/. The per-detector columns read those
    files. They are combined only in the overview, and there as two traces with
    their overlap visible, never concatenated. A run with no lit cell writes no
    file: its column is skipped, and the preprocessing cell says "dark".
    """

    @Variable("preprocessing", data="proc", cluster=True, tags=["offline"])
    def preprocessing(self, run, proposal: "meta#proposal", run_no: "meta#run_number"):
        """Both detectors' passes: a status per detector (OK counts, dark, failed)."""
        from analysis.waxs import damnit

        return damnit.preprocess(proposal, run_no)

    @Variable("JF1 I(q)", data="proc", cluster=True, tags=["offline"])
    def jf1(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        """I(q) per (trainId, cellId) for JUNGFRAU 1, read from its file.

        There is no pulse axis: JUNGFRAU's reader has no pulse ids, and which of
        a train's X-ray pulses each memory cell sampled is unresolved, so the
        cell id is stored and no pulse id is invented.
        """
        from analysis.waxs import damnit

        return damnit.jungfrau_waxs_from_file(proposal, run_no, "jf1")

    @Variable("JF2 I(q)", data="proc", cluster=True, tags=["offline"])
    def jf2(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        """I(q) per (trainId, cellId) for JUNGFRAU 2, read from its file."""
        from analysis.waxs import damnit

        return damnit.jungfrau_waxs_from_file(proposal, run_no, "jf2")

    @Variable("I(q) overview", data="proc", cluster=True, tags=["offline"])
    def iq_overview(self, run, jf1: "self#jf1" = None, jf2: "self#jf2" = None):
        from analysis.waxs import damnit

        return damnit.overview_figure(jf1, jf2)

    @Variable("I(q) combined", data="proc", cluster=True, tags=["offline"])
    def combined(self, run, jf1: "self#jf1" = None, jf2: "self#jf2" = None):
        """Both detectors on one scale, jf2 fitted onto jf1 over their overlap.

        `attrs` carry the factor, the overlap it was fitted over and the reduced
        chi-squared of the residual. That chi-squared is a cross-check on the
        two PONIs only when the overlap carries a feature: on a run with no
        Bragg peaks a q error is absorbed by the factor instead of showing up
        here. Either detector may be missing (a dark run): then there is
        nothing to combine, and the cell is empty.
        """
        from analysis.waxs import damnit

        return damnit.combined_curve(jf1, jf2)

    @Variable("JF1 frames used", data="proc", tags=["diagnostics"])
    def jf1_frames_used(
        self, run, proposal: "meta#proposal", run_no: "meta#run_number"
    ):
        """As agipd.frames_used, for jungfrau_waxs_jf1.h5 (one row per lit cell)."""
        from analysis.diagnostics import damnit

        return damnit.frames_used(proposal, run_no, "jf1")

    @Variable("JF2 frames used", data="proc", tags=["diagnostics"])
    def jf2_frames_used(
        self, run, proposal: "meta#proposal", run_no: "meta#run_number"
    ):
        """As agipd.frames_used, for jungfrau_waxs_jf2.h5 (one row per lit cell)."""
        from analysis.diagnostics import damnit

        return damnit.frames_used(proposal, run_no, "jf2")


jungfrau = Jungfrau(title="JUNGFRAU", sep=" ")


@Group(tags=["XPCS"])
class Xpcs:
    """Hit-train XPCS (analysis.xpcs; context/xpcs-hits.md), the only XPCS pass.

    It replaced the old extra-speckle pipeline columns, whose figures
    `analysis.xpcs.figures` reproduces from the hit store. One variable reads
    proc, once more than the AGIPD pass: the hit rule needs this run's whole
    agipd.saxs (and the flux), so the XPCS pass cannot share the first read. It
    extracts the hit trains into usr/cached_files/agipd_ring_photons (which
    outlives proc) and writes their exact sums to usr/cached_files/xpcs_hits.
    Every other column reads that file. The hits come from DAMNIT's values in
    memory, because inside the cluster job this run's agipd.saxs is not in the
    per-run file until the job ends. A complete run is answered from its files
    without writing (CLAUDE.md pitfall 26).
    """

    @Variable("preprocessing", data="proc", cluster=True, tags=["offline"])
    def preprocessing(
        self,
        run,
        saxs: "var#agipd.saxs",
        xgm: "var#xgm_intensity",
        transmission: "var#total_transmission",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
        hits: "var#hits.summary" = None,
    ):
        """Extract the droplet-hit trains and write their sums; a status, not a result.

        Green when every hit is usable, amber when some are final non-OK, red
        when the pass failed, grey when the run has no hit. `run` is unused.
        `hits` is only there for the order: the hit rule runs, and its summary
        cell exists, before this pass (it finds the same hits again).
        """
        from analysis.xpcs import damnit

        return damnit.preprocess(proposal, run_no, saxs, xgm, transmission)

    @Variable("g2", data="proc", cluster=True, tags=["offline"])
    def g2(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
        rep: "var#rep_rate" = None,
    ):
        """Pooled g2, baseline and c2 per ring with jackknife errors, from the file.

        `rep` is DAMNIT's rep rate [MHz]: a pulse pattern that implies another
        rate refuses to make a lag axis.
        """
        from analysis.xpcs import damnit

        return damnit.g2_from_file(proposal, run_no, rep)

    @Variable("TTCF", data="proc", cluster=True, tags=["offline"])
    def ttcf(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        """The pooled two-time maps raw, off and c2 per ring, from the file."""
        from analysis.xpcs import damnit

        return damnit.ttcf_from_file(proposal, run_no)

    @Variable("overview", data="proc", cluster=True, tags=["offline"])
    def overview(self, run, g2: "self#g2", ttcf: "self#ttcf"):
        from analysis.xpcs import damnit

        return damnit.overview_figure(g2, ttcf)

    # The old pipeline's figures and data columns, from the hit store
    # (analysis.xpcs.figures).

    @Variable("q-rings", data="proc", cluster=True, tags=["offline"])
    def q_rings(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        from analysis.xpcs import damnit

        return damnit.q_rings_figure(proposal, run_no)

    @Variable("beamcenter", data="proc", cluster=True, tags=["offline"])
    def beamcenter(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        from analysis.xpcs import damnit

        return damnit.beamcenter_figure(proposal, run_no)

    @Variable("outliers", data="proc", cluster=True, tags=["offline"])
    def outliers(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        from analysis.xpcs import damnit

        return damnit.outliers_figure(proposal, run_no)

    @Variable("intensity outliers", data="proc", cluster=True, tags=["offline"])
    def intensity_outliers(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        from analysis.xpcs import damnit

        return damnit.intensity_outliers_figure(proposal, run_no)

    @Variable("kbar", data="proc", cluster=True, tags=["offline"])
    def kbar(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        from analysis.xpcs import damnit

        return damnit.kbar_figure(proposal, run_no)

    @Variable("correction", data="proc", cluster=True, tags=["offline"])
    def correction(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        from analysis.xpcs import damnit

        return damnit.correction_figure(proposal, run_no)

    @Variable("TTCF's", data="proc", cluster=True, tags=["offline"])
    def ttcfs(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        from analysis.xpcs import damnit

        return damnit.ttcfs_figure(proposal, run_no)

    @Variable("g2 figure", data="proc", cluster=True, tags=["offline"])
    def figure_g2(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
        rep: "var#rep_rate" = None,
    ):
        from analysis.xpcs import damnit

        return damnit.g2_figure(proposal, run_no, rep)

    @Variable("g2 fit", data="proc", cluster=True, tags=["offline"])
    def g2_fit(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
        rep: "var#rep_rate" = None,
    ):
        from analysis.xpcs import damnit

        return damnit.g2_fit_figure(proposal, run_no, rep)

    @Variable("SAXS overview", data="proc", cluster=True, tags=["offline"])
    def saxs_figure(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        from analysis.xpcs import damnit

        return damnit.saxs_figure(proposal, run_no)

    @Variable("TTCF per train", data="proc", cluster=True, tags=["offline"])
    def ttcf_per_train(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        """Each hit train's own normalised two-time map, raw minus off (0.65 MB each).

        The old pipeline's `XPCS TTCF` column, for the hit trains only.
        """
        from analysis.xpcs import damnit

        return damnit.legacy_ttcf(proposal, run_no)

    @Variable("mean dataset", data="proc", cluster=True, tags=["offline"])
    def mean_dataset(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
        rep: "var#rep_rate" = None,
    ):
        """ttcf_mean and g2, as the old pipeline's mean dataset (pooled)."""
        from analysis.xpcs import damnit

        return damnit.legacy_mean_dataset(proposal, run_no, rep)

    @Variable("SAXS", data="proc", cluster=True, tags=["offline"])
    def saxs_curves(
        self,
        run,
        prep: "self#preprocessing",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        """I(q) on the XPCS annulus, as the old pipeline's SAXS column."""
        from analysis.xpcs import damnit

        return damnit.legacy_saxs(proposal, run_no)

    @Variable("frames used", data="proc", tags=["diagnostics"])
    def frames_used(self, run, proposal: "meta#proposal", run_no: "meta#run_number"):
        """Hits used over all hits of the XPCS results file, coloured by status.

        Like the other frames-used cells there is no var# on the pass, so asking
        for it cannot relaunch the cluster job; refresh it after a reprocess
        with `damnit reprocess --match "frames used"`. Grey: the run has no hit.
        Skipped when the pass has not written the run.
        """
        from analysis.xpcs import damnit

        return damnit.frames_used(proposal, run_no)

    @Variable("photon statistics", data="proc", cluster=True, tags=["diagnostics"])
    def photon_statistics(
        self,
        run,
        prep: "var#agipd.preprocessing",
        saxs: "var#agipd.saxs",
        xgm: "var#xgm_intensity",
        transmission: "var#total_transmission",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
        hits: "var#hits.summary" = None,
    ):
        """k̄ and single-frame contrast per XPCS ROI and ring, over the droplet hits.

        Read from agipd.preprocessing's photon histograms (context/xpcs-quality.md).
        The hits come from DAMNIT's values in memory, because inside the cluster
        job this run's agipd.saxs is not in the per-run file until the job ends.
        """
        from analysis.diagnostics import damnit

        return damnit.photon_statistics(proposal, run_no, saxs, xgm, transmission)

    @Variable(
        "photon statistics overview", data="proc", cluster=True, tags=["diagnostics"]
    )
    def photon_statistics_overview(self, run, stats: "self#photon_statistics"):
        from analysis.diagnostics import damnit

        return damnit.photon_statistics_overview(stats)

    @Variable("score", data="proc", tags=["diagnostics"])
    def score(self, run, proposal: "meta#proposal", run_no: "meta#run_number"):
        """Hits, ROIs at k̄ ≥ 3e-3, best 30 s SNR with the fixed contrast.

        The XPCS keep criterion in one cell (context/run-triage.md §4.6), read
        from the hit store without a var# dependency on the XPCS variables, so
        it shows the run as of its last processing: refresh with
        `damnit reprocess --match "XPCS score"`.
        """
        from analysis.diagnostics import damnit

        return damnit.xpcs_score_cell(proposal, run_no)


xpcs = Xpcs(title="XPCS", sep=" ")


@Group(tags=["Droplet", "diagnostics"])
class Hits:
    """Droplet-hit recognition (analysis.triage.classify; context/run-triage.md §4.4).

    What the hit rule decided for the run, so a surprising XPCS result can be
    traced to its hits. Both read DAMNIT's values in memory, like the XPCS pass:
    inside a job this run's agipd.saxs is not in the per-run file yet.
    """

    @Variable("summary", tags=["diagnostics"])
    def summary(
        self,
        run,
        saxs: "var#agipd.saxs",
        xgm: "var#xgm_intensity",
        transmission: "var#total_transmission",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
    ):
        """Hit, edge and off trains of the run; grey when it has no hit."""
        from analysis.triage import damnit

        return damnit.hit_summary_cell(proposal, run_no, saxs, xgm, transmission)

    @Variable("geometry", tags=["diagnostics"])
    def geometry(
        self,
        run,
        saxs: "var#agipd.saxs",
        xgm: "var#xgm_intensity",
        transmission: "var#total_transmission",
        proposal: "meta#proposal",
        run_no: "meta#run_number",
        summary: "self#summary" = None,
        fit: "var#droplet_fit" = None,
    ):
        """Mesh scan map with the droplet's geometric approximation, f against time.

        Raw variable: the hexapod positions are a raw source. Static runs get
        the class timeline only.
        """
        from analysis.triage import damnit

        return damnit.hit_geometry_figure(
            proposal, run_no, saxs, xgm, transmission, run=run, droplet_fit=fit
        )


hits = Hits(title="Hits", sep=" ")


@Group(tags=["Droplet", "diagnostics"])
class Camera:
    """The droplet camera (analysis.camera; context/diagnostics.md §6).

    CAM5's images go to tape with raw, so each run keeps a labelled video in
    usr/cached_files/droplet_videos/. Both are raw variables: the camera is a
    raw source.
    """

    @Variable("video", cluster=True, tags=["diagnostics"])
    def video(self, run, run_no: "meta#run_number"):
        """Every CAM5 image as r{run:04d}_cam5.mp4, 3× real time, labelled per frame.

        The cell holds the path and "N frames, G gaps"; N is counted by decoding
        the written file and must equal the number of trains with an image.
        """
        from analysis.diagnostics import damnit

        return damnit.droplet_video(run, run_no)

    @Variable("first image", tags=["diagnostics"])
    def first_image(self, run):
        """The run's first CAM5 image, reading only that one."""
        from analysis.diagnostics import damnit

        return damnit.droplet_first_image(run)


camera = Camera(title="Camera", sep=" ")

"""The levitated droplets of the beamtime and the t/t* at which they are compared.

A droplet is a series of runs on one levitated drop, from injection to its end: its
V0 is the first train of its first kept run and its t* comes from the D² fit over its
kept runs (`analysis.cache.droplet_timeline`). The catalogue is the run triage's
series (`scripts/run_triage.py`) with the runs left out and why. Its ids are the
first run of the series, ``s<run>``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import xarray as xr

from analysis.utils import droplet_composition

__all__ = [
    "CONDITIONS",
    "DROPLETS",
    "FERRITIN_MG_PER_ML",
    "MATCH_T",
    "MATCH_TOLERANCE",
    "NEAR_ZERO_BLOCKS",
    "PROPOSAL",
    "REPRESENTATIVE",
    "Droplet",
    "by_id",
    "condition_title",
    "groups",
]

PROPOSAL = 10400

#: Ferritin concentration of every sample at injection (Sigma F4503).
FERRITIN_MG_PER_ML = 50.0

#: Figure titles per sample. They are set in LaTeX, where an unescaped % starts a
#: comment.
CONDITIONS = {
    "ferritin50": "ferritin 50 mg/ml, no PEG",
    "ferritin50-peg1k": r"ferritin 50 mg/ml + PEG 1K 5\,\% w/v",
    "ferritin50-peg6k": r"ferritin 50 mg/ml + PEG 6K 5\,\% w/v",
}

#: The t/t* at which droplets are compared: a droplet shows its nearest SAXS block,
#: or pools its hit trains, within MATCH_TOLERANCE of each (about half the 0.13-0.18
#: spacing of the blocks). Pure ferritin ends near t/t* = 2.1, hence 2.0 as the last.
MATCH_T = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0)
MATCH_TOLERANCE = 0.08


@dataclass(frozen=True)
class Droplet:
    """One droplet: its runs, the runs left out and why, and how it was probed.

    ``mesh`` marks SSHEX fly scans, whose curves average only the trains that hit
    the droplet. ``figures`` is False for a droplet that only enters the V/V0
    comparison.
    """

    id: str
    sample: str
    runs: range
    peg_percent_wv: float
    mesh: bool = False
    excluded: dict[int, str] = field(default_factory=dict)
    figures: bool = True

    @property
    def kept_runs(self) -> list[int]:
        return [r for r in self.runs if r not in self.excluded]

    @property
    def name(self) -> str:
        """``<id>_<sample>_r<first>-<last>``, the droplet's figure folder."""
        runs = self.kept_runs
        return f"{self.id}_{self.sample}_r{runs[0]}-{runs[-1]}"

    @property
    def run_label(self) -> str:
        runs = self.kept_runs
        return f"r{runs[0]}-{runs[-1]}"

    def ferritin_mg_per_ml(self, timeline: xr.Dataset) -> xr.DataArray:
        """Ferritin concentration per train of `timeline`, from its volume over V0."""
        return xr.DataArray(
            droplet_composition(
                timeline.volume_mm3,
                timeline.volume_mm3[0].values,
                ferritin_mg_per_ml_initial=FERRITIN_MG_PER_ML,
                peg_percent_wv_initial=self.peg_percent_wv,
            ).ferritin_mg_per_ml,
            dims="trainId",
            coords={"trainId": timeline.trainId},
        )


_PEG6K, _NO_PEG, _PEG1K = "ferritin50-peg6k", "ferritin50", "ferritin50-peg1k"

#: Every droplet after r240, then those that only enter the V/V0 comparison. Left
#: out: the silica references (r288, r382-385, r389), r463 (exploded) and the buffer
#: r464-466 (it is the background). s289's attenuated runs and s402 (attenuator 0.6)
#: have no flux correction in the SAXS notebook. The mesh droplets before r240 carry
#: DAMNIT's labels, unchecked; s088, s093, s129, s134, s140 and s144 are left out
#: because their D² fit finds no break (V/V0 0.55-1.6 at the fitted t*, or t* at the
#: end of the data), so t/t* means nothing for them.
DROPLETS = (
    Droplet("s240", _PEG6K, range(240, 258), 5.0, mesh=True),
    Droplet("s258", _PEG6K, range(258, 276), 5.0, mesh=True),
    Droplet(
        "s289",
        _PEG6K,
        range(289, 308),
        5.0,
        mesh=True,
        excluded={
            289: "attenuator 0.041, no flux correction",
            307: "attenuator 0.0047, no flux correction",
        },
    ),
    # its last two runs, r336-337, are static; a static run keeps every train
    Droplet("s308", _NO_PEG, range(308, 338), 0.0, mesh=True),
    Droplet(
        "s338",
        _NO_PEG,
        range(338, 370),
        0.0,
        excluded={
            349: "no droplet fit",
            353: "dark",
            354: "dark",
            355: "dark",
            356: "left out by clean_range",
        },
    ),
    Droplet("s370", _NO_PEG, range(370, 382), 0.0),
    Droplet("s390", _NO_PEG, range(390, 394), 0.0),
    Droplet("s394", _NO_PEG, range(394, 401), 0.0),
    Droplet("s409", _PEG1K, range(409, 420), 5.0),
    Droplet("s423", _PEG6K, range(423, 433), 5.0),
    Droplet("s433", _PEG6K, range(433, 443), 5.0),
    Droplet(
        "s443", _PEG6K, range(443, 453), 5.0, excluded={447: "left out by clean_range"}
    ),
    Droplet("s453", _PEG6K, range(453, 463), 5.0),
    Droplet("s467", _PEG6K, range(467, 477), 5.0),
    Droplet("s477", _PEG6K, range(477, 487), 5.0),
    Droplet("s487", _PEG6K, range(487, 497), 5.0),
    Droplet("s070", _PEG6K, range(70, 88), 5.0, mesh=True, figures=False),
    Droplet("s168", _PEG6K, range(168, 186), 5.0, mesh=True, figures=False),
    Droplet("s186", _PEG6K, range(186, 204), 5.0, mesh=True, figures=False),
    Droplet("s204", _PEG6K, range(204, 222), 5.0, mesh=True, figures=False),
    Droplet("s222", _PEG6K, range(222, 240), 5.0, mesh=True, figures=False),
    Droplet("s402", _NO_PEG, range(402, 409), 0.0, figures=False),
)

#: The droplet that stands for a (sample, mesh) group where its droplets vary too
#: much to pool (user's choice). A group not listed with one droplet uses that one.
REPRESENTATIVE = {
    (_PEG6K, True): "s258",
    (_PEG6K, False): "s443",
    (_NO_PEG, False): "s394",
}

#: SAXS blocks the t/t* matching passes over, (run, first trainId) -> why. They are
#: the only blocks in any curves.nc whose mean I over q 0.3-0.4 is under a tenth of
#: the median of up to two blocks on each side.
NEAR_ZERO_BLOCKS = {
    (357, 2637483940): "near zero: I(q 0.3-0.4) 0.8 % of its neighbours'",
    (477, 2637914817): "near zero: I(q 0.3-0.4) 1.3 % of its neighbours'",
}


def by_id(droplet_id: str) -> Droplet:
    for droplet in DROPLETS:
        if droplet.id == droplet_id:
            return droplet
    raise KeyError(f"no droplet {droplet_id!r} in analysis.droplets.DROPLETS")


def groups() -> list[tuple[str, bool]]:
    """The (sample, mesh) groups of the figure droplets, in CONDITIONS order."""
    present = {(d.sample, d.mesh) for d in DROPLETS if d.figures}
    return [(s, m) for s in CONDITIONS for m in (False, True) if (s, m) in present]


def condition_title(sample: str, mesh: bool) -> str:
    return CONDITIONS[sample] + (", mesh scans (hit trains)" if mesh else "")

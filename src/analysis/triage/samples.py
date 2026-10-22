"""The sample each run really holds.

DAMNIT's ``sample`` column is myMdC's label and is stale for several runs. The
corrections and the runs with a background role live in `OVERRIDES`, each with the
source it was taken from. Labels the user confirmed are not repeated here: an
override range is applied blindly, and a confirming range drawn too wide would
relabel the runs inside it that hold something else (r382-385 and r389, silica, sit
inside the ferritin block r336-408).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "CANONICAL_LABELS",
    "CHECKED_LABEL_CHANGES",
    "DROPLET_SAMPLES",
    "OVERRIDES",
    "REFERENCE_SAMPLES",
    "Sample",
    "SampleOverride",
    "effective_sample",
    "override_for",
]

#: DAMNIT label -> canonical sample name.
CANONICAL_LABELS: dict[str, str] = {
    "Ferritin_50_PEG6K_5p": "ferritin50_peg6k",
    "Ferritin_50_PEG1K_5p": "ferritin50_peg1k",
    "Ferritin_50": "ferritin50",
    "Silica NP": "silica_np",
    "LaB6": "lab6",
    "Air": "air",
    "LUDOX": "ludox",
    "No Sample": "none",
    "Acqueous solution of ferritin and PEG": "ferritin_peg_commissioning",
}
#: Standards. The triage does not analyse them for content.
REFERENCE_SAMPLES = frozenset({"lab6", "air", "ludox", "silica_np"})
#: Samples levitated as a droplet, so the droplet fit and the series apply.
DROPLET_SAMPLES = frozenset(
    {
        "ferritin50",
        "ferritin50_peg6k",
        "ferritin50_peg1k",
        "peg6k_buffer",
        "silica_np",
    }
)


@dataclass(frozen=True, slots=True)
class SampleOverride:
    """A correction of DAMNIT's label, or a role, for runs ``first``..``last``."""

    first: int
    last: int
    sample: str
    source: str
    background_role: str | None = None
    exclude: str | None = None

    def covers(self, run: int) -> bool:
        return self.first <= run <= self.last


OVERRIDES: tuple[SampleOverride, ...] = (
    SampleOverride(53, 55, "air", "label; r55 confirmed by the user 2026-09-28",
                   background_role="empty_beam"),
    SampleOverride(289, 303, "ferritin50_peg6k", "user 2026-10-01 (label 'Silica NP')"),
    SampleOverride(308, 335, "ferritin50",
                   "user 2026-10-01 (labels 'Silica NP'/'PEG6K'; r308-309 checked)"),
    SampleOverride(463, 463, "peg6k_buffer", "memory 2026-09-29",
                   exclude="explodes at ~train 1550 (a bubble); a fragment remains"),
    SampleOverride(464, 466, "peg6k_buffer", "user 2026-09-28 (label 'PEG6K')",
                   background_role="saxs_buffer_peg6k"),
)  # fmt: skip


#: Label changes across a continuous droplet volume that a person has checked, so
#: `series.Boundary.label_suspect` reports the verdict instead of asking again.
#: Keyed by (run before, run after).
CHECKED_LABEL_CHANGES: dict[tuple[int, int], str] = {
    (389, 390): (
        "user 2026-10-01: labels correct. r389 is silica NP, cut short before it "
        "evaporated; r390 is a fresh ferritin droplet that started at a similar volume"
    ),
}


@dataclass(frozen=True, slots=True)
class Sample:
    """What a run holds, and where that statement comes from."""

    name: str
    source: str
    label: str | None
    background_role: str | None = None
    exclude: str | None = None


def override_for(run: int) -> SampleOverride | None:
    """The override covering ``run``; raises if two do, which would be a table bug."""
    hits = [o for o in OVERRIDES if o.covers(run)]
    if len(hits) > 1:
        raise RuntimeError(f"r{run} is covered by {len(hits)} overrides: {hits}")
    return hits[0] if hits else None


def effective_sample(run: int, label: str | None) -> Sample:
    """The override for ``run`` if there is one, else DAMNIT's label, canonicalised.

    A label missing from `CANONICAL_LABELS` passes through unchanged, with the
    source saying so, rather than being guessed.
    """
    if (o := override_for(run)) is not None:
        return Sample(o.sample, o.source, label, o.background_role, o.exclude)
    if label is None:
        return Sample("unknown", "no DAMNIT label", None)
    if label in CANONICAL_LABELS:
        return Sample(CANONICAL_LABELS[label], "DAMNIT label", label)
    return Sample(label, "DAMNIT label (not canonicalised)", label)

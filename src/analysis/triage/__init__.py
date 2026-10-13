"""Run triage for p010400: what each run holds, and whether it earns red-box space.

Design and every threshold's source: ``context/run-triage.md``. The modules read
DAMNIT's database and per-run files read-only and never alter a stored number.

``samples``    the true sample behind DAMNIT's stale labels, and background roles
``stats``      robust spread, the multiple-testing cut, the empty band
``inventory``  one row per run: class, sizes, which DAMNIT products exist
``series``     consecutive runs grouped into droplets, with per-run droplet metrics
"""

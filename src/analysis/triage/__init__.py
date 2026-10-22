"""Run triage for p010400: what each run holds, and whether it earns red-box space.

The modules read DAMNIT's database and per-run files read-only and never alter a
stored number. They classify runs, find the droplet-hit trains, group runs into
droplets and rate each run's XPCS signal.
"""

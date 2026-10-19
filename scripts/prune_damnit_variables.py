#!/usr/bin/env python
"""Remove from DAMNIT's table every variable the live context no longer defines.

(not unit-tested, CLAUDE.md working rule 7.) Renaming columns into ``Group`` classes
(2026-10-19) left the old flat names in ``runs.sqlite``: their columns kept showing
the old results beside the new ones, with the same titles.

    python scripts/prune_damnit_variables.py             # dry run: what would go
    python scripts/prune_damnit_variables.py --delete    # back up runs.sqlite, delete

What is removed is the rows of ``variables`` and ``run_variables`` for each such name,
and the ``runs`` view is rebuilt, which is what the GUI's "Delete" does to the database.
**The per-run ``extracted_data/*.h5`` are not touched**: deleting inside 500 files of
2.2 TB that the listener and other users write is where an interrupted run corrupts a
file, and HDF5 does not give the space back without a repack. The old groups stay in
the files, unlisted.

Refuses a name the live context still defines, and keeps any variable with a ``type``
(a user-editable column made in the GUI). The names come from evaluating the live
context file with DAMNIT's own ``ctxrunner``. A backup of ``runs.sqlite`` is written
to ``backups/`` first, through sqlite's online backup, so it is consistent while others
write.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from pathlib import Path

from _common import report_header, stamp, write_report

AMORE = Path("/gpfs/exfel/exp/MID/202601/p010400/usr/Shared/amore")
RETRIES = 6
WAIT_S = 30.0


def context_names(live_dir: Path) -> set[str]:
    """Every variable name the live context defines, groups expanded as DAMNIT does."""
    import damnit

    support = Path(damnit.__file__).parent / "ctxsupport"
    sys.path[:0] = [str(support), str(live_dir)]
    import ctxrunner

    code = (live_dir / "context.py").read_text()
    return set(ctxrunner.build_context_from_code(code, "context.py").vars)


def stale_variables(db: sqlite3.Connection, keep: set[str]):
    """``(name, title, type)`` of every database variable not in ``keep``."""
    rows = db.execute(
        "SELECT name, title, type FROM variables ORDER BY name"
    ).fetchall()
    return [row for row in rows if row[0] not in keep]


def backup(source: Path, target: Path) -> None:
    """A consistent copy of a database other processes are writing."""
    src = sqlite3.connect(f"file:{source}?mode=ro", uri=True, timeout=120)
    dst = sqlite3.connect(target)
    try:
        src.backup(dst)
    finally:
        dst.close()
        src.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--delete", action="store_true", help="really delete")
    parser.add_argument("--live-dir", type=Path, default=AMORE)
    parser.add_argument("--json", type=Path, help="where to write the report")
    args = parser.parse_args()

    db_path = args.live_dir / "runs.sqlite"
    keep = context_names(args.live_dir)
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, timeout=120)
    try:
        stale = stale_variables(con, keep)
        in_db = {row[0] for row in con.execute("SELECT name FROM variables")}
    finally:
        con.close()
    editable = [row for row in stale if row[2]]
    remove = [row for row in stale if not row[2]]
    report = report_header(
        script="prune_damnit_variables",
        delete=args.delete,
        context_variables=len(keep),
        database_variables=len(in_db),
        remove=[list(row) for row in remove],
        kept_user_editable=[list(row) for row in editable],
        context_missing_from_database=sorted(keep - in_db),
    )
    for name, title, _ in remove:
        print(f"{'delete' if args.delete else 'would delete'} {name}  ({title})")
    for name, title, kind in editable:
        print(f"keep {name} ({title}): type {kind}, a user-editable column")
    print(f"{len(remove)} of {len(in_db)} database variables are not in the context")
    if not args.delete or not remove:
        write_report(report, args.json, "prune_damnit_variables")
        return 0

    target = args.live_dir / "backups" / f"runs_before_prune_{stamp()}.sqlite"
    backup(db_path, target)
    print(f"backup {target} ({target.stat().st_size / 1e6:.0f} MB)")
    report["backup"] = str(target)

    from damnit.backend.db import DamnitDB

    db = DamnitDB(db_path)
    deleted, failed = [], []
    for name, _, _ in remove:
        for attempt in range(RETRIES):
            try:
                db.delete_variable(name)
                deleted.append(name)
                break
            except sqlite3.OperationalError as error:
                if attempt == RETRIES - 1:
                    failed.append([name, str(error)])
                else:
                    time.sleep(WAIT_S)
    report |= {"deleted": deleted, "failed": failed}
    print(f"deleted {len(deleted)}; failed {len(failed)} {failed}")
    write_report(report, args.json, "prune_damnit_variables")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

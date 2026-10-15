#!/bin/bash
# Reprocess every run through DAMNIT after the pass outputs were deleted by hand.
#
#   bash scripts/reprocess_all.sh launch [--dry-run] [-n CONCURRENT]
#   bash scripts/reprocess_all.sh status
#   bash scripts/reprocess_all.sh refresh
#   bash scripts/reprocess_all.sh xpcs [-n CONCURRENT]
#
# launch   AGIPD, jf1 and jf2 for every run with a proc directory that DAMNIT
#          knows, through `reprocess_safe.py start` (busy-run and lock checks; it
#          builds the `--match` titles). Not `start all`: that lists the runs from
#          the output directories, which are gone. -n caps the jobs DAMNIT runs at
#          once (default: the database's `concurrent_jobs`).
# status   `reprocess_safe.py status` over the same runs: complete / missing /
#          INCOMPLETE per run and detector. (`check` needs the backups `start`
#          makes; with the outputs deleted there are none.)
# refresh  once no DAMNIT job is left: the non-cluster cells that read the pass
#          files instead of depending on them ("frames used", "XPCS score").
#          Run with launch they would see no file yet.
# xpcs     extra-speckle's XPCS pipeline. Not part of launch: it overwrites
#          scratch/xpcs/r*/ in place (no lock, no backup) and those files are
#          Leonau's. Run `refresh` again after it, for the XPCS score.
#
# Not unit-tested (CLAUDE.md working rule 7).
set -euo pipefail

PROP=/gpfs/exfel/exp/MID/202601/p010400
AMORE=$PROP/usr/Shared/amore
PYTHON=$PROP/usr/Software/MID-10400/.venv/bin/python
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

command=${1:?usage: reprocess_all.sh launch|status|refresh|xpcs [--dry-run] [-n N]}
shift
dry_run=()
concurrent=()
while (($#)); do
    case $1 in
        --dry-run) dry_run=(--dry-run) ;;
        -n) concurrent=(--concurrent-jobs "$2"); shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
    shift
done

# Runs with a proc directory that are also in DAMNIT's run table.
mapfile -t runs < <(
    comm -12 \
        <(ls "$PROP/proc" | sed -n 's/^r0*\([0-9][0-9]*\)$/\1/p' | sort) \
        <(sqlite3 -readonly "$AMORE/runs.sqlite" "select distinct run from run_info" | sort) |
        sort -n
)
echo "${#runs[@]} runs: ${runs[0]}-${runs[-1]}"

damnit() {
    # shellcheck disable=SC1091
    source /etc/profile.d/modules.sh
    module load exfel damnit/stable
    command damnit "$@"
}

case $command in
    launch)
        cd "$HERE"
        EXTRA_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" reprocess_safe.py start \
            "${runs[@]}" -d agipd -d jf1 -d jf2 "${dry_run[@]}" "${concurrent[@]}"
        ;;
    status)
        cd "$HERE"
        EXTRA_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PYTHON" reprocess_safe.py status \
            "${runs[@]}" -d agipd -d jf1 -d jf2 -j 8
        ;;
    refresh)
        if squeue --clusters=all --me --noheader -o '%j' | grep -q damnit; then
            echo "DAMNIT jobs are still queued or running; refresh once they end" >&2
            exit 1
        fi
        damnit reprocess --in "$AMORE" --match 'frames used' --match 'XPCS score' \
            "${runs[@]}"
        ;;
    xpcs)
        damnit reprocess --in "$AMORE" --match 'XPCS' "${concurrent[@]}" "${runs[@]}"
        ;;
    *)
        echo "unknown command $command" >&2
        exit 2
        ;;
esac

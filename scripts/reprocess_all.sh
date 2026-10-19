#!/bin/bash
# Reprocess every run through DAMNIT after the pass outputs were deleted by hand.
#
#   bash scripts/reprocess_all.sh launch [--dry-run] [-n CONCURRENT]
#   bash scripts/reprocess_all.sh status
#   bash scripts/reprocess_all.sh refresh
#   bash scripts/reprocess_all.sh xpcs-hits [--dry-run] [-n CONCURRENT] [RUNS...]
#
# Order: launch (and let every job finish) -> xpcs-hits -> refresh. The
# hit-train XPCS variable depends on agipd_saxs, so xpcs-hits goes only after
# the AGIPD output is complete, and refresh goes last so the "frames used" cells
# (including "XPCS hits frames used") read finished files.
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
#          files instead of depending on them ("frames used", "XPCS score", and
#          the "Hits" cells, which read agipd.saxs).
#          Run with launch they would see no file yet.
# xpcs-hits the hit-train XPCS pass (analysis.xpcs), through `reprocess_safe.py
#          start -d xpcs`: it prints the AGIPD and XPCS status, then refuses
#          unless every run's AGIPD output is complete, a job or lock is absent,
#          and no backup is left; moves xpcs_hits.h5 aside (never the photon
#          store). With RUNS, only those runs; without, every run above, so one
#          run without a complete AGIPD file refuses all: name the runs then.
#          Afterwards `reprocess_safe.py check RUNS -d xpcs`, then `cleanup`.
#
# Not unit-tested (CLAUDE.md working rule 7).
set -euo pipefail

PROP=/gpfs/exfel/exp/MID/202601/p010400
AMORE=$PROP/usr/Shared/amore
PYTHON=$PROP/usr/Software/MID-10400/.venv/bin/python
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

command=${1:?usage: reprocess_all.sh launch|status|refresh|xpcs-hits [--dry-run] [-n N] [RUNS...]}
shift
dry_run=()
concurrent=()
only=()
while (($#)); do
    case $1 in
        --dry-run) dry_run=(--dry-run) ;;
        -n) concurrent=(--concurrent-jobs "$2"); shift ;;
        [0-9]*) only+=("$1") ;;
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
        damnit reprocess --in "$AMORE" --match 'frames used' --match 'XPCS score' --match 'Hits ' \
            "${runs[@]}"
        ;;
    xpcs-hits)
        cd "$HERE"
        targets=("${runs[@]}")
        if ((${#only[@]})); then targets=("${only[@]}"); fi
        export EXTRA_NUM_THREADS=1 OMP_NUM_THREADS=1
        "$PYTHON" reprocess_safe.py status "${targets[@]}" -d agipd -d xpcs -j 8
        "$PYTHON" reprocess_safe.py start "${targets[@]}" -d xpcs "${dry_run[@]}" \
            "${concurrent[@]}"
        ;;
    *)
        echo "unknown command $command" >&2
        exit 2
        ;;
esac

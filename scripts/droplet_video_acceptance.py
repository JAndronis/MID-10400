#!/usr/bin/env python
"""Phase E acceptance: droplet videos (``analysis.camera``; context/diagnostics.md §6).

Per run: the CAM5 facts, then the video written to its real place
(``usr/cached_files/droplet_videos/r{run:04d}_cam5.mp4``) with the wall time and
size, then the frame count three ways: images in the run, frames decoded by our
ffmpeg (``count_frames``), and frames decoded by the EuXFEL env's ffprobe (an
independent build), with the stream's size, codec, rate and comment.

    python scripts/droplet_video_acceptance.py --runs 423 318 --workers 16

Writes droplet_video_acceptance_<stamp>.json beside itself. Not unit-tested
(CLAUDE.md rule 7): run it on a node.
"""

from __future__ import annotations

from _common import report_header, write_report  # noqa: E402

from analysis.threadenv import set_thread_env

set_thread_env()

import argparse  # noqa: E402
import json  # noqa: E402
import subprocess  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

from analysis.camera import (  # noqa: E402
    VIDEO_ROOT,
    camera_facts,
    count_frames,
    video_path,
    write_camera_video,
)

FFPROBE = Path(
    "/gpfs/exfel/sw/software/euxfel-environment-management/current-python-env/bin/ffprobe"
)


def ffprobe(path: Path) -> dict | None:
    """The stream as an independent ffprobe decodes it, or None without one."""
    if not FFPROBE.exists():
        return None
    out = subprocess.run(
        [str(FFPROBE), "-v", "error", "-count_frames", "-select_streams", "v:0",
         "-show_entries",
         "stream=nb_read_frames,width,height,codec_name,pix_fmt,r_frame_rate"
         ":format=duration:format_tags=comment",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout  # fmt: skip
    probe = json.loads(out)
    stream = probe["streams"][0]
    return {
        "frames": int(stream["nb_read_frames"]),
        "width": int(stream["width"]),
        "height": int(stream["height"]),
        "codec": stream["codec_name"],
        "pix_fmt": stream["pix_fmt"],
        "rate": stream["r_frame_rate"],
        "duration_s": float(probe["format"]["duration"]),
        "comment": probe["format"].get("tags", {}).get("comment"),
    }


def one_run(run_no: int, workers: int, root: Path) -> dict:
    from extra_data import open_run

    run = open_run(10400, run_no, data="raw")
    facts = camera_facts(run)
    out: dict = {"run": run_no, "facts": facts}
    if not facts["n_images"]:
        return out
    path = video_path(run_no, root=root)
    info = write_camera_video(run, path, label=f"r{run_no:04d}", workers=workers)
    started = time.perf_counter()
    decoded = count_frames(path)
    out.update(
        video={k: v for k, v in info.items() if k != "comment"},
        comment=info["comment"],
        count_frames=decoded,
        count_frames_s=round(time.perf_counter() - started, 2),
        ffprobe=ffprobe(path),
        bytes_per_frame=round(info["bytes"] / info["frames"], 1),
    )
    probe = out["ffprobe"]
    out["passed"] = bool(
        decoded == facts["n_images"] == info["frames"]
        and (probe is None or probe["frames"] == decoded)
    )
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, nargs="+", default=[423, 318])
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--root", type=Path, default=VIDEO_ROOT)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)
    t0 = time.perf_counter()
    results, errors = [], {}
    for run in args.runs:
        try:
            results.append(one_run(run, args.workers, args.root))
        except Exception as error:  # noqa: BLE001 - recorded per run
            errors[run] = f"{type(error).__name__}: {error}"
    passed = not errors and all(r.get("passed", False) for r in results)
    report = report_header(
        script="droplet_video_acceptance", elapsed_s=round(time.perf_counter() - t0, 1),
        workers=args.workers, root=str(args.root), ffprobe=str(FFPROBE),
        runs=results, errors=errors, passed=passed,
    )  # fmt: skip
    write_report(report, args.json, "droplet_video_acceptance")
    for r in results:
        v, probe = r.get("video", {}), r.get("ffprobe") or {}
        print(f"r{r['run']}: {r['facts'].get('n_images')} images -> "
              f"{r.get('count_frames')} decoded, ffprobe {probe.get('frames')}, "
              f"{v.get('bytes', 0) / 1e6:.1f} MB in {v.get('wall_s')} s, "
              f"gaps {v.get('gaps')}, limits {v.get('limits')}, "
              f"saturated {v.get('saturated_fraction')}")  # fmt: skip
    print("errors:", errors, "\npassed:", passed)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Timeline-preserving video frame extractor for SmarterLabelMe.

This is intended as a SECOND extractor alongside the legacy
smarter_labelme_video2frames tool.

Canonical filename:

    f00012345_t000412078.jpg

meaning:

    f00012345   = decoded source-frame index 12345 in the ORIGINAL video
    t000412078  = source-relative presentation time 412.078 seconds

The source-frame index is the canonical identity. The timestamp is included
for human readability and temporal annotation.

Key properties
--------------
* One full decode pass only.
* --dry-run is fast: it probes only container/stream metadata.
* Existing timeline-named images are preserved by default.
* Re-running later with --fps full fills in missing source frames without
  renaming existing images, so matching LabelMe JSON files remain valid.
* No CSV sidecar is required.
* Timestamp source:
    - decoder-reported presentation timestamps when OpenCV exposes them
      reliably;
    - otherwise a stable CFR fallback from ffprobe's average frame rate.
  The script reports which mode it is using.

Python 3.8 compatible.
"""

import argparse
import json
import math
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, Optional, Tuple

import cv2


FRAME_RE = re.compile(
    r"^f(?P<frame>\d{8})_t(?P<ms>\d{9})\.jpg$",
    re.IGNORECASE,
)


def require_program(name: str) -> str:
    path = shutil.which(name)
    if path is None:
        raise RuntimeError(
            "{} was not found in PATH. Please install ffmpeg/ffprobe.".format(name)
        )
    return path


def run_json(command):
    proc = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            "Command failed:\n{}\n\n{}".format(
                " ".join(command), proc.stderr.strip()
            )
        )

    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError("Could not parse ffprobe JSON output.") from exc


def parse_rate(value: Optional[str]) -> Optional[float]:
    if not value or value == "0/0":
        return None

    if "/" in value:
        num_s, den_s = value.split("/", 1)
        try:
            num = float(num_s)
            den = float(den_s)
        except ValueError:
            return None
        if den == 0:
            return None
        return num / den

    try:
        return float(value)
    except ValueError:
        return None


def probe_video(video: str, ffprobe: str) -> Dict:
    data = run_json(
        [
            ffprobe,
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            (
                "stream=width,height,avg_frame_rate,r_frame_rate,"
                "duration,nb_frames:"
                "format=duration"
            ),
            "-of",
            "json",
            video,
        ]
    )

    streams = data.get("streams", [])
    if not streams:
        raise RuntimeError("No video stream found in {}".format(video))

    stream = streams[0]
    fmt = data.get("format", {})

    duration = stream.get("duration")
    if duration in (None, "N/A"):
        duration = fmt.get("duration")

    avg_fps = parse_rate(stream.get("avg_frame_rate"))
    nominal_fps = parse_rate(stream.get("r_frame_rate"))

    nb_frames = stream.get("nb_frames")
    try:
        nb_frames_int = int(nb_frames) if nb_frames not in (None, "N/A") else None
    except (TypeError, ValueError):
        nb_frames_int = None

    return {
        "width": stream.get("width"),
        "height": stream.get("height"),
        "avg_fps": avg_fps,
        "nominal_fps": nominal_fps,
        "duration": float(duration) if duration not in (None, "N/A") else None,
        "nb_frames": nb_frames_int,
    }


def decoder_timestamp_is_usable(video: str, fallback_fps: float) -> bool:
    """
    Cheap capability test: decode only a few frames and see whether
    CAP_PROP_POS_MSEC behaves like a monotonic presentation timestamp.
    """
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return False

    values = []

    try:
        for _ in range(8):
            ok, _frame = cap.read()
            if not ok:
                break
            values.append(float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0)
    finally:
        cap.release()

    if len(values) < 2:
        return False

    for value in values:
        if not math.isfinite(value) or value < -1e-6:
            return False

    for a, b in zip(values, values[1:]):
        if b < a - 1e-6:
            return False

    # Some OpenCV/codec combinations return 0.0 for every frame.
    if max(values) - min(values) < 1e-6:
        return False

    # Very generous sanity check so VFR material can still use decoder PTS.
    expected = (len(values) - 1) / fallback_fps
    observed = values[-1] - values[0]
    if expected > 0 and not (0.2 * expected <= observed <= 5.0 * expected):
        return False

    return True


def make_filename(source_index: int, timestamp_sec: float) -> str:
    timestamp_ms = int(round(timestamp_sec * 1000.0))

    if timestamp_ms < 0:
        timestamp_ms = 0

    if source_index > 99999999:
        raise RuntimeError(
            "Source frame index {} exceeds the current 8-digit filename field.".format(
                source_index
            )
        )

    if timestamp_ms > 999999999:
        raise RuntimeError(
            "Timestamp {} ms exceeds the current 9-digit filename field.".format(
                timestamp_ms
            )
        )

    return "f{:08d}_t{:09d}.jpg".format(source_index, timestamp_ms)


def scan_existing_frames(output_dir: str) -> Dict[int, str]:
    """
    Return source-frame-index -> existing canonical filename.

    This allows an 8-fps directory to be expanded later without touching
    existing image/JSON basenames.
    """
    existing = {}

    if not os.path.isdir(output_dir):
        return existing

    for name in os.listdir(output_dir):
        match = FRAME_RE.match(name)
        if match is None:
            continue

        source_index = int(match.group("frame"))

        if source_index in existing and existing[source_index] != name:
            raise RuntimeError(
                "Multiple timeline images exist for source frame {}:\n"
                "  {}\n"
                "  {}".format(source_index, existing[source_index], name)
            )

        existing[source_index] = name

    return existing


def choose_fallback_fps(meta: Dict) -> float:
    fps = meta.get("avg_fps") or meta.get("nominal_fps")
    if fps is None or fps <= 0:
        raise RuntimeError(
            "Could not determine a usable source frame rate for timestamp fallback."
        )
    return float(fps)


def estimated_output_count(meta: Dict, fps_arg: str) -> Optional[int]:
    duration = meta.get("duration")

    if fps_arg.lower() == "full":
        if meta.get("nb_frames") is not None:
            return int(meta["nb_frames"])

        source_fps = meta.get("avg_fps") or meta.get("nominal_fps")
        if duration is not None and source_fps:
            return int(round(duration * source_fps))

        return None

    try:
        target_fps = float(fps_arg)
    except ValueError:
        return None

    if duration is None:
        return None

    return int(math.ceil(duration * target_fps))


def extract(
    video: str,
    output_dir: str,
    fps_arg: str,
    start_sec: Optional[float],
    end_sec: Optional[float],
    jpeg_quality: int,
    overwrite: bool,
    meta: Dict,
) -> Tuple[int, int, int, str]:
    """
    Decode the source exactly once.

    Returns:
        written, preserved, decoded_frames, timestamp_mode
    """
    os.makedirs(output_dir, exist_ok=True)

    existing_by_index = scan_existing_frames(output_dir)
    fallback_fps = choose_fallback_fps(meta)

    use_decoder_pts = decoder_timestamp_is_usable(video, fallback_fps)
    timestamp_mode = (
        "decoder presentation timestamps"
        if use_decoder_pts
        else "CFR fallback from ffprobe average frame rate"
    )

    if fps_arg.lower() == "full":
        target_fps = None
    else:
        try:
            target_fps = float(fps_arg)
        except ValueError:
            raise ValueError('--fps must be a positive number or "full"')

        if target_fps <= 0:
            raise ValueError('--fps must be a positive number or "full"')

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise RuntimeError("OpenCV could not open {}".format(video))

    first_decoder_pts = None
    last_timestamp = None

    # Numeric-FPS extraction is tied to a fixed timeline grid:
    # 0, 1/fps, 2/fps, ...
    #
    # We select the first source frame at/after each target time. This gives
    # ~8 fps from 29.97 fps with a mixture of 3- and 4-frame gaps.
    if target_fps is not None:
        grid_step = 1.0 / target_fps
        lower = 0.0 if start_sec is None else start_sec
        grid_index = int(math.ceil(lower / grid_step - 1e-12))
        next_target = grid_index * grid_step
    else:
        grid_step = None
        next_target = None

    written = 0
    preserved = 0
    decoded_frames = 0
    source_index = 0

    params = [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)]

    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break

            if use_decoder_pts:
                decoder_pts = float(cap.get(cv2.CAP_PROP_POS_MSEC)) / 1000.0

                if first_decoder_pts is None:
                    first_decoder_pts = decoder_pts

                timestamp_sec = decoder_pts - first_decoder_pts

                if (
                    not math.isfinite(timestamp_sec)
                    or timestamp_sec < -1e-6
                    or (
                        last_timestamp is not None
                        and timestamp_sec < last_timestamp - 1e-6
                    )
                ):
                    raise RuntimeError(
                        "Decoder timestamps became invalid/non-monotonic at "
                        "source frame {}. Aborting rather than create unstable "
                        "filenames.".format(source_index)
                    )
            else:
                timestamp_sec = source_index / fallback_fps

            if timestamp_sec < 0:
                timestamp_sec = 0.0

            last_timestamp = timestamp_sec
            decoded_frames += 1

            if start_sec is not None and timestamp_sec < start_sec - 1e-9:
                source_index += 1
                continue

            if end_sec is not None and timestamp_sec > end_sec + 1e-9:
                break

            should_write = False

            if target_fps is None:
                should_write = True
            else:
                if timestamp_sec + 1e-9 >= next_target:
                    should_write = True

                    # Advance the fixed sampling grid beyond this frame.
                    # If requested fps >= source fps, the frame is still
                    # written only once.
                    while next_target <= timestamp_sec + 1e-9:
                        next_target += grid_step

            if should_write:
                filename = make_filename(source_index, timestamp_sec)

                current = existing_by_index.get(source_index)
                if current is not None and current != filename:
                    raise RuntimeError(
                        "Source frame {} already exists as:\n"
                        "  {}\n"
                        "but this run expects:\n"
                        "  {}\n\n"
                        "The source video or timestamp method differs from the "
                        "previous extraction. Refusing to risk breaking existing "
                        "LabelMe annotations.".format(
                            source_index, current, filename
                        )
                    )

                output_path = os.path.join(output_dir, filename)

                if os.path.exists(output_path) and not overwrite:
                    preserved += 1
                else:
                    if not cv2.imwrite(output_path, image, params):
                        raise RuntimeError(
                            "Failed to write {}".format(output_path)
                        )
                    written += 1

            source_index += 1

    finally:
        cap.release()

    return written, preserved, decoded_frames, timestamp_mode


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Timeline-preserving SmarterLabelMe frame extractor. "
            "Existing annotated frames can later be expanded to denser/full "
            "source FPS without renaming them."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("video", help="Input video")
    parser.add_argument("output_dir", help="Directory for extracted JPEG frames")

    parser.add_argument(
        "--fps",
        default="8",
        help='Extraction rate such as "8", "10", "29.97", or "full"',
    )
    parser.add_argument(
        "--start-sec",
        type=float,
        default=None,
        help="Optional source-relative start time in seconds",
    )
    parser.add_argument(
        "--end-sec",
        type=float,
        default=None,
        help="Optional source-relative end time in seconds",
    )
    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        choices=range(1, 101),
        metavar="[1-100]",
        help="JPEG quality",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help=(
            "Overwrite existing canonical JPEGs. Normally leave this OFF "
            "once annotations exist."
        ),
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Fast metadata-only preview. It does NOT scan/decode the full video."
        ),
    )

    args = parser.parse_args()

    video = os.path.abspath(os.path.expanduser(args.video))
    output_dir = os.path.abspath(os.path.expanduser(args.output_dir))

    if not os.path.isfile(video):
        print("ERROR: input video does not exist: {}".format(video), file=sys.stderr)
        sys.exit(2)

    if args.start_sec is not None and args.start_sec < 0:
        print("ERROR: --start-sec must be >= 0", file=sys.stderr)
        sys.exit(2)

    if args.end_sec is not None and args.end_sec < 0:
        print("ERROR: --end-sec must be >= 0", file=sys.stderr)
        sys.exit(2)

    if (
        args.start_sec is not None
        and args.end_sec is not None
        and args.end_sec < args.start_sec
    ):
        print("ERROR: --end-sec must be >= --start-sec", file=sys.stderr)
        sys.exit(2)

    if args.fps.lower() != "full":
        try:
            requested_fps = float(args.fps)
        except ValueError:
            print(
                'ERROR: --fps must be a positive number or "full"',
                file=sys.stderr,
            )
            sys.exit(2)

        if requested_fps <= 0:
            print(
                'ERROR: --fps must be a positive number or "full"',
                file=sys.stderr,
            )
            sys.exit(2)

    try:
        ffprobe = require_program("ffprobe")
        meta = probe_video(video, ffprobe)
    except RuntimeError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)

    avg_fps = meta.get("avg_fps")
    nominal_fps = meta.get("nominal_fps")
    duration = meta.get("duration")

    print("Source video : {}".format(video))
    print(
        "Resolution   : {} x {}".format(
            meta.get("width", "?"),
            meta.get("height", "?"),
        )
    )
    print(
        "Source FPS   : avg={} nominal={}".format(
            "{:.6f}".format(avg_fps) if avg_fps else "unknown",
            "{:.6f}".format(nominal_fps) if nominal_fps else "unknown",
        )
    )
    print(
        "Duration     : {}".format(
            "{:.3f} s".format(duration) if duration is not None else "unknown"
        )
    )
    print(
        "Source frames: {}".format(
            meta.get("nb_frames")
            if meta.get("nb_frames") is not None
            else "not stored in container"
        )
    )
    print("Extract mode : {} fps".format(args.fps))
    print("Output dir   : {}".format(output_dir))
    print("Filename     : fXXXXXXXX_tXXXXXXXXX.jpg")

    estimate = estimated_output_count(meta, args.fps)
    if estimate is not None:
        print("Est. output  : ~{} frames (whole video)".format(estimate))

    if args.dry_run:
        print("Dry run      : metadata only; no full decode performed.")
        return

    try:
        written, preserved, decoded, timestamp_mode = extract(
            video=video,
            output_dir=output_dir,
            fps_arg=args.fps,
            start_sec=args.start_sec,
            end_sec=args.end_sec,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
            meta=meta,
        )
    except (RuntimeError, ValueError) as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        sys.exit(1)

    print("Timestamp    : {}".format(timestamp_mode))
    print("Decoded      : {} source frames".format(decoded))
    print("Written      : {} new images".format(written))
    print("Preserved    : {} existing images".format(preserved))
    print("Done.")


if __name__ == "__main__":
    main()

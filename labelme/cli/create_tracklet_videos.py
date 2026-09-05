#!/usr/bin/env python3
"""
Create timestamp-driven behavior-review tracklet videos from SmarterLabelMe.

Canonical generator
===================

This generator supports both dense and sparse manual/tracker annotations.

Example:
    Original video ~29.97 fps
        -> extracted images at ~8 fps
        -> SmarterLabelMe tracking performed every 4th extracted image (~2 Hz)

The generator DOES NOT repeat the sparse annotated crop.

Instead it:
    1. Reads ALL real extracted images between trusted annotations.
    2. Linearly interpolates the bounding box between the two trusted boxes.
    3. Crops the REAL intermediate images.
    4. Produces a smooth review video at the extracted-image cadence.

Large gaps for the same identity are automatically split:
    ph_vid03_007_seg01.mp4
    ph_vid03_007_seg02.mp4

No interpolation is performed across such gaps.

The JSON manifest records for every rendered frame whether its bbox was:
    - "annotated"    : directly from SmarterLabelMe
    - "interpolated" : linearly interpolated between trusted annotations

Source timestamps encoded in canonical filenames are authoritative:
    f00001234_t000041175.jpg

Python 3.8 compatible.
"""

import argparse
import bisect
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np


TIMELINE_RE = re.compile(
    r"^f(?P<frame>\d+)_t(?P<ms>\d+)\.(?:jpg|jpeg|png)$",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Timeline helpers
# ---------------------------------------------------------------------------

def parse_timeline_name(filename: str) -> Tuple[Optional[int], Optional[int]]:
    match = TIMELINE_RE.match(os.path.basename(filename))
    if match is None:
        return None, None

    return int(match.group("frame")), int(match.group("ms"))


def scan_timeline_images(frames_dir: str) -> List[Dict]:
    """
    Scan canonical timeline-named images in frames_dir.

    Returns entries sorted by source timestamp, then source frame.
    """
    entries = []

    for name in os.listdir(frames_dir):
        path = os.path.join(frames_dir, name)

        if not os.path.isfile(path):
            continue

        source_frame, source_ms = parse_timeline_name(name)

        if source_frame is None or source_ms is None:
            continue

        entries.append(
            {
                "image_path": path,
                "image_name": name,
                "source_frame": source_frame,
                "source_ms": source_ms,
            }
        )

    entries.sort(
        key=lambda e: (
            e["source_ms"],
            e["source_frame"],
            e["image_name"],
        )
    )

    return entries


def estimate_timeline_fps(timeline: Sequence[Dict]) -> Optional[float]:
    """
    Estimate effective extracted-image cadence from timestamps.

    Uses a trimmed mean of consecutive timestamp differences, rather than the
    median alone, because 29.97 -> 8 fps extraction naturally alternates
    roughly 100 ms / 133 ms gaps.
    """
    if len(timeline) < 2:
        return None

    deltas_ms = []

    for a, b in zip(timeline[:-1], timeline[1:]):
        dt = b["source_ms"] - a["source_ms"]
        if dt > 0:
            deltas_ms.append(float(dt))

    if not deltas_ms:
        return None

    med = statistics.median(deltas_ms)

    # Remove true holes while preserving ordinary sampling jitter.
    filtered = [
        dt for dt in deltas_ms
        if dt <= max(1000.0, med * 4.0)
    ]

    if not filtered:
        filtered = deltas_ms

    mean_dt = statistics.mean(filtered)

    if mean_dt <= 0:
        return None

    return 1000.0 / mean_dt


# ---------------------------------------------------------------------------
# Annotation loading
# ---------------------------------------------------------------------------

def shape_bbox(shape: Dict) -> Optional[Tuple[float, float, float, float]]:
    if shape.get("shape_type") == "point":
        return None

    points = shape.get("points") or []

    if len(points) < 2:
        return None

    try:
        xs = [float(p[0]) for p in points]
        ys = [float(p[1]) for p in points]
    except (TypeError, ValueError, IndexError):
        return None

    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def resolve_image_path(
    frames_dir: str,
    annotations_dir: str,
    json_name: str,
    data: Dict,
) -> Optional[str]:
    image_path_field = data.get("imagePath")

    if image_path_field:
        # Try path relative to annotation directory.
        candidate = os.path.normpath(
            os.path.join(annotations_dir, image_path_field)
        )
        if os.path.isfile(candidate):
            return candidate

        # Standard SmarterLabelMe layout: images one level outside Annotations.
        candidate = os.path.join(
            frames_dir,
            os.path.basename(image_path_field),
        )
        if os.path.isfile(candidate):
            return candidate

    # Fall back to JSON basename.
    stem = os.path.splitext(json_name)[0]

    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = os.path.join(frames_dir, stem + ext)
        if os.path.isfile(candidate):
            return candidate

    return None


def load_track_entries(
    frames_dir: str,
    annotations_dir: str,
    label_prefix: Optional[str],
) -> Dict[str, List[Dict]]:
    """
    Scan LabelMe JSON files and group trusted annotations by identity label.
    """
    if not os.path.isdir(annotations_dir):
        raise RuntimeError(
            "Annotations directory does not exist: {}".format(
                annotations_dir
            )
        )

    tracks = defaultdict(list)

    json_names = sorted(
        name for name in os.listdir(annotations_dir)
        if name.lower().endswith(".json")
    )

    for json_name in json_names:
        json_path = os.path.join(annotations_dir, json_name)

        try:
            with open(json_path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception as exc:
            print(
                "WARNING: could not read {}: {}".format(json_path, exc),
                file=sys.stderr,
            )
            continue

        image_path = resolve_image_path(
            frames_dir,
            annotations_dir,
            json_name,
            data,
        )

        if not image_path:
            print(
                "WARNING: source image for {} not found".format(json_path),
                file=sys.stderr,
            )
            continue

        image_name = os.path.basename(image_path)
        source_frame, source_ms = parse_timeline_name(image_name)

        if source_frame is None or source_ms is None:
            print(
                "WARNING: skipping non-canonical timeline image {}".format(
                    image_name
                ),
                file=sys.stderr,
            )
            continue

        for shape in data.get("shapes", []):
            label = str(shape.get("label", ""))

            if not label:
                continue

            if label_prefix and not label.startswith(label_prefix):
                continue

            if "_kpt_" in label or shape.get("shape_type") == "point":
                continue

            bbox = shape_bbox(shape)

            if bbox is None:
                continue

            viewpoint = shape.get("viewpoint")
            if viewpoint is None:
                viewpoint = (shape.get("other_data") or {}).get("viewpoint")

            tracks[label].append(
                {
                    "image_path": image_path,
                    "image_name": image_name,
                    "json_path": json_path,
                    "bbox": bbox,
                    "source_frame": source_frame,
                    "source_ms": source_ms,
                    "viewpoint": viewpoint,
                }
            )

    cleaned = {}

    for label, entries in tracks.items():
        entries.sort(
            key=lambda e: (
                e["source_ms"],
                e["source_frame"],
                e["image_name"],
            )
        )

        deduped = []
        seen = set()

        for entry in entries:
            key = (entry["source_ms"], entry["source_frame"])

            if key in seen:
                print(
                    "WARNING: duplicate annotation for {} at {}".format(
                        label,
                        entry["image_name"],
                    ),
                    file=sys.stderr,
                )
                continue

            seen.add(key)
            deduped.append(entry)

        cleaned[label] = deduped

    return cleaned


# ---------------------------------------------------------------------------
# Gap splitting
# ---------------------------------------------------------------------------

def positive_annotation_deltas_sec(
    entries: Sequence[Dict],
) -> List[float]:
    deltas = []

    for a, b in zip(entries[:-1], entries[1:]):
        dt = (b["source_ms"] - a["source_ms"]) / 1000.0
        if dt > 0:
            deltas.append(dt)

    return deltas


def robust_median_annotation_delta_sec(
    entries: Sequence[Dict],
) -> Optional[float]:
    deltas = positive_annotation_deltas_sec(entries)

    if not deltas:
        return None

    med = statistics.median(deltas)

    if med <= 0:
        return None

    filtered = [d for d in deltas if d <= med * 6.0]

    if filtered:
        med = statistics.median(filtered)

    return float(med)


def automatic_gap_threshold_sec(entries: Sequence[Dict]) -> float:
    """
    Conservative discontinuity threshold.

    Examples:
        annotations ~8 Hz -> minimum threshold 3 s
        annotations ~2 Hz -> minimum threshold 3 s
        annotations ~1 Hz -> threshold ~6 s

    A deliberate one-minute jump is therefore clearly split.
    """
    med = robust_median_annotation_delta_sec(entries)

    if med is None:
        return 3.0

    return max(3.0, 6.0 * med)


def split_track(
    entries: Sequence[Dict],
    threshold_sec: Optional[float],
) -> List[List[Dict]]:
    if not entries:
        return []

    if threshold_sec is None:
        return [list(entries)]

    segments = [[entries[0]]]

    for prev, cur in zip(entries[:-1], entries[1:]):
        gap_sec = (cur["source_ms"] - prev["source_ms"]) / 1000.0

        if gap_sec > threshold_sec:
            segments.append([cur])
        else:
            segments[-1].append(cur)

    return segments


# ---------------------------------------------------------------------------
# Interpolation
# ---------------------------------------------------------------------------

def lerp(a: float, b: float, alpha: float) -> float:
    return a + alpha * (b - a)


def interpolate_bbox(
    bbox_a: Sequence[float],
    bbox_b: Sequence[float],
    alpha: float,
) -> Tuple[float, float, float, float]:
    return tuple(
        lerp(float(a), float(b), alpha)
        for a, b in zip(bbox_a, bbox_b)
    )


def interpolate_angle_deg(
    angle_a: Optional[float],
    angle_b: Optional[float],
    alpha: float,
) -> Optional[float]:
    """
    Circular interpolation using shortest angular path.
    """
    if angle_a is None and angle_b is None:
        return None

    if angle_a is None:
        try:
            return float(angle_b)
        except (TypeError, ValueError):
            return None

    if angle_b is None:
        try:
            return float(angle_a)
        except (TypeError, ValueError):
            return None

    try:
        a = float(angle_a) % 360.0
        b = float(angle_b) % 360.0
    except (TypeError, ValueError):
        return None

    delta = ((b - a + 180.0) % 360.0) - 180.0
    return (a + alpha * delta) % 360.0


def interpolate_annotation_at_time(
    annotations: Sequence[Dict],
    annotation_times: Sequence[int],
    source_ms: int,
) -> Optional[Dict]:
    """
    Return interpolated bbox/viewpoint metadata for one real source image.

    Requires source_ms within [first annotation, last annotation].
    """
    if not annotations:
        return None

    pos = bisect.bisect_left(annotation_times, source_ms)

    # Exact trusted annotation.
    if pos < len(annotation_times) and annotation_times[pos] == source_ms:
        ann = annotations[pos]

        return {
            "bbox": tuple(ann["bbox"]),
            "viewpoint": ann["viewpoint"],
            "bbox_origin": "annotated",
            "left_annotation_index": pos,
            "right_annotation_index": pos,
            "left_annotation_image": ann["image_name"],
            "right_annotation_image": ann["image_name"],
            "interpolation_alpha": 0.0,
        }

    # Outside trusted temporal support: do not extrapolate.
    if pos == 0 or pos >= len(annotations):
        return None

    left_idx = pos - 1
    right_idx = pos

    left = annotations[left_idx]
    right = annotations[right_idx]

    t0 = left["source_ms"]
    t1 = right["source_ms"]

    if t1 <= t0:
        return None

    alpha = float(source_ms - t0) / float(t1 - t0)
    alpha = max(0.0, min(1.0, alpha))

    return {
        "bbox": interpolate_bbox(
            left["bbox"],
            right["bbox"],
            alpha,
        ),
        "viewpoint": interpolate_angle_deg(
            left["viewpoint"],
            right["viewpoint"],
            alpha,
        ),
        "bbox_origin": "interpolated",
        "left_annotation_index": left_idx,
        "right_annotation_index": right_idx,
        "left_annotation_image": left["image_name"],
        "right_annotation_image": right["image_name"],
        "interpolation_alpha": alpha,
    }


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------

def padded_crop(
    image: np.ndarray,
    bbox: Sequence[float],
    padding: float,
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    h, w = image.shape[:2]

    x1, y1, x2, y2 = [float(v) for v in bbox]

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 0 or bh <= 0:
        raise ValueError("Degenerate bbox")

    px = padding * bw
    py = padding * bh

    cx1 = max(0, int(math.floor(x1 - px)))
    cy1 = max(0, int(math.floor(y1 - py)))
    cx2 = min(w, int(math.ceil(x2 + px)))
    cy2 = min(h, int(math.ceil(y2 + py)))

    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError("Degenerate crop after clamping")

    return image[cy1:cy2, cx1:cx2].copy(), (cx1, cy1, cx2, cy2)


def letterbox(
    image: np.ndarray,
    out_w: int,
    out_h: int,
) -> np.ndarray:
    h, w = image.shape[:2]

    if w <= 0 or h <= 0:
        raise ValueError("Empty crop")

    scale = min(float(out_w) / float(w), float(out_h) / float(h))

    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
    )

    canvas = np.zeros((out_h, out_w, 3), dtype=np.uint8)

    ox = (out_w - new_w) // 2
    oy = (out_h - new_h) // 2

    canvas[oy:oy + new_h, ox:ox + new_w] = resized

    return canvas


def draw_text_overlay(
    frame: np.ndarray,
    label: str,
    segment_number: int,
    video_time_sec: float,
    source_frame: int,
    source_ms: int,
    bbox_origin: str,
    viewpoint: Optional[float],
) -> None:
    lines = [
        label,
        "segment {:02d}".format(segment_number),
        "video {:8.3f}s".format(video_time_sec),
        "source {:8.3f}s".format(source_ms / 1000.0),
        "src frame {}".format(source_frame),
        "bbox {}".format(bbox_origin),
    ]

    if viewpoint is not None:
        try:
            lines.append(
                "viewpoint {:5.1f} deg".format(float(viewpoint))
            )
        except (TypeError, ValueError):
            pass

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.50
    thickness = 1
    line_h = 20
    x = 10
    y0 = 22

    max_w = 0

    for line in lines:
        (tw, _th), _base = cv2.getTextSize(
            line,
            font,
            font_scale,
            thickness,
        )
        max_w = max(max_w, tw)

    overlay = frame.copy()

    cv2.rectangle(
        overlay,
        (4, 4),
        (
            min(frame.shape[1] - 1, max_w + 18),
            min(
                frame.shape[0] - 1,
                y0 + line_h * (len(lines) - 1) + 8,
            ),
        ),
        (0, 0, 0),
        -1,
    )

    cv2.addWeighted(
        overlay,
        0.60,
        frame,
        0.40,
        0,
        dst=frame,
    )

    for i, line in enumerate(lines):
        cv2.putText(
            frame,
            line,
            (x, y0 + i * line_h),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )


def draw_viewpoint_arrow(
    frame: np.ndarray,
    viewpoint: Optional[float],
    length_px: int = 55,
) -> None:
    """
    Verified convention:
        0 deg   = up
        90 deg  = right
        180 deg = down
        270 deg = left
    """
    if viewpoint is None:
        return

    try:
        angle = math.radians(float(viewpoint))
    except (TypeError, ValueError):
        return

    h, w = frame.shape[:2]
    center = (w - 70, 70)

    dx = math.sin(angle)
    dy = -math.cos(angle)

    tip = (
        int(round(center[0] + length_px * dx)),
        int(round(center[1] + length_px * dy)),
    )

    cv2.arrowedLine(
        frame,
        center,
        tip,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
        tipLength=0.28,
    )


def choose_fourcc():
    return cv2.VideoWriter_fourcc(*"mp4v")


def timeline_slice_for_segment(
    timeline: Sequence[Dict],
    segment: Sequence[Dict],
) -> List[Dict]:
    """
    Return all real extracted images from first trusted annotation timestamp
    through last trusted annotation timestamp, inclusive.
    """
    if not timeline or not segment:
        return []

    start_ms = segment[0]["source_ms"]
    end_ms = segment[-1]["source_ms"]

    times = [entry["source_ms"] for entry in timeline]

    left = bisect.bisect_left(times, start_ms)
    right = bisect.bisect_right(times, end_ms)

    return list(timeline[left:right])


def create_segment_video(
    label: str,
    segment_number: int,
    annotations: Sequence[Dict],
    timeline_frames: Sequence[Dict],
    video_path: str,
    manifest_path: str,
    render_fps: float,
    padding: float,
    out_w: int,
    out_h: int,
    overlay: bool,
    viewpoint_arrow: bool,
) -> Dict:
    if not annotations:
        raise RuntimeError("Empty annotation segment")

    if not timeline_frames:
        raise RuntimeError("No extracted frames found for segment")

    annotation_times = [ann["source_ms"] for ann in annotations]

    writer = cv2.VideoWriter(
        video_path,
        choose_fourcc(),
        render_fps,
        (out_w, out_h),
    )

    if not writer.isOpened():
        raise RuntimeError(
            "Could not open video writer for {}".format(video_path)
        )

    manifest_frames = []
    written = 0
    annotated_count = 0
    interpolated_count = 0

    first_source_ms = timeline_frames[0]["source_ms"]

    try:
        for source_entry in timeline_frames:
            interp = interpolate_annotation_at_time(
                annotations,
                annotation_times,
                source_entry["source_ms"],
            )

            if interp is None:
                continue

            image = cv2.imread(
                source_entry["image_path"],
                cv2.IMREAD_COLOR,
            )

            if image is None:
                print(
                    "WARNING: failed to read {}".format(
                        source_entry["image_path"]
                    ),
                    file=sys.stderr,
                )
                continue

            try:
                crop, crop_box = padded_crop(
                    image,
                    interp["bbox"],
                    padding,
                )

                frame = letterbox(
                    crop,
                    out_w,
                    out_h,
                )

            except Exception as exc:
                print(
                    "WARNING: {} seg {:02d} skipping {}: {}".format(
                        label,
                        segment_number,
                        source_entry["image_name"],
                        exc,
                    ),
                    file=sys.stderr,
                )
                continue

            if interp["bbox_origin"] == "annotated":
                annotated_count += 1
            else:
                interpolated_count += 1

            # MP4 is CFR. This is playback time.
            video_time_sec = written / render_fps

            # This is the real source-relative time within the segment.
            source_relative_sec = (
                source_entry["source_ms"] - first_source_ms
            ) / 1000.0

            if overlay:
                draw_text_overlay(
                    frame=frame,
                    label=label,
                    segment_number=segment_number,
                    video_time_sec=video_time_sec,
                    source_frame=source_entry["source_frame"],
                    source_ms=source_entry["source_ms"],
                    bbox_origin=interp["bbox_origin"],
                    viewpoint=interp["viewpoint"],
                )

            if viewpoint_arrow:
                draw_viewpoint_arrow(
                    frame,
                    interp["viewpoint"],
                )

            writer.write(frame)

            manifest_frames.append(
                {
                    "video_frame": written,
                    "video_time_sec": round(video_time_sec, 6),
                    "source_relative_sec": round(
                        source_relative_sec,
                        6,
                    ),
                    "source_image": source_entry["image_name"],
                    "source_frame": source_entry["source_frame"],
                    "source_timestamp_ms": source_entry["source_ms"],
                    "source_timestamp_sec": round(
                        source_entry["source_ms"] / 1000.0,
                        6,
                    ),
                    "bbox_xyxy": [
                        round(float(v), 3)
                        for v in interp["bbox"]
                    ],
                    "bbox_origin": interp["bbox_origin"],
                    "crop_xyxy": list(crop_box),
                    "viewpoint": (
                        round(float(interp["viewpoint"]), 6)
                        if interp["viewpoint"] is not None
                        else None
                    ),
                    "left_annotation_index": interp[
                        "left_annotation_index"
                    ],
                    "right_annotation_index": interp[
                        "right_annotation_index"
                    ],
                    "left_annotation_image": interp[
                        "left_annotation_image"
                    ],
                    "right_annotation_image": interp[
                        "right_annotation_image"
                    ],
                    "interpolation_alpha": round(
                        float(interp["interpolation_alpha"]),
                        6,
                    ),
                }
            )

            written += 1

    finally:
        writer.release()

    source_start_ms = timeline_frames[0]["source_ms"]
    source_end_ms = timeline_frames[-1]["source_ms"]
    source_span_sec = (
        source_end_ms - source_start_ms
    ) / 1000.0

    manifest = {
        "generator_mode": "real_frames_with_interpolated_bbox",
        "label": label,
        "segment_number": segment_number,
        "render_fps": render_fps,
        "output_size": [out_w, out_h],
        "padding_fraction": padding,
        "video_file": os.path.basename(video_path),
        "trusted_annotation_count": len(annotations),
        "real_source_frame_count": written,
        "direct_annotated_frame_count": annotated_count,
        "interpolated_bbox_frame_count": interpolated_count,
        "rendered_duration_sec": round(
            written / render_fps,
            6,
        ),
        "source_start_timestamp_ms": source_start_ms,
        "source_end_timestamp_ms": source_end_ms,
        "source_span_sec": round(source_span_sec, 6),
        "frames": manifest_frames,
    }

    with open(
        manifest_path,
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            manifest,
            handle,
            indent=2,
        )

    return manifest


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_split_gap_arg(
    value: str,
) -> Union[str, float, None]:
    value = str(value).strip().lower()

    if value == "auto":
        return "auto"

    if value in (
        "none",
        "off",
        "disable",
        "disabled",
    ):
        return None

    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--split-gap-sec must be 'auto', 'none', or a positive number"
        )

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "--split-gap-sec numeric value must be > 0"
        )

    return parsed


def parse_render_fps_arg(
    value: str,
) -> Union[str, float]:
    value = str(value).strip().lower()

    if value == "auto":
        return "auto"

    try:
        parsed = float(value)
    except ValueError:
        raise argparse.ArgumentTypeError(
            "--render-fps must be 'auto' or a positive number"
        )

    if parsed <= 0:
        raise argparse.ArgumentTypeError(
            "--render-fps numeric value must be > 0"
        )

    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create smooth behavior-review videos using REAL extracted "
            "frames and interpolated bboxes between sparse SmarterLabelMe "
            "tracking annotations."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "frames_dir",
        help="Directory containing canonical timeline images",
    )

    parser.add_argument(
        "--annotations-dir",
        default=None,
        help="Defaults to <frames_dir>/Annotations",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Defaults to <frames_dir>/Tracklets",
    )

    parser.add_argument(
        "--label-prefix",
        default=None,
        help='Optional identity prefix, e.g. "ph_vid03_"',
    )

    parser.add_argument(
        "--render-fps",
        type=parse_render_fps_arg,
        default="auto",
        help=(
            "Output MP4 FPS. 'auto' infers extracted-image cadence from "
            "source timestamps."
        ),
    )

    parser.add_argument(
        "--split-gap-sec",
        type=parse_split_gap_arg,
        default="auto",
        help=(
            "Split same ID across large source-time gaps. Use 'auto', "
            "'none', or seconds such as 5."
        ),
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.25,
        help="Fractional padding around interpolated bbox",
    )

    parser.add_argument(
        "--size",
        type=int,
        default=640,
        help="Square output-video size in pixels",
    )

    parser.add_argument(
        "--no-overlay",
        action="store_true",
        help="Do not draw metadata text",
    )

    parser.add_argument(
        "--viewpoint-arrow",
        action="store_true",
        help="Draw viewpoint arrow; viewpoint is circularly interpolated",
    )

    parser.add_argument(
        "--list-only",
        action="store_true",
        help="Inspect identities/segments without creating videos",
    )

    args = parser.parse_args()

    if args.padding < 0:
        print(
            "ERROR: --padding must be >= 0",
            file=sys.stderr,
        )
        sys.exit(2)

    if args.size < 64:
        print(
            "ERROR: --size must be >= 64",
            file=sys.stderr,
        )
        sys.exit(2)

    frames_dir = os.path.abspath(
        os.path.expanduser(args.frames_dir)
    )

    annotations_dir = (
        os.path.abspath(
            os.path.expanduser(args.annotations_dir)
        )
        if args.annotations_dir
        else os.path.join(frames_dir, "Annotations")
    )

    output_dir = (
        os.path.abspath(
            os.path.expanduser(args.output_dir)
        )
        if args.output_dir
        else os.path.join(frames_dir, "Tracklets")
    )

    if not os.path.isdir(frames_dir):
        print(
            "ERROR: frames directory does not exist: {}".format(
                frames_dir
            ),
            file=sys.stderr,
        )
        sys.exit(2)

    timeline = scan_timeline_images(frames_dir)

    if not timeline:
        print(
            "ERROR: no canonical timeline images found in {}".format(
                frames_dir
            ),
            file=sys.stderr,
        )
        sys.exit(1)

    inferred_fps = estimate_timeline_fps(timeline)

    if args.render_fps == "auto":
        if inferred_fps is None:
            print(
                "ERROR: could not infer extracted-image FPS",
                file=sys.stderr,
            )
            sys.exit(1)

        render_fps = inferred_fps
    else:
        render_fps = float(args.render_fps)

    try:
        tracks = load_track_entries(
            frames_dir,
            annotations_dir,
            args.label_prefix,
        )
    except RuntimeError as exc:
        print(
            "ERROR: {}".format(exc),
            file=sys.stderr,
        )
        sys.exit(1)

    if not tracks:
        print("No matching tracks found.")
        return

    print("Timeline images : {}".format(len(timeline)))
    print(
        "Timeline span   : {:.3f}s -> {:.3f}s".format(
            timeline[0]["source_ms"] / 1000.0,
            timeline[-1]["source_ms"] / 1000.0,
        )
    )

    if inferred_fps is not None:
        print(
            "Inferred rate  : {:.6f} fps".format(
                inferred_fps
            )
        )

    print(
        "Render rate    : {:.6f} fps".format(
            render_fps
        )
    )

    all_segments = {}

    print("")
    print(
        "Found {} identity track(s):".format(
            len(tracks)
        )
    )

    for label in sorted(tracks):
        entries = tracks[label]

        if args.split_gap_sec == "auto":
            threshold = automatic_gap_threshold_sec(entries)
        else:
            threshold = args.split_gap_sec

        segments = split_track(
            entries,
            threshold,
        )

        all_segments[label] = (
            segments,
            threshold,
        )

        med = robust_median_annotation_delta_sec(entries)

        med_txt = (
            "{:.3f}s (~{:.2f} Hz)".format(
                med,
                1.0 / med,
            )
            if med is not None and med > 0
            else "unknown"
        )

        threshold_txt = (
            "{:.3f}s".format(threshold)
            if threshold is not None
            else "disabled"
        )

        print("")
        print("  {}".format(label))
        print(
            "    trusted annotations : {}".format(
                len(entries)
            )
        )
        print(
            "    annotation cadence  : {}".format(
                med_txt
            )
        )
        print(
            "    split gap threshold : {}".format(
                threshold_txt
            )
        )
        print(
            "    segments            : {}".format(
                len(segments)
            )
        )

        for i, segment in enumerate(
            segments,
            1,
        ):
            real_frames = timeline_slice_for_segment(
                timeline,
                segment,
            )

            first_ms = segment[0]["source_ms"]
            last_ms = segment[-1]["source_ms"]
            span = (last_ms - first_ms) / 1000.0

            print(
                "      seg {:02d}: {:5d} annotations -> {:5d} real frames  "
                "{:.3f}s -> {:.3f}s ({:.3f}s span)".format(
                    i,
                    len(segment),
                    len(real_frames),
                    first_ms / 1000.0,
                    last_ms / 1000.0,
                    span,
                )
            )

    if args.list_only:
        return

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    print("")
    print(
        "Creating tracklets in: {}".format(
            output_dir
        )
    )

    for label in sorted(all_segments):
        segments, threshold = all_segments[label]
        multi_segment = len(segments) > 1
        identity_index = []

        for segment_number, annotations in enumerate(
            segments,
            1,
        ):
            timeline_frames = timeline_slice_for_segment(
                timeline,
                annotations,
            )

            if multi_segment:
                stem = "{}_seg{:02d}".format(
                    label,
                    segment_number,
                )
            else:
                stem = label

            video_path = os.path.join(
                output_dir,
                stem + ".mp4",
            )

            manifest_path = os.path.join(
                output_dir,
                stem + ".json",
            )

            try:
                manifest = create_segment_video(
                    label=label,
                    segment_number=segment_number,
                    annotations=annotations,
                    timeline_frames=timeline_frames,
                    video_path=video_path,
                    manifest_path=manifest_path,
                    render_fps=render_fps,
                    padding=args.padding,
                    out_w=args.size,
                    out_h=args.size,
                    overlay=not args.no_overlay,
                    viewpoint_arrow=args.viewpoint_arrow,
                )

            except Exception as exc:
                print(
                    "ERROR creating {}: {}".format(
                        stem,
                        exc,
                    ),
                    file=sys.stderr,
                )
                continue

            identity_index.append(
                {
                    "segment_number": segment_number,
                    "video_file": os.path.basename(
                        video_path
                    ),
                    "manifest_file": os.path.basename(
                        manifest_path
                    ),
                    "trusted_annotation_count": manifest[
                        "trusted_annotation_count"
                    ],
                    "real_source_frame_count": manifest[
                        "real_source_frame_count"
                    ],
                    "direct_annotated_frame_count": manifest[
                        "direct_annotated_frame_count"
                    ],
                    "interpolated_bbox_frame_count": manifest[
                        "interpolated_bbox_frame_count"
                    ],
                    "source_start_timestamp_ms": manifest[
                        "source_start_timestamp_ms"
                    ],
                    "source_end_timestamp_ms": manifest[
                        "source_end_timestamp_ms"
                    ],
                }
            )

            print(
                "  {}: {} trusted boxes -> {} real frames "
                "({} direct + {} interpolated)".format(
                    os.path.basename(video_path),
                    manifest["trusted_annotation_count"],
                    manifest["real_source_frame_count"],
                    manifest["direct_annotated_frame_count"],
                    manifest["interpolated_bbox_frame_count"],
                )
            )

        if multi_segment:
            index_path = os.path.join(
                output_dir,
                "{}_index.json".format(label),
            )

            with open(
                index_path,
                "w",
                encoding="utf-8",
            ) as handle:
                json.dump(
                    {
                                        "label": label,
                        "gap_split_threshold_sec": threshold,
                        "render_fps": render_fps,
                        "segments": identity_index,
                    },
                    handle,
                    indent=2,
                )

    print("")
    print("Done.")


if __name__ == "__main__":
    main()

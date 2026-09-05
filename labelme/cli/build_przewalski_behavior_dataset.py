#!/usr/bin/env python3
"""
Build Przewalski's-horse behavior training datasets from:

    - verified behavior interval CSV
    - tracklet JSON manifests
    - original extracted source images

It creates two datasets from the SAME labels:

B_frame_classifier/
    train/grazing/*.jpg
    train/standing/*.jpg
    train/walking/*.jpg
    val/...
    test/...

C_temporal/
    train/grazing/<window_id>/000.jpg ... 015.jpg
    train/standing/...
    train/walking/...
    val/...
    test/...

plus:
    metadata/frames.csv
    metadata/windows.csv
    metadata/splits.csv
    metadata/build_config.json
    metadata/summary.json
    crops/...

Important design choices
------------------------
* Behavior labels come ONLY from the verified interval CSV.
* Bboxes/time mapping come from the tracklet manifests.
* Original source JPEGs are cropped; rendered/overlay tracklet MP4 frames
  are NOT used as training images.
* Dense old tracklets and sparse/interpolated newer tracklets both work.
* Temporal windows never cross behavior boundaries or segment gaps.
* Train/val/test are assigned by source video and/or horse ID.
* A small margin can be removed around behavior transitions.
* "running", "other", and "uncertain" are absent from the supplied 3-class
  interval CSV and therefore never enter this dataset.

Python 3.8 compatible.
"""

import argparse
import csv
import json
import math
import os
import shutil
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np


CLASSES = ("grazing", "standing", "walking")
CONFIDENCE_RANK = {
    "low": 0,
    "medium": 1,
    "high": 2,
}


def comma_list(value):
    if value is None:
        return []
    return [
        x.strip()
        for x in str(value).split(",")
        if x.strip()
    ]


def safe_bool(value):
    return str(value).strip().lower() in (
        "1", "true", "yes", "y", "on"
    )


def ensure_empty_output(path, overwrite):
    path = Path(path)

    if path.exists():
        if not overwrite:
            raise RuntimeError(
                "Output already exists: {}\n"
                "Use --overwrite if you want to rebuild it.".format(path)
            )
        shutil.rmtree(str(path))

    path.mkdir(parents=True, exist_ok=True)


def read_intervals(csv_path, min_confidence):
    rows = []
    minimum = CONFIDENCE_RANK[min_confidence]

    with open(csv_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)

        required = {
            "source_video",
            "horse_id",
            "tracklet",
            "start_sec",
            "end_sec",
            "behavior",
            "confidence",
        }

        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise RuntimeError(
                "Interval CSV missing columns: {}".format(
                    ", ".join(sorted(missing))
                )
            )

        for row in reader:
            behavior = row["behavior"].strip().lower()
            confidence = row["confidence"].strip().lower()

            if behavior not in CLASSES:
                continue

            if confidence not in CONFIDENCE_RANK:
                raise RuntimeError(
                    "Unknown confidence '{}' in interval CSV".format(
                        confidence
                    )
                )

            if CONFIDENCE_RANK[confidence] < minimum:
                continue

            if (
                "use_for_3class" in row
                and row["use_for_3class"]
                and not safe_bool(row["use_for_3class"])
            ):
                continue

            start = float(row["start_sec"])
            end = float(row["end_sec"])

            if end <= start:
                continue

            rows.append({
                "source_video": row["source_video"].strip(),
                "horse_id": row["horse_id"].strip(),
                "tracklet": row["tracklet"].strip(),
                "start_sec": start,
                "end_sec": end,
                "behavior": behavior,
                "confidence": confidence,
                "notes": row.get("notes", ""),
            })

    if not rows:
        raise RuntimeError("No usable 3-class intervals found.")

    return rows


def split_for_row(
    source_video,
    horse_id,
    test_videos,
    val_videos,
    val_horses,
):
    if source_video in test_videos:
        return "test"

    if source_video in val_videos or horse_id in val_horses:
        return "val"

    return "train"


def discover_video_dirs(root):
    """
    Find per-video directories.

    Preferred signal:
        a directory containing Tracklets/

    Fallback:
        a directory whose name contains _vidXX / vidXX
    """
    root = Path(root)
    result = defaultdict(list)

    for tracklets_dir in root.rglob("Tracklets"):
        if not tracklets_dir.is_dir():
            continue

        parent = tracklets_dir.parent
        lower = parent.name.lower()

        for i in range(1, 100):
            token = "vid{:02d}".format(i)
            if token in lower:
                result[token].append(parent)
                break

    return result


def choose_video_dir(candidates, source_video):
    if not candidates:
        raise RuntimeError(
            "Could not find folder for {}.".format(source_video)
        )

    # Prefer shortest path / closest to root.
    candidates = sorted(
        set(candidates),
        key=lambda p: (len(p.parts), str(p))
    )

    return candidates[0]


def discover_tracklet_manifests(video_dir):
    manifests = {}

    for path in Path(video_dir).rglob("*.json"):
        if "Tracklets" not in path.parts:
            continue

        try:
            with open(path, "r", encoding="utf-8") as handle:
                data = json.load(handle)
        except Exception:
            continue

        if not isinstance(data, dict):
            continue

        if "frames" not in data:
            continue

        stem = path.stem
        manifests[stem] = path

    return manifests


def build_image_index(video_dir):
    """
    Index original source images by basename inside one video folder.

    Searches recursively because users may keep JPEGs either:
        video_dir/*.jpg
    or
        video_dir/Annotations/*.jpg
    or another nested image folder.

    Tracklet rendered videos/images are ignored.
    """
    index = {}
    duplicates = defaultdict(list)

    for path in Path(video_dir).rglob("*"):
        if not path.is_file():
            continue

        if path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        if "Tracklets" in path.parts:
            continue

        name = path.name

        if name in index:
            duplicates[name].append(path)
            continue

        index[name] = path

    return index, duplicates


def load_manifest(path):
    with open(path, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    frames = data.get("frames", [])

    usable = []

    for frame in frames:
        source_image = frame.get("source_image")
        bbox = frame.get("bbox_xyxy")

        if not source_image or not bbox or len(bbox) != 4:
            continue

        # Tracklet time is what the human behavior intervals refer to.
        if "video_time_sec" in frame:
            t = float(frame["video_time_sec"])
        elif "track_time_sec" in frame:
            # Older generator.
            t = float(frame["track_time_sec"])
        else:
            continue

        usable.append({
            "time_sec": t,
            "source_image": source_image,
            "source_frame": frame.get("source_frame"),
            "source_timestamp_ms": frame.get("source_timestamp_ms"),
            "bbox_xyxy": [float(v) for v in bbox],
            "bbox_origin": frame.get("bbox_origin", "annotated"),
            "viewpoint": frame.get("viewpoint"),
        })

    usable.sort(key=lambda x: x["time_sec"])

    return {
        "label": data.get("label"),
        "video_file": data.get("video_file"),
        "frames": usable,
    }


def interval_clean_bounds(start, end, margin):
    """
    Remove a safety margin on BOTH sides of a behavior interval.

    If the interval is too short to support that margin, use the interval
    midpoint region rather than eliminating it entirely for frame model B.
    Temporal model C will still require a full clean window.
    """
    clean_start = start + margin
    clean_end = end - margin

    if clean_end <= clean_start:
        mid = 0.5 * (start + end)
        half = max(0.0, 0.25 * (end - start))
        clean_start = mid - half
        clean_end = mid + half

    return clean_start, clean_end


def crop_image(image, bbox, padding):
    h, w = image.shape[:2]

    x1, y1, x2, y2 = [float(v) for v in bbox]

    bw = x2 - x1
    bh = y2 - y1

    if bw <= 1 or bh <= 1:
        raise ValueError("Degenerate bbox")

    px = padding * bw
    py = padding * bh

    cx1 = max(0, int(math.floor(x1 - px)))
    cy1 = max(0, int(math.floor(y1 - py)))
    cx2 = min(w, int(math.ceil(x2 + px)))
    cy2 = min(h, int(math.ceil(y2 + py)))

    if cx2 <= cx1 or cy2 <= cy1:
        raise ValueError("Degenerate crop")

    return image[cy1:cy2, cx1:cx2].copy(), (
        cx1, cy1, cx2, cy2
    )


def link_or_copy(src, dst):
    """
    Prefer hardlinks so B/C dataset views do not duplicate crop storage.
    Fall back to copy if hardlinking is impossible.
    """
    src = Path(src)
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)

    if dst.exists():
        return

    try:
        os.link(str(src), str(dst))
    except OSError:
        shutil.copy2(str(src), str(dst))


def nearest_frame(rows, target_time):
    if not rows:
        return None, None

    def row_time(row):
        if "time_sec" in row:
            return float(row["time_sec"])
        return float(row["tracklet_time_sec"])

    best = min(
        rows,
        key=lambda r: abs(row_time(r) - target_time)
    )

    return best, abs(row_time(best) - target_time)


def make_window_id(tracklet, interval_idx, window_idx):
    return "{}_i{:03d}_w{:04d}".format(
        tracklet,
        interval_idx,
        window_idx,
    )


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Build frame and temporal Przewalski behavior datasets "
            "from verified intervals + tracklet manifests + source images."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "root",
        help=(
            "Root containing the per-video folders and interval CSV."
        ),
    )

    parser.add_argument(
        "--intervals",
        default="Przewalski_behavior_3class_intervals.csv",
        help=(
            "Verified 3-class interval CSV. Relative paths are resolved "
            "under root."
        ),
    )

    parser.add_argument(
        "--images-root",
        default=None,
        help=(
            "Optional separate root containing the original per-video "
            "image folders. Defaults to root."
        ),
    )

    parser.add_argument(
        "--output",
        default="Przewalski_behavior_training_dataset",
        help="Output directory; relative paths are resolved under root.",
    )

    parser.add_argument(
        "--test-videos",
        default="",
        help='Comma-separated final held-out videos, e.g. "vid03".',
    )

    parser.add_argument(
        "--val-videos",
        default="",
        help='Comma-separated validation videos, e.g. "vid04".',
    )

    parser.add_argument(
        "--val-horses",
        default="",
        help=(
            "Comma-separated additional validation horse IDs, e.g. "
            '"ph_vid01_004,ph_vid01_010".'
        ),
    )

    parser.add_argument(
        "--min-confidence",
        choices=("low", "medium", "high"),
        default="medium",
        help="Minimum interval confidence to include.",
    )

    parser.add_argument(
        "--boundary-margin-sec",
        type=float,
        default=0.25,
        help=(
            "Discard this much time on each side of behavior transitions."
        ),
    )

    parser.add_argument(
        "--padding",
        type=float,
        default=0.15,
        help="BBox padding fraction for clean training crops.",
    )

    parser.add_argument(
        "--jpeg-quality",
        type=int,
        default=95,
        help="JPEG quality for saved crops.",
    )

    parser.add_argument(
        "--frame-sample-fps",
        type=float,
        default=2.0,
        help=(
            "Sampling rate for single-frame model B. Full-rate crops are "
            "still retained for temporal model C."
        ),
    )

    parser.add_argument(
        "--window-sec",
        type=float,
        default=2.0,
        help="Temporal window duration for model C.",
    )

    parser.add_argument(
        "--window-stride-sec",
        type=float,
        default=1.0,
        help="Temporal window stride for model C.",
    )

    parser.add_argument(
        "--sequence-length",
        type=int,
        default=16,
        help="Number of frames per temporal window.",
    )

    parser.add_argument(
        "--max-frame-time-error-sec",
        type=float,
        default=0.10,
        help=(
            "Maximum error when matching an ideal temporal sample time "
            "to a real extracted frame."
        ),
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild an existing output directory.",
    )

    parser.add_argument(
        "--check-only",
        action="store_true",
        help=(
            "Validate folder/manifests/image availability and print the "
            "planned split without creating crops."
        ),
    )

    args = parser.parse_args()

    root = Path(args.root).expanduser().resolve()
    images_root = (
        Path(args.images_root).expanduser().resolve()
        if args.images_root
        else root
    )

    intervals_path = Path(args.intervals)
    if not intervals_path.is_absolute():
        intervals_path = root / intervals_path

    output = Path(args.output)
    if not output.is_absolute():
        output = root / output

    if not root.is_dir():
        raise RuntimeError(
            "Root does not exist: {}".format(root)
        )

    if not images_root.is_dir():
        raise RuntimeError(
            "Images root does not exist: {}".format(images_root)
        )

    if not intervals_path.is_file():
        raise RuntimeError(
            "Interval CSV does not exist: {}".format(intervals_path)
        )

    if args.boundary_margin_sec < 0:
        raise RuntimeError("--boundary-margin-sec must be >= 0")

    if args.padding < 0:
        raise RuntimeError("--padding must be >= 0")

    if args.frame_sample_fps <= 0:
        raise RuntimeError("--frame-sample-fps must be > 0")

    if args.window_sec <= 0:
        raise RuntimeError("--window-sec must be > 0")

    if args.window_stride_sec <= 0:
        raise RuntimeError("--window-stride-sec must be > 0")

    if args.sequence_length < 2:
        raise RuntimeError("--sequence-length must be >= 2")

    test_videos = set(comma_list(args.test_videos))
    val_videos = set(comma_list(args.val_videos))
    val_horses = set(comma_list(args.val_horses))

    overlap = test_videos.intersection(val_videos)
    if overlap:
        raise RuntimeError(
            "Videos cannot be both val and test: {}".format(
                ", ".join(sorted(overlap))
            )
        )

    intervals = read_intervals(
        str(intervals_path),
        args.min_confidence,
    )

    needed_videos = sorted(
        set(row["source_video"] for row in intervals)
    )

    data_video_candidates = discover_video_dirs(root)
    image_video_candidates = discover_video_dirs(images_root)

    video_dirs = {}
    image_dirs = {}

    for vid in needed_videos:
        video_dirs[vid] = choose_video_dir(
            data_video_candidates.get(vid, []),
            vid,
        )

        image_candidates = image_video_candidates.get(vid, [])
        if image_candidates:
            image_dirs[vid] = choose_video_dir(
                image_candidates,
                vid,
            )
        else:
            # If images are beneath the same folder as manifests.
            image_dirs[vid] = video_dirs[vid]

    print("Intervals CSV : {}".format(intervals_path))
    print("Data root     : {}".format(root))
    print("Images root   : {}".format(images_root))
    print("")
    print("Split policy:")
    print("  test videos : {}".format(
        ", ".join(sorted(test_videos)) or "(none)"
    ))
    print("  val videos  : {}".format(
        ", ".join(sorted(val_videos)) or "(none)"
    ))
    print("  val horses  : {}".format(
        ", ".join(sorted(val_horses)) or "(none)"
    ))
    print("")

    manifest_indexes = {}
    image_indexes = {}

    for vid in needed_videos:
        manifests = discover_tracklet_manifests(
            video_dirs[vid]
        )
        images, duplicates = build_image_index(
            image_dirs[vid]
        )

        manifest_indexes[vid] = manifests
        image_indexes[vid] = images

        print("{}:".format(vid))
        print("  data folder : {}".format(video_dirs[vid]))
        print("  image folder: {}".format(image_dirs[vid]))
        print("  manifests   : {}".format(len(manifests)))
        print("  images      : {}".format(len(images)))

        if duplicates:
            print(
                "  WARNING: {} duplicate image basenames; first copy "
                "will be used".format(len(duplicates))
            )

    print("")

    # Validate all referenced manifests/images before doing any destructive work.
    missing_manifests = []
    missing_images = []
    loaded_manifests = {}

    tracklets_needed = sorted(
        set(
            (row["source_video"], row["tracklet"])
            for row in intervals
        )
    )

    for vid, tracklet in tracklets_needed:
        path = manifest_indexes[vid].get(tracklet)

        if path is None:
            missing_manifests.append(
                (vid, tracklet)
            )
            continue

        manifest = load_manifest(path)
        loaded_manifests[(vid, tracklet)] = manifest

        for frame in manifest["frames"]:
            if frame["source_image"] not in image_indexes[vid]:
                missing_images.append(
                    (
                        vid,
                        tracklet,
                        frame["source_image"],
                    )
                )

    if missing_manifests:
        print("ERROR: missing tracklet manifests:")
        for item in missing_manifests[:20]:
            print("  {} / {}".format(*item))
        if len(missing_manifests) > 20:
            print("  ... and {} more".format(
                len(missing_manifests) - 20
            ))
        sys.exit(1)

    if missing_images:
        unique_missing = []
        seen = set()

        for item in missing_images:
            key = (item[0], item[2])
            if key in seen:
                continue
            seen.add(key)
            unique_missing.append(item)

        print("ERROR: missing original source images:")
        for vid, tracklet, name in unique_missing[:30]:
            print(
                "  {} / {} -> {}".format(
                    vid,
                    tracklet,
                    name,
                )
            )

        if len(unique_missing) > 30:
            print("  ... and {} more".format(
                len(unique_missing) - 30
            ))

        print("")
        print(
            "If your images live in another copy of the per-video folders, "
            "rerun with --images-root /path/to/that/root"
        )
        sys.exit(1)

    # Planned split summary.
    split_horses = defaultdict(set)
    split_videos = defaultdict(set)
    split_interval_sec = defaultdict(
        lambda: defaultdict(float)
    )

    for row in intervals:
        split = split_for_row(
            row["source_video"],
            row["horse_id"],
            test_videos,
            val_videos,
            val_horses,
        )

        split_horses[split].add(row["horse_id"])
        split_videos[split].add(row["source_video"])
        split_interval_sec[split][row["behavior"]] += (
            row["end_sec"] - row["start_sec"]
        )

    print("Planned split:")
    for split in ("train", "val", "test"):
        print(
            "  {:5s}: {:2d} horses; videos {}; raw interval time "
            "grazing={:.1f}s standing={:.1f}s walking={:.1f}s".format(
                split,
                len(split_horses[split]),
                ",".join(sorted(split_videos[split])) or "-",
                split_interval_sec[split]["grazing"],
                split_interval_sec[split]["standing"],
                split_interval_sec[split]["walking"],
            )
        )

    if args.check_only:
        print("")
        print("Check passed: all required manifests and images were found.")
        return

    ensure_empty_output(
        output,
        args.overwrite,
    )

    crops_root = output / "crops"
    b_root = output / "B_frame_classifier"
    c_root = output / "C_temporal"
    metadata_root = output / "metadata"

    crops_root.mkdir(parents=True, exist_ok=True)
    metadata_root.mkdir(parents=True, exist_ok=True)

    frame_rows = []
    window_rows = []
    split_rows = []

    # Cache crops so a crop is written only once.
    crop_cache = {}

    # For temporal window construction, collect eligible frame rows
    # per verified interval.
    interval_frame_rows = {}

    interval_counter = 0

    for interval in intervals:
        vid = interval["source_video"]
        tracklet = interval["tracklet"]
        horse_id = interval["horse_id"]
        behavior = interval["behavior"]

        split = split_for_row(
            vid,
            horse_id,
            test_videos,
            val_videos,
            val_horses,
        )

        manifest = loaded_manifests[(vid, tracklet)]
        frames = manifest["frames"]

        clean_start, clean_end = interval_clean_bounds(
            interval["start_sec"],
            interval["end_sec"],
            args.boundary_margin_sec,
        )

        current_interval_id = "I{:05d}".format(
            interval_counter
        )
        interval_counter += 1

        eligible = [
            f for f in frames
            if clean_start <= f["time_sec"] <= clean_end
        ]

        # Single-frame model B sampling target.
        sample_period = 1.0 / args.frame_sample_fps
        next_b_time = clean_start

        this_interval_rows = []

        for f in eligible:
            source_name = f["source_image"]
            image_path = image_indexes[vid][source_name]

            crop_key = (
                vid,
                tracklet,
                source_name,
            )

            if crop_key not in crop_cache:
                image = cv2.imread(
                    str(image_path),
                    cv2.IMREAD_COLOR,
                )

                if image is None:
                    raise RuntimeError(
                        "Could not read image: {}".format(
                            image_path
                        )
                    )

                crop, crop_box = crop_image(
                    image,
                    f["bbox_xyxy"],
                    args.padding,
                )

                crop_dir = (
                    crops_root
                    / vid
                    / tracklet
                )
                crop_dir.mkdir(
                    parents=True,
                    exist_ok=True,
                )

                crop_name = Path(source_name).stem + ".jpg"
                crop_path = crop_dir / crop_name

                ok = cv2.imwrite(
                    str(crop_path),
                    crop,
                    [
                        cv2.IMWRITE_JPEG_QUALITY,
                        int(args.jpeg_quality),
                    ],
                )

                if not ok:
                    raise RuntimeError(
                        "Failed to write crop {}".format(
                            crop_path
                        )
                    )

                crop_cache[crop_key] = (
                    crop_path,
                    crop_box,
                )

            crop_path, crop_box = crop_cache[crop_key]

            selected_for_b = False

            if f["time_sec"] + 1e-9 >= next_b_time:
                selected_for_b = True

                while next_b_time <= f["time_sec"] + 1e-9:
                    next_b_time += sample_period

            relative_crop = crop_path.relative_to(output)

            row = {
                "split": split,
                "source_video": vid,
                "horse_id": horse_id,
                "tracklet": tracklet,
                "interval_id": current_interval_id,
                "behavior": behavior,
                "confidence": interval["confidence"],
                "tracklet_time_sec": round(
                    f["time_sec"],
                    6,
                ),
                "source_image": source_name,
                "source_frame": (
                    f["source_frame"]
                    if f["source_frame"] is not None
                    else ""
                ),
                "source_timestamp_ms": (
                    f["source_timestamp_ms"]
                    if f["source_timestamp_ms"] is not None
                    else ""
                ),
                "bbox_origin": f["bbox_origin"],
                "viewpoint": (
                    f["viewpoint"]
                    if f["viewpoint"] is not None
                    else ""
                ),
                "crop_path": str(relative_crop),
                "selected_for_B": selected_for_b,
            }

            frame_rows.append(row)
            this_interval_rows.append(row)

            if selected_for_b:
                unique_name = (
                    "{}__{}__{}".format(
                        tracklet,
                        current_interval_id,
                        Path(source_name).stem,
                    )
                    + ".jpg"
                )

                b_path = (
                    b_root
                    / split
                    / behavior
                    / unique_name
                )

                link_or_copy(
                    crop_path,
                    b_path,
                )

        interval_frame_rows[current_interval_id] = {
            "interval": interval,
            "split": split,
            "clean_start": clean_start,
            "clean_end": clean_end,
            "rows": this_interval_rows,
        }

    # Temporal windows.
    for interval_id, pack in interval_frame_rows.items():
        interval = pack["interval"]
        split = pack["split"]
        rows = pack["rows"]

        clean_start = pack["clean_start"]
        clean_end = pack["clean_end"]

        if clean_end - clean_start < args.window_sec:
            continue

        start = clean_start
        window_idx = 0

        while start + args.window_sec <= clean_end + 1e-9:
            end = start + args.window_sec

            # 16 target samples across the 2-second interval, including
            # both ends. The actual source rate can be 7.999..., 8, etc.
            target_times = [
                start
                + j * args.window_sec
                / float(args.sequence_length - 1)
                for j in range(args.sequence_length)
            ]

            chosen = []
            valid = True

            for target in target_times:
                match, error = nearest_frame(
                    rows,
                    target,
                )

                if (
                    match is None
                    or error is None
                    or error > args.max_frame_time_error_sec
                ):
                    valid = False
                    break

                chosen.append(match)

            # Reject duplicate source crops in a nominal fixed-rate
            # sequence. This prevents hidden low-FPS windows.
            if valid:
                crop_paths = [
                    x["crop_path"]
                    for x in chosen
                ]

                if len(set(crop_paths)) != len(crop_paths):
                    valid = False

            if valid:
                window_id = make_window_id(
                    interval["tracklet"],
                    int(interval_id[1:]),
                    window_idx,
                )

                window_dir = (
                    c_root
                    / split
                    / interval["behavior"]
                    / window_id
                )

                frame_paths = []

                for j, chosen_row in enumerate(chosen):
                    src_crop = (
                        output
                        / chosen_row["crop_path"]
                    )

                    dst = (
                        window_dir
                        / "{:03d}.jpg".format(j)
                    )

                    link_or_copy(
                        src_crop,
                        dst,
                    )

                    frame_paths.append(
                        str(dst.relative_to(output))
                    )

                window_rows.append({
                    "window_id": window_id,
                    "split": split,
                    "source_video": interval["source_video"],
                    "horse_id": interval["horse_id"],
                    "tracklet": interval["tracklet"],
                    "interval_id": interval_id,
                    "behavior": interval["behavior"],
                    "confidence": interval["confidence"],
                    "start_sec": round(start, 6),
                    "end_sec": round(end, 6),
                    "sequence_length": args.sequence_length,
                    "window_dir": str(
                        window_dir.relative_to(output)
                    ),
                    "frame_paths_json": json.dumps(
                        frame_paths
                    ),
                })

                window_idx += 1

            start += args.window_stride_sec

    # Split assignment metadata: one row per horse.
    all_horses = sorted(
        set(
            (r["source_video"], r["horse_id"])
            for r in intervals
        )
    )

    for vid, horse in all_horses:
        split_rows.append({
            "source_video": vid,
            "horse_id": horse,
            "split": split_for_row(
                vid,
                horse,
                test_videos,
                val_videos,
                val_horses,
            ),
        })

    frame_fields = [
        "split",
        "source_video",
        "horse_id",
        "tracklet",
        "interval_id",
        "behavior",
        "confidence",
        "tracklet_time_sec",
        "source_image",
        "source_frame",
        "source_timestamp_ms",
        "bbox_origin",
        "viewpoint",
        "crop_path",
        "selected_for_B",
    ]

    window_fields = [
        "window_id",
        "split",
        "source_video",
        "horse_id",
        "tracklet",
        "interval_id",
        "behavior",
        "confidence",
        "start_sec",
        "end_sec",
        "sequence_length",
        "window_dir",
        "frame_paths_json",
    ]

    split_fields = [
        "source_video",
        "horse_id",
        "split",
    ]

    write_csv(
        metadata_root / "frames.csv",
        frame_rows,
        frame_fields,
    )

    write_csv(
        metadata_root / "windows.csv",
        window_rows,
        window_fields,
    )

    write_csv(
        metadata_root / "splits.csv",
        split_rows,
        split_fields,
    )

    # Per-split convenience CSVs.
    for split in ("train", "val", "test"):
        write_csv(
            metadata_root / "frames_{}.csv".format(split),
            [
                r for r in frame_rows
                if r["split"] == split
            ],
            frame_fields,
        )

        write_csv(
            metadata_root / "windows_{}.csv".format(split),
            [
                r for r in window_rows
                if r["split"] == split
            ],
            window_fields,
        )

    config = {
        "interval_csv": str(intervals_path),
        "classes": list(CLASSES),
        "test_videos": sorted(test_videos),
        "val_videos": sorted(val_videos),
        "val_horses": sorted(val_horses),
        "min_confidence": args.min_confidence,
        "boundary_margin_sec": args.boundary_margin_sec,
        "crop_padding_fraction": args.padding,
        "frame_model_sample_fps": args.frame_sample_fps,
        "temporal_window_sec": args.window_sec,
        "temporal_window_stride_sec": args.window_stride_sec,
        "temporal_sequence_length": args.sequence_length,
        "max_frame_time_error_sec": args.max_frame_time_error_sec,
    }

    with open(
        metadata_root / "build_config.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            config,
            handle,
            indent=2,
        )

    summary = {
        "B_frame_counts": {},
        "C_window_counts": {},
        "all_crop_count": len(crop_cache),
    }

    for split in ("train", "val", "test"):
        summary["B_frame_counts"][split] = {}
        summary["C_window_counts"][split] = {}

        for behavior in CLASSES:
            summary["B_frame_counts"][split][behavior] = sum(
                1
                for r in frame_rows
                if (
                    r["split"] == split
                    and r["behavior"] == behavior
                    and r["selected_for_B"]
                )
            )

            summary["C_window_counts"][split][behavior] = sum(
                1
                for r in window_rows
                if (
                    r["split"] == split
                    and r["behavior"] == behavior
                )
            )

    with open(
        metadata_root / "summary.json",
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
        )

    print("")
    print("Dataset created at:")
    print("  {}".format(output))
    print("")
    print("B: single-frame classifier @ {:.2f} fps sampling".format(
        args.frame_sample_fps
    ))
    for split in ("train", "val", "test"):
        vals = summary["B_frame_counts"][split]
        print(
            "  {:5s}: grazing {:4d} | standing {:4d} | walking {:4d}".format(
                split,
                vals["grazing"],
                vals["standing"],
                vals["walking"],
            )
        )

    print("")
    print(
        "C: {:.2f}s temporal windows, {} frames, {:.2f}s stride".format(
            args.window_sec,
            args.sequence_length,
            args.window_stride_sec,
        )
    )
    for split in ("train", "val", "test"):
        vals = summary["C_window_counts"][split]
        print(
            "  {:5s}: grazing {:4d} | standing {:4d} | walking {:4d}".format(
                split,
                vals["grazing"],
                vals["standing"],
                vals["walking"],
            )
        )

    print("")
    print(
        "Use metadata/summary.json and metadata/splits.csv to verify "
        "the final dataset before training."
    )


if __name__ == "__main__":
    main()

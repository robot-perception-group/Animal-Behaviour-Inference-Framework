#!/usr/bin/env python3
import argparse, csv, json, math, shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CLASSES = ("grazing", "standing", "walking")
SPLITS = ("train", "val", "test")


def read_csv(path):
    with open(path, "r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fieldnames):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def discover_video_dirs(root):
    out = defaultdict(list)
    for tracklets_dir in Path(root).rglob("Tracklets"):
        if not tracklets_dir.is_dir():
            continue
        parent = tracklets_dir.parent
        name = parent.name.lower()
        for i in range(1, 100):
            token = "vid{:02d}".format(i)
            if token in name:
                out[token].append(parent)
                break
    return out


def choose_video_dir(candidates, vid):
    if not candidates:
        raise RuntimeError("Could not find per-video folder for {}".format(vid))
    return sorted(set(candidates), key=lambda p: (len(p.parts), str(p)))[0]


def discover_manifests(video_dir):
    out = {}
    for p in Path(video_dir).rglob("*.json"):
        if "Tracklets" not in p.parts:
            continue
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(d, dict) and "frames" in d:
            out[p.stem] = p
    return out


def build_image_index(video_dir):
    out = {}
    for p in Path(video_dir).rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue
        if "Tracklets" in p.parts:
            continue
        out.setdefault(p.name, p)
    return out


def load_bbox_lookup(manifest_path):
    d = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    out = {}
    for fr in d.get("frames", []):
        name = fr.get("source_image")
        bbox = fr.get("bbox_xyxy")
        if name and bbox and len(bbox) == 4:
            out[name] = np.asarray([float(v) for v in bbox], dtype=np.float64)
    return out


def expand_bbox(bbox, margin):
    x1, y1, x2, y2 = map(float, bbox)
    w, h = x2 - x1, y2 - y1
    if w <= 1 or h <= 1:
        raise RuntimeError("Degenerate bbox {}".format(bbox))
    return (
        x1 - margin * w,
        y1 - margin * h,
        x2 + margin * w,
        y2 + margin * h,
    )


def crop_with_padding(image, bbox, border_mode):
    x1f, y1f, x2f, y2f = bbox
    x1, y1 = int(math.floor(x1f)), int(math.floor(y1f))
    x2, y2 = int(math.ceil(x2f)), int(math.ceil(y2f))

    h, w = image.shape[:2]
    sx1, sy1 = max(0, x1), max(0, y1)
    sx2, sy2 = min(w, x2), min(h, y2)

    if sx2 <= sx1 or sy2 <= sy1:
        raise RuntimeError("Expanded bbox outside image")

    crop = image[sy1:sy2, sx1:sx2].copy()
    left, top = sx1 - x1, sy1 - y1
    right, bottom = x2 - sx2, y2 - sy2
    used_padding = any(v > 0 for v in (left, top, right, bottom))

    if used_padding:
        if border_mode == "replicate":
            border_type, kwargs = cv2.BORDER_REPLICATE, {}
        elif border_mode == "reflect":
            border_type, kwargs = cv2.BORDER_REFLECT_101, {}
        else:
            border_type, kwargs = cv2.BORDER_CONSTANT, {"value": (0, 0, 0)}
        crop = cv2.copyMakeBorder(
            crop, top, bottom, left, right, border_type, **kwargs
        )

    return crop, used_padding, (x1, y1, x2, y2)


def reconstruct_window_frames(window, frame_groups):
    key = (
        window["source_video"],
        window["tracklet"],
        window["interval_id"],
    )
    candidates = frame_groups.get(key, [])
    if not candidates:
        raise RuntimeError("No frames for interval {}".format(key))

    n = int(window["sequence_length"])
    start = float(window["start_sec"])
    end = float(window["end_sec"])

    targets = [
        start + i * (end - start) / float(n - 1)
        for i in range(n)
    ]

    chosen = [
        min(
            candidates,
            key=lambda r: abs(float(r["tracklet_time_sec"]) - t)
        )
        for t in targets
    ]

    if len(set(r["crop_path"] for r in chosen)) != n:
        raise RuntimeError(
            "Duplicate reconstructed frames in {}".format(window["window_id"])
        )

    return chosen


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--output", default=None)
    ap.add_argument("--margins", default="0.10,0.15,0.20")
    ap.add_argument("--jpeg-quality", type=int, default=95)
    ap.add_argument(
        "--border-mode",
        choices=("replicate", "reflect", "black"),
        default="replicate",
    )
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    margins = [float(x.strip()) for x in args.margins.split(",") if x.strip()]
    for m in margins:
        if m < 0 or m > 1:
            raise RuntimeError("Bad margin {}".format(m))

    data_root = Path(args.data_root).expanduser().resolve()
    dataset = Path(args.dataset).expanduser().resolve()
    output = (
        Path(args.output).expanduser().resolve()
        if args.output else dataset / "C2_context_ablation"
    )

    windows_csv = dataset / "metadata" / "windows.csv"
    frames_csv = dataset / "metadata" / "frames.csv"

    windows = read_csv(windows_csv)
    frames = read_csv(frames_csv)

    if output.exists():
        if not args.overwrite:
            raise RuntimeError("Output exists: {}. Use --overwrite".format(output))
        shutil.rmtree(str(output))
    output.mkdir(parents=True, exist_ok=True)

    frame_groups = defaultdict(list)
    for row in frames:
        key = (
            row["source_video"],
            row["tracklet"],
            row["interval_id"],
        )
        frame_groups[key].append(row)

    for key in frame_groups:
        frame_groups[key].sort(
            key=lambda r: float(r["tracklet_time_sec"])
        )

    needed_videos = sorted(set(w["source_video"] for w in windows))
    candidates = discover_video_dirs(data_root)
    image_indexes = {}
    manifest_indexes = {}

    for vid in needed_videos:
        video_dir = choose_video_dir(candidates.get(vid, []), vid)
        image_indexes[vid] = build_image_index(video_dir)
        manifest_indexes[vid] = discover_manifests(video_dir)
        print(
            "{}: {} images, {} manifests".format(
                vid, len(image_indexes[vid]), len(manifest_indexes[vid])
            )
        )

    margin_keys = {}
    counts = {}
    padding_counts = {}
    metadata = {}

    for m in margins:
        pct = int(round(m * 100))
        key = "context_{:02d}pct".format(pct)
        margin_keys[m] = key
        counts[key] = defaultdict(lambda: defaultdict(int))
        padding_counts[key] = 0
        metadata[key] = []

    bbox_cache = {}

    print("")
    print("Temporal windows:", len(windows))
    print("Margins:", ", ".join("{}%".format(int(m * 100)) for m in margins))

    for wi, window in enumerate(windows, 1):
        split = window["split"]
        behavior = window["behavior"]
        vid = window["source_video"]
        tracklet = window["tracklet"]
        window_id = window["window_id"]

        chosen = reconstruct_window_frames(window, frame_groups)

        manifest_key = (vid, tracklet)
        if manifest_key not in bbox_cache:
            mp = manifest_indexes[vid].get(tracklet)
            if mp is None:
                raise RuntimeError(
                    "Missing manifest for {} / {}".format(vid, tracklet)
                )
            bbox_cache[manifest_key] = load_bbox_lookup(mp)
        bbox_lookup = bbox_cache[manifest_key]

        dirs = {}
        for m in margins:
            key = margin_keys[m]
            d = output / key / split / behavior / window_id
            d.mkdir(parents=True, exist_ok=True)
            dirs[m] = d

        source_names = []
        expanded_boxes = {m: [] for m in margins}
        padded = {m: False for m in margins}

        for j, row in enumerate(chosen):
            source_name = row["source_image"]
            source_names.append(source_name)

            source_path = image_indexes[vid].get(source_name)
            bbox = bbox_lookup.get(source_name)

            if source_path is None:
                raise RuntimeError("Missing source image {}".format(source_name))
            if bbox is None:
                raise RuntimeError("Missing bbox for {}".format(source_name))

            image = cv2.imread(str(source_path), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError("Could not read {}".format(source_path))

            for m in margins:
                crop, used_pad, box_i = crop_with_padding(
                    image,
                    expand_bbox(bbox, m),
                    args.border_mode,
                )
                dst = dirs[m] / "{:03d}.jpg".format(j)
                ok = cv2.imwrite(
                    str(dst),
                    crop,
                    [cv2.IMWRITE_JPEG_QUALITY, int(args.jpeg_quality)],
                )
                if not ok:
                    raise RuntimeError("Failed writing {}".format(dst))

                expanded_boxes[m].append(list(box_i))
                padded[m] = padded[m] or used_pad

        for m in margins:
            key = margin_keys[m]
            counts[key][split][behavior] += 1
            if padded[m]:
                padding_counts[key] += 1

            metadata[key].append({
                "window_id": window_id,
                "split": split,
                "source_video": vid,
                "horse_id": window["horse_id"],
                "tracklet": tracklet,
                "interval_id": window["interval_id"],
                "behavior": behavior,
                "confidence": window["confidence"],
                "start_sec": window["start_sec"],
                "end_sec": window["end_sec"],
                "sequence_length": window["sequence_length"],
                "context_fraction_each_side": m,
                "context_percent_each_side": int(round(m * 100)),
                "window_dir": str(dirs[m].relative_to(output / key)),
                "source_images_json": json.dumps(source_names),
                "expanded_boxes_json": json.dumps(expanded_boxes[m]),
                "used_border_padding": padded[m],
            })

        if wi % 50 == 0 or wi == len(windows):
            print(
                "Built {:4d}/{:4d} windows x {} margins".format(
                    wi, len(windows), len(margins)
                )
            )

    fields = [
        "window_id", "split", "source_video", "horse_id", "tracklet",
        "interval_id", "behavior", "confidence", "start_sec", "end_sec",
        "sequence_length", "context_fraction_each_side",
        "context_percent_each_side", "window_dir", "source_images_json",
        "expanded_boxes_json", "used_border_padding",
    ]

    print("")
    for m in margins:
        key = margin_keys[m]
        root = output / key
        md = root / "_metadata"
        md.mkdir(parents=True, exist_ok=True)
        write_csv(md / "windows.csv", metadata[key], fields)

        summary = {
            "representation":
                "target-following per-frame bbox crop with local context",
            "context_fraction_each_side": m,
            "context_percent_each_side": int(round(m * 100)),
            "source_windows_csv": str(windows_csv),
            "window_count": len(metadata[key]),
            "padded_window_count": padding_counts[key],
            "counts": {
                split: {
                    cls: counts[key][split][cls]
                    for cls in CLASSES
                }
                for split in SPLITS
            },
        }
        (md / "summary.json").write_text(
            json.dumps(summary, indent=2),
            encoding="utf-8",
        )

        print("{}:".format(key))
        for split in SPLITS:
            print(
                "  {:5s}: grazing {:4d} | standing {:4d} | walking {:4d}".format(
                    split,
                    counts[key][split]["grazing"],
                    counts[key][split]["standing"],
                    counts[key][split]["walking"],
                )
            )
        print(
            "  boundary-padded windows: {} / {}".format(
                padding_counts[key], len(metadata[key])
            )
        )


if __name__ == "__main__":
    main()

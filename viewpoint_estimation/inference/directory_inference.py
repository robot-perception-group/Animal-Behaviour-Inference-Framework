from __future__ import annotations

import csv
import json
from pathlib import Path

from viewpoint_estimation.predictor import ViewpointInference
from viewpoint_estimation.inference.load_config import load_config

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

CONFIG_PATH = Path(__file__).with_name("viewpoint_config.yaml")


def predict_image_directory(cfg: dict) -> list[dict[str, str]]:
    predictor = ViewpointInference(cfg)
    image_paths = sorted(p for p in Path(cfg["data_path"]).iterdir() if p.suffix.lower() in IMG_EXTS)
    rows: list[dict[str, str]] = []
    total = len(image_paths)
    for index, image_path in enumerate(image_paths, start=1):
        angle, keypoints = predictor.predict_with_keypoints(image_path)
        kpts_str = ""
        if keypoints is not None:
            kpts_str = json.dumps(keypoints.tolist())
        rows.append({
            "filename": image_path.name,
            "angle": "" if angle is None else f"{angle:.4f}",
            "keypoints": kpts_str
        })
        print(f"\rProcessing images: {index}/{total}", end="", flush=True)

    out_path = Path(cfg["data_path"]) / "viewpoint_predictions.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as file_handle:
        writer = csv.DictWriter(file_handle, fieldnames=["filename", "angle", "keypoints"])
        writer.writeheader()
        writer.writerows(rows)
    if total:
        print()
    print(f"Saved CSV to {out_path}")
    return rows


def run_from_config(config_path: str | Path = CONFIG_PATH):
    return predict_image_directory(load_config(config_path))


if __name__ == "__main__":
    run_from_config()

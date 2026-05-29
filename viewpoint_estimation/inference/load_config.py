from __future__ import annotations

from pathlib import Path

import yaml

CONFIG_PATH = Path(__file__).resolve().parent.parent / "viewpoint_config.yaml"


def load_config(config_path: str | Path = CONFIG_PATH) -> dict:
    config_path = Path(config_path)
    cfg = yaml.safe_load(config_path.read_text()) or {}
    root = config_path.parent
    mode = cfg.get("mode", "sideview")
    return {
        "mode": mode,
        "run": cfg.get("run", "single"),
        "data_path": str(root / cfg.get("data_path", "")),
        "output_csv": str(root / cfg.get("output_csv", "viewpoint_predictions.csv")),
        "feature_type": cfg.get("feature_type", "expert"),
        "hidden_dims": cfg.get("hidden_dims", [1024, 512, 256]),
        "device": cfg.get("device", "auto"),
        "yolo_conf": float(cfg.get("yolo_conf", 0.25)),
        "min_keypoints": int(cfg.get("min_keypoints", 6)),
        "min_kpt_conf": float(cfg.get("min_kpt_conf", 0.15)),
        "keypoint_weights": str(root / cfg["models"][mode]["keypoint_weights"]),
        "viewpoint_weights": str(root / cfg["viewpoint_weights"]),
    }

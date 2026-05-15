from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
import tempfile

import yaml

from viewpoint_estimation.predictor import ViewpointInference


CONFIG_PATH = Path(__file__).with_name("viewpoint_config.yaml")


@lru_cache(maxsize=4)
def load_viewpoint_config(config_path: str | Path = CONFIG_PATH) -> dict:
    """Load and normalize viewpoint-estimation configuration."""
    path = Path(config_path)
    cfg = yaml.safe_load(path.read_text()) or {}
    root = path.parent
    mode = cfg.get("mode", "sideview")
    return {
        "mode": mode,
        "run": cfg.get("run", "single"),
        "image_path": str(root / cfg.get("image_path", "")),
        "image_dir": str(root / cfg.get("image_dir", "")),
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


def estimate_viewpoints(qimage, shapes, cfg):
    """Estimate viewpoint for each shape via cropped qimage regions.

    Returns list of `(shape, angle)` tuples, where `angle` can be None.
    """
    inf = ViewpointInference(cfg)
    results = []
    for shape in shapes:
        pts = getattr(shape, "points", [])
        xs = [float(p.x() if hasattr(p, "x") else p[0]) for p in pts]
        ys = [float(p.y() if hasattr(p, "y") else p[1]) for p in pts]
        if xs and ys:
            x0, y0 = max(0, int(min(xs))), max(0, int(min(ys)))
            x1, y1 = min(qimage.width() - 1, int(max(xs))), min(qimage.height() - 1, int(max(ys)))
            w, h = x1 - x0 + 1, y1 - y0 + 1
            crop = qimage.copy(x0, y0, w, h)
            tf = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            tf.close()
            crop.save(tf.name, "PNG")
            try:
                angle = inf.predict_angle(tf.name)
            except Exception:
                angle = None
            finally:
                try:
                    os.remove(tf.name)
                except Exception:
                    pass
        else:
            angle = None
        results.append((shape, angle))
    return results

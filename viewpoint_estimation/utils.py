"""Utility functions for viewpoint estimation integration with labelme."""
from __future__ import annotations

import os
import tempfile
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import yaml
from qtpy.QtGui import QImage

from viewpoint_estimation.predictor import ViewpointInference


CONFIG_PATH = Path(__file__).resolve().parent / "viewpoint_config.yaml"


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
        "feature_type": cfg.get("feature_type", "expert"),
        "hidden_dims": cfg.get("hidden_dims", [1024, 512, 256]),
        "device": cfg.get("device", "auto"),
        "yolo_conf": float(cfg.get("yolo_conf", 0.25)),
        "min_keypoints": int(cfg.get("min_keypoints", 6)),
        "min_kpt_conf": float(cfg.get("min_kpt_conf", 0.15)),
        "keypoint_weights": str(root / cfg["models"][mode]["keypoint_weights"]),
        "viewpoint_weights": str(root / cfg["viewpoint_weights"]),
    }


def _qimage_to_cv2(qimage: QImage) -> np.ndarray:
    """Convert Qt QImage to OpenCV format (BGR)."""
    width = qimage.width()
    height = qimage.height()
    ptr = qimage.bits()
    ptr.setsize(qimage.byteCount())
    arr = np.array(ptr).reshape(height, width, 4)  # RGBA
    # Convert RGBA to BGR for OpenCV
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
    return bgr


def estimate_viewpoints(qimage: QImage, shapes: list, cfg: dict) -> list[tuple]:
    """Estimate viewpoint for each shape via cropped image regions.

    Args:
        qimage: Qt QImage object
        shapes: List of Shape objects with points
        cfg: Configuration dictionary

    Returns:
        List of (shape, angle) tuples, where angle can be None.
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
            
            # Convert QImage crop to OpenCV format
            crop_cv = _qimage_to_cv2(crop)
            
            try:
                angle = inf.predict_angle_from_image(crop_cv)
            except Exception:
                angle = None
        else:
            angle = None
        
        results.append((shape, angle))
    
    return results


def estimate_viewpoints_with_keypoints(
    qimage: QImage, shapes: list, cfg: dict
) -> list[tuple]:
    """Estimate viewpoint and keypoints for each shape.

    Args:
        qimage: Qt QImage object
        shapes: List of Shape objects with points
        cfg: Configuration dictionary

    Returns:
        List of (shape, angle, keypoints) tuples, where keypoints is
        a list of (x, y, confidence) tuples.
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
            
            # Convert QImage crop to OpenCV format
            crop_cv = _qimage_to_cv2(crop)
            
            try:
                angle, keypoints_xy = inf._predict_angle_from_image(crop_cv)
                if keypoints_xy is not None:
                    # Get keypoint confidences
                    result = inf.yolo.predict(
                        source=crop_cv,
                        conf=cfg["yolo_conf"],
                        device=str(inf.device),
                        verbose=False,
                    )[0]
                    
                    if result.keypoints is not None and result.keypoints.conf is not None:
                        idx = 0  # Use first detection (already filtered by _predict_angle_from_image)
                        kpts_conf = result.keypoints.conf[idx].cpu().numpy()
                        keypoints = [
                            (float(pt[0]), float(pt[1]), float(kpts_conf[i]))
                            for i, pt in enumerate(keypoints_xy)
                        ]
                    else:
                        keypoints = [
                            (float(pt[0]), float(pt[1]), 1.0)
                            for pt in keypoints_xy
                        ]
                else:
                    angle = None
                    keypoints = []
            except Exception:
                angle = None
                keypoints = []
        else:
            angle = None
            keypoints = []
        
        results.append((shape, angle, keypoints))
    
    return results

from __future__ import annotations

from pathlib import Path
import logging

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO

from viewpoint_estimation.util.keypoints import _build_features

logger = logging.getLogger(__name__)

class ViewpointPredictor(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int]):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(0.15)]
            prev = h
        layers.append(nn.Linear(prev, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return F.normalize(self.net(x), p=2, dim=1, eps=1e-8)




def _load_state_dict(path: str, device: torch.device):
    
    ckpt = torch.load(path, map_location=device, weights_only=True)

    return ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt


def _infer_hidden_dims_from_state_dict(state_dict: dict) -> list[int]:
    hidden_dims: list[int] = []
    idx = 0
    while True:
        weight_key = f"net.{idx}.weight"
        if weight_key not in state_dict:
            break
        out_dim = int(state_dict[weight_key].shape[0])
        if out_dim == 2:
            break
        hidden_dims.append(out_dim)
        idx += 4
    if not hidden_dims:
        raise ValueError("Could not infer hidden dimensions from viewpoint checkpoint")
    return hidden_dims


class ViewpointInference:


    def __init__(self, cfg: dict):
        self.cfg = cfg
        requested_device = str(cfg["device"]).strip()
        # Handle "auto" device selection
        if requested_device == "auto":
            requested_device = "cuda" if torch.cuda.is_available() else "cpu"
        elif requested_device.startswith("cuda") and not torch.cuda.is_available():
            logger.warning("CUDA requested (%s) but not available. Falling back to CPU.", requested_device)
            requested_device = "cpu"
        self.device = torch.device(requested_device)
        self.yolo = YOLO(cfg["keypoint_weights"])
        state = _load_state_dict(cfg["viewpoint_weights"], self.device)
        input_dim = int(state["net.0.weight"].shape[1])
        hidden_dims = list(cfg.get("hidden_dims") or _infer_hidden_dims_from_state_dict(state))
        inferred_hidden_dims = _infer_hidden_dims_from_state_dict(state)
        if hidden_dims != inferred_hidden_dims:
            hidden_dims = inferred_hidden_dims
        self.cfg["hidden_dims"] = hidden_dims
        self.model = ViewpointPredictor(input_dim=input_dim, hidden_dims=hidden_dims).to(self.device)
        self.model.load_state_dict(state)
        self.model.eval()

    def _predict_angle_from_image(self, image: np.ndarray) -> tuple[float | None, np.ndarray | None]:
        result = self.yolo.predict(source=image, conf=self.cfg["yolo_conf"], device=str(self.device), verbose=False)[0]
        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            logger.warning("No keypoints detected by YOLO.")
            return None, None
        idx = int(np.argmax(result.boxes.conf.cpu().numpy())) if result.boxes is not None and result.boxes.conf is not None else 0
        xy = result.keypoints.xy[idx].cpu().numpy()
        conf = result.keypoints.conf[idx].cpu().numpy()
        if int((conf >= self.cfg["min_kpt_conf"]).sum()) < self.cfg["min_keypoints"]:
            logger.warning("Insufficient keypoints above confidence threshold.")
            return None, None
        h, w = image.shape[:2]
        kpts_xyc = np.column_stack((xy[:, 0] / w, xy[:, 1] / h, conf))
        feats = _build_features(kpts_xyc, self.cfg["feature_type"])
        out = self.model(torch.from_numpy(feats).unsqueeze(0).to(self.device))[0]
        angle = float(np.degrees(np.arctan2(out[0].item(), out[1].item())) % 360.0)
        return angle, xy

    @torch.no_grad()
    def predict_angle(self, image_path: str | Path) -> float | None:
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Failed to read image at %s", image_path)
            return None
        angle, _ = self._predict_angle_from_image(image)
        return angle

    @torch.no_grad()
    def predict_angle_from_image(self, image: np.ndarray) -> float | None:
        angle, _ = self._predict_angle_from_image(image)
        return angle

    @torch.no_grad()
    def predict_with_keypoints(self, image_path: str | Path) -> tuple[float | None, np.ndarray | None]:
        """Predict angle and return keypoints."""
        image = cv2.imread(str(image_path))
        if image is None:
            logger.warning("Failed to read image at %s", image_path)
            return None, None
        return self._predict_angle_from_image(image)

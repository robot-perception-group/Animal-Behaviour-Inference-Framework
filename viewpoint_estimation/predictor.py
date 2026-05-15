from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics import YOLO


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


def _build_features(kpts_xyc: np.ndarray, feature_type: str) -> np.ndarray:
    x_raw = kpts_xyc.flatten()
    min_pt, max_pt = kpts_xyc[:, :2].min(axis=0), kpts_xyc[:, :2].max(axis=0)
    width, height = max_pt[0] - min_pt[0], max_pt[1] - min_pt[1]
    aspect_ratio, bbox_area = width / (height + 1e-6), width * height
    spine_vec = kpts_xyc[0, :2] - kpts_xyc[16, :2]
    l_f_sho, r_f_sho, l_b_hip, r_b_hip, l_f_ft, l_b_ft = 5, 8, 11, 14, 7, 13
    hip_dist = np.linalg.norm(kpts_xyc[l_b_hip, :2] - kpts_xyc[r_b_hip, :2])
    sho_dist = np.linalg.norm(kpts_xyc[l_f_sho, :2] - kpts_xyc[r_f_sho, :2])
    stance_dist = np.linalg.norm(kpts_xyc[l_f_ft, :2] - kpts_xyc[l_b_ft, :2])
    direction_sign = kpts_xyc[l_f_sho, 0] - kpts_xyc[r_f_sho, 0]
    torso_pts = kpts_xyc[[l_f_sho, r_f_sho, r_b_hip, l_b_hip], :2]
    x_coords, y_coords = torso_pts[:, 0], torso_pts[:, 1]
    torso_area = 0.5 * np.abs(np.dot(x_coords, np.roll(y_coords, 1)) - np.dot(y_coords, np.roll(x_coords, 1)))
    if feature_type == "symmetry":
        feats = np.concatenate([x_raw, [aspect_ratio, bbox_area], spine_vec, [direction_sign]])
    elif feature_type == "stance":
        feats = np.concatenate([x_raw, [aspect_ratio, bbox_area], spine_vec, [stance_dist]])
    else:
        if feature_type != "expert":
            raise ValueError(f"Unsupported feature_type: {feature_type}")
        feats = np.concatenate([x_raw, [aspect_ratio, bbox_area], spine_vec, [hip_dist, sho_dist, stance_dist, direction_sign, torso_area]])
    return feats.astype(np.float32)


def _load_state_dict(path: str, device: torch.device):
    ckpt = torch.load(path, map_location=device)
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
        self.device = torch.device("cuda:0" if cfg["device"] == "auto" and torch.cuda.is_available() else cfg["device"])
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

    @torch.no_grad()
    def predict_angle(self, image_path: str | Path) -> float | None:
        image = cv2.imread(str(image_path))
        if image is None:
            return None
        result = self.yolo.predict(source=image, conf=self.cfg["yolo_conf"], device=str(self.device), verbose=False)[0]
        if result.keypoints is None or result.keypoints.xy is None or len(result.keypoints.xy) == 0:
            return None
        idx = int(np.argmax(result.boxes.conf.cpu().numpy())) if result.boxes is not None and result.boxes.conf is not None else 0
        xy = result.keypoints.xy[idx].cpu().numpy()
        conf = result.keypoints.conf[idx].cpu().numpy()
        if int((conf >= self.cfg["min_kpt_conf"]).sum()) < self.cfg["min_keypoints"]:
            return None
        h, w = image.shape[:2]
        kpts_xyc = np.column_stack((xy[:, 0] / w, xy[:, 1] / h, conf))
        feats = _build_features(kpts_xyc, self.cfg["feature_type"])
        out = self.model(torch.from_numpy(feats).unsqueeze(0).to(self.device))[0]
        angle = float(np.degrees(np.arctan2(out[0].item(), out[1].item())) % 360.0)
        return angle

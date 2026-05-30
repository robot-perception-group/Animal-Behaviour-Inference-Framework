from __future__ import annotations

from pathlib import Path

from viewpoint_estimation.predictor import ViewpointInference
from viewpoint_estimation.inference.load_config import load_config


def predict_single_image(cfg: dict) -> float | None:
    angle = ViewpointInference(cfg).predict_angle(cfg["data_path"])
    print(angle)
    return angle


if __name__ == "__main__":
    config_path = Path(__file__).resolve().parent.parent / "viewpoint_config.yaml"
    predict_single_image(load_config(config_path)) 
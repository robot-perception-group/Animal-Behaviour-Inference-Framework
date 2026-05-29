from pathlib import Path

from inference.directory_inference import predict_image_directory
from inference.frame_inference import predict_single_image
from inference.load_config import load_config



if __name__ == "__main__":
    cfg = load_config()
    data_path = Path(cfg["data_path"])
    if data_path.is_file():
        predict_single_image(cfg)
    elif data_path.is_dir():
        predict_image_directory(cfg)
    else:
        print(f"Invalid data_path: {data_path}")
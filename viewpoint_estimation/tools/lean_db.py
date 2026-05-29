import pandas as pd
import shutil
from pathlib import Path
from collections import defaultdict
import re
import numpy as np
import sys; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from inference.directory_inference import predict_image_directory
from inference.load_config import load_config

def build_lean_db(db_source_path, csv_path, dest_path, min_per_bin=10):

    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['angle', 'filename'])
    
    db_source_path = Path(db_source_path)
    dest_path = Path(dest_path)
    
    if dest_path.exists():
        shutil.rmtree(dest_path)
    dest_path.mkdir(parents=True, exist_ok=True)

    def get_bin(angle):
        try:
            return int((round(float(angle) / 45) * 45) % 360)
        except (ValueError, TypeError):
            return None

    def get_id(fname):
        match = re.match(r'([a-zA-Z0-9]+)_', str(fname))
        return match.group(1).lower() if match else None

    df['zebra_id'] = df['filename'].apply(get_id)
    df['vp_bin'] = df['angle'].apply(get_bin)
    df = df.dropna(subset=['zebra_id', 'vp_bin'])

    organizer = defaultdict(lambda: defaultdict(list))
    for _, row in df.iterrows():
        organizer[row['zebra_id']][int(row['vp_bin'])].append(row['filename'])

    total_copied = 0

    for _, bins in organizer.items():
        selected_files = []
        
        for b in sorted(bins.keys()):
            images = bins[b]
            num_available = len(images)
            
            if num_available <= min_per_bin:
                # take everything if we have less than the target
                selected_files.extend(images)
            else:
                #  striding, pick indices evenly spaced across the sequence
                indices = np.linspace(0, num_available - 1, min_per_bin, dtype=int)
                selected_files.extend([images[i] for i in indices])

        for fname in selected_files:
            src = db_source_path / fname
            if src.exists():
                shutil.copy(src, dest_path / fname)
                total_copied += 1

    print(f"Success! Created {total_copied} images.")

if __name__ == "__main__": 
    SOURCE_DB = Path(".../database")  
    
    CSV_PATH = None
    
    DEST_ROOT = Path(".../")
    
    DEST_DB = DEST_ROOT / "lean_db"

    if CSV_PATH is None:
        cfg = load_config()
        cfg["data_path"] = str(SOURCE_DB)
        predict_image_directory(cfg)
        CSV_PATH = SOURCE_DB / "viewpoint_predictions.csv"
    if not SOURCE_DB.exists():
        print(f"ERROR: Source database not found at {SOURCE_DB}")
    elif not CSV_PATH.exists():
        print(f"ERROR: Viewpoint CSV not found at {CSV_PATH}")
    else:
        build_lean_db(
            db_source_path=SOURCE_DB,
            csv_path=CSV_PATH,
            dest_path=DEST_DB
        )
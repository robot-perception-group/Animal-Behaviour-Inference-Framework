import csv
import re
import os
from collections import defaultdict
from pathlib import Path

def analyze_db(csv_path: str | Path):
    csv_path = str(csv_path)
    
    if not os.path.exists(csv_path):
        print(f"Error: Could not find CSV file at {csv_path}")
        return

    data = defaultdict(list)
    
    with open(csv_path, mode='r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get('angle'):
                continue
                
            id_match = re.match(r'([a-zA-Z0-9]+)_', row['filename'])
            if id_match:
                try:
                    data[id_match.group(1).lower()].append(float(row['angle']))
                except ValueError:
                    continue 

    bins = [0, 45, 90, 135, 180, 225, 270, 315]
    
    output_path = os.path.join(os.path.dirname(csv_path), "viewpoint_coverage.txt")
    
    with open(output_path, mode='w', encoding='utf-8') as out_file:
        out_file.write(f"{'ID':<10} | {'IMG':<4} | {'BINS':<4} | {'COV%':<6} | {'GAPS'}\n")
        out_file.write("-" * 55 + "\n")

        for aid in sorted(data.keys()):
            angs = data[aid]
            covered = {round(a / 45) * 45 % 360 for a in angs}
            gaps = [str(b) for b in bins if b not in covered]
            
            count = len(angs)
            b_hits = len(covered)
            perc = (b_hits / 8) * 100

            line = f"{aid:<10} | {count:<4} | {b_hits:<4} | {perc:>5.1f}% | {', '.join(gaps)}\n"
            out_file.write(line)
    
    print(f"Analysis saved to: {output_path}")

if __name__ == "__main__":
    CSV_PATH = Path("...demo/viewpoint_predictions.csv")
    analyze_db(CSV_PATH)
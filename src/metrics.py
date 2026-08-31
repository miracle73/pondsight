import csv
from pathlib import Path
from typing import Optional
import pandas as pd


def write_timeseries(rows: list, out: str):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "fish_count", "biomass_kg", "feeding_score"])
        w.writerows(rows)


def write_metrics(
    out: str,
    ap50: float,
    ap50_95: float,
    id_switches: int,
    mean_track_len: float,
    fps: float,
    gt_csv: Optional[str] = None,
):
    ap50_str = "N/A" if ap50 is None else f"{ap50:.4f}"
    ap50_95_str = "N/A" if ap50_95 is None else f"{ap50_95:.4f}"
    lines = [
        "# Detection & Tracking Metrics",
        f"- mAP50: {ap50_str}",
        f"- mAP50-95: {ap50_95_str}",
        f"- Total ID switches: {id_switches}",
        f"- Mean track length (frames): {mean_track_len:.1f}",
        f"- Runtime FPS: {fps:.1f}",
    ]
    if gt_csv and Path(gt_csv).exists():
        lines.append(f"- Ground-truth file: {gt_csv}")
        lines.append("- Weight MAE: (compute against GT)")
    Path(out).write_text("\n".join(lines) + "\n")


def compute_weight_mae(pred_tracks: dict, gt_csv: str) -> float:
    import numpy as np
    gt = pd.read_csv(gt_csv)
    if "weight_kg" not in gt.columns:
        return float("nan")
    return float(np.mean(np.abs(gt["weight_kg"].values)))

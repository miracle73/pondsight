import argparse
import time
from pathlib import Path

import cv2
import yaml
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.detect import get_model, train_model, validate, fish_class_ids
from src.track import Tracker
from src.biomass import BiomassEstimator
from src.activity import FeedingActivity
from src.metrics import write_timeseries, write_metrics


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def run(cfg: dict, video_path: str, skip_train: bool, progress=None):
    progress = progress or (lambda message, percent: None)
    progress("Loading detection model...", 2)
    out_dir = Path(cfg["paths"]["output_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # Train or load
    data_yaml = None
    if skip_train:
        print("Skipping training, using provided/fallback weights.")
    else:
        if cfg["training"]["dataset_slug"]:
            try:
                print("Training on Roboflow dataset...")
                w = train_model(cfg)
                cfg["paths"]["model_weights"] = w
            except Exception as e:
                print(f"Training failed ({e}), falling back to pretrained.")
                data_yaml = None

    model = get_model(cfg)
    fish_ids = fish_class_ids(model, cfg)

    # Validate
    print("Running validation...")
    ap = validate(model, data_yaml)
    if ap["mAP50"] is not None:
        print(f"  mAP50={ap['mAP50']:.4f}  mAP50-95={ap['mAP50-95']:.4f}")
    else:
        print("  No validation dataset; metrics reported as N/A.")

    # Setup components
    tracker = Tracker()
    estimator = BiomassEstimator(
        px_per_cm=cfg["scale"]["px_per_cm"],
        a=cfg["allometric"]["a"],
        b=cfg["allometric"]["b"],
    )
    activity = FeedingActivity(
        smooth_window=cfg["activity"]["smooth_window"],
        flow_scale=cfg["activity"]["flow_scale"],
    )

    # Video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Cannot open video. Please upload a readable video file.")
    fps_vid = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_dir / "annotated.mp4"), fourcc, fps_vid, (w, h))

    if not writer.isOpened():
        cap.release()
        writer.release()
        raise RuntimeError("Cannot create the output video.")
    total_expected = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress("Analyzing video...", 5)

    rows = []
    frame_idx = 0
    t_start = time.time()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect + Track
            results = model.track(
                frame,
                conf=cfg["model"]["conf"],
                iou=cfg["model"]["iou"],
                imgsz=cfg["model"]["imgsz"],
                tracker=cfg["tracking"].get("tracker", "bytetrack.yaml"),
                classes=fish_ids,
                persist=cfg["tracking"]["persist"],
                verbose=False,
            )
            r = results[0]
            if r.boxes is not None:
                r = r[np.isin(r.boxes.cls.cpu().numpy().astype(int), fish_ids)]
            tracker.update(r)

            active = [tracker.tracks[tid] for tid in tracker.active_ids if len(tracker.tracks[tid].lengths_px) > 0]
            biomass = estimator.frame_biomass(active)

            # Boxes for flow
            boxes_for_flow = []
            if r.boxes is not None and r.boxes.xyxy is not None:
                boxes_for_flow = r.boxes.xyxy.cpu().numpy()

            feed_score = activity.update(gray, boxes_for_flow)
            fish_count = len(active)

            # Draw
            if r.boxes is not None and r.boxes.xyxy is not None:
                xyxy = r.boxes.xyxy.cpu().numpy()
                ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [0] * len(xyxy)
                for box, tid in zip(xyxy, ids):
                    x1, y1, x2, y2 = map(int, box)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    lbl = f"ID{tid}"
                    if tid in tracker.tracks:
                        t = tracker.tracks[tid]
                        cm = estimator.length_cm(t.mean_length_px)
                        kg = estimator.per_track_kg(t)
                        lbl += f" {cm:.0f}cm {kg:.2f}kg"
                    cv2.putText(frame, lbl, (x1, y1 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            # HUD
            cv2.putText(frame, f"Approx. biomass: {biomass:.2f} kg", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.putText(frame, f"Movement: {feed_score:.2f}", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 200, 0), 2)
            cv2.putText(frame, f"Fish: {fish_count}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

            writer.write(frame)
            rows.append([frame_idx, fish_count, round(biomass, 4), round(feed_score, 4)])
            frame_idx += 1
            if frame_idx == 1 or frame_idx % 10 == 0:
                progress(f"Analyzed {frame_idx} / {total_expected or '?'} frames",
                         min(95, 5 + 90 * frame_idx / max(total_expected, frame_idx)))

    finally:
        cap.release()
        writer.release()

    elapsed = time.time() - t_start
    if frame_idx == 0:
        raise ValueError("The video contains no readable frames.")
    progress("Saving results...", 97)

    total_frames = frame_idx
    runtime_fps = total_frames / elapsed if elapsed > 0 else 0

    # Write outputs
    write_timeseries(rows, str(out_dir / "timeseries.csv"))
    write_metrics(
        str(out_dir / "metrics.md"),
        ap["mAP50"], ap["mAP50-95"],
        tracker.total_id_switches(),
        tracker.mean_track_length(),
        runtime_fps,
        cfg["paths"].get("gt_csv"),
    )
    plot_timeseries(str(out_dir / "timeseries.csv"), str(out_dir / "plot.png"))

    print(f"\nDone. {total_frames} frames in {elapsed:.1f}s ({runtime_fps:.1f} FPS)")
    print(f"Outputs in {out_dir}/")


def plot_timeseries(csv_path: str, out: str):
    import pandas as pd
    df = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    ax1.plot(df["frame"], df["biomass_kg"], color="tab:red")
    ax1.set_ylabel("Biomass (kg)")
    ax1.set_title("Approximate visible biomass (calibration required)")
    ax1.grid(True, alpha=0.3)
    ax2.plot(df["frame"], df["feeding_score"], color="tab:blue")
    ax2.set_ylabel("Movement Score")
    ax2.set_xlabel("Frame")
    ax2.set_title("Relative movement (not verified feeding)")
    ax2.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out, dpi=150)
    plt.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--video", required=True)
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--skip-train", action="store_true")
    args = p.parse_args()
    cfg = load_config(args.config)
    run(cfg, args.video, args.skip_train)

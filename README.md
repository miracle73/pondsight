# PondSight

Fish feed accounts for ~75% of aquaculture pond operating costs, yet farmers have no reliable way to see whether fish are actively consuming feed or whether it is sinking uneaten to the bottom. PondSight is a lightweight computer-vision pipeline that detects, tracks, and estimates the biomass of fish in underwater video, computes a feeding-activity score from optical flow, and outputs annotated video with per-fish length/weight estimates alongside summary metrics.

## Web App

PondSight ships with a Flask web app for uploading fish videos and running the pipeline:

```bash
python app.py
```

Open http://localhost:5000, upload a fish video, and watch it run. Results (annotated video, plot, metrics.md, timeseries.csv) appear for each job.

### CLI

```bash
pip install -r requirements.txt
python download_sample.py          # grab a test clip
python run.py --video data/sample.mp4 --config config.yaml --skip-train
```

### Options

| Flag | Description |
|---|---|
| `--video PATH` | Input video |
| `--config PATH` | YAML config (default `config.yaml`) |
| `--skip-train` | Skip Roboflow fine-tuning, use existing/pretrained weights |

To fine-tune on a Roboflow dataset, set `ROBOFLOW_API_KEY` in `.env` and configure `training.dataset_slug` in `config.yaml`.

## Outputs

After running, check `outputs/`:

- **annotated.mp4** — bounding boxes, track IDs, length/weight per fish, HUD overlay
- **metrics.md** — mAP50, mAP50-95, ID switches, track length, FPS
- **timeseries.csv** — frame-by-frame fish count, biomass, feeding score
- **plot.png** — biomass and feeding score over time

## Results

![Results Plot](outputs/plot.png)

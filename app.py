from pathlib import Path
import shutil
import uuid
import time
import threading

from flask import Flask, request, jsonify, render_template, send_from_directory

from run import load_config, run as run_pipeline

APP_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = APP_DIR / "uploads"
RUNS_DIR = APP_DIR / "runs"
UPLOAD_DIR.mkdir(exist_ok=True)
RUNS_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200MB

jobs = {}  # job_id -> status dict


def worker(job_id, video_path, cfg_path):
    try:
        jobs[job_id]["status"] = "running"
        jobs[job_id]["log"] = "Pipeline started..."
        cfg = load_config(str(cfg_path))
        cfg["paths"]["output_dir"] = str(Path(cfg_path).parent / "outputs")
        run_pipeline(cfg, str(video_path), skip_train=True)
        jobs[job_id]["status"] = "done"
        jobs[job_id]["log"] = "Finished."
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["log"] = f"Error: {e}"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "no file"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "empty filename"}), 400

    job_id = uuid.uuid4().hex[:8]
    run_dir = RUNS_DIR / job_id
    run_dir.mkdir()

    video_path = run_dir / "input.mp4"
    f.save(str(video_path))

    cfg_path = run_dir / "config.yaml"
    shutil.copy(APP_DIR / "config.yaml", cfg_path)

    jobs[job_id] = {
        "status": "queued",
        "log": "Queued.",
        "dir": str(run_dir),
        "outputs": [],
    }
    threading.Thread(
        target=worker, args=(job_id, video_path, cfg_path), daemon=True
    ).start()
    return jsonify({"job_id": job_id})


@app.route("/api/status/<job_id>")
def status(job_id):
    if job_id not in jobs:
        return jsonify({"error": "not found"}), 404
    j = jobs[job_id]
    out_dir = Path(j["dir"]) / "outputs"
    outputs = [
        n for n in ("annotated.mp4", "metrics.md", "timeseries.csv", "plot.png")
        if (out_dir / n).exists()
    ]
    return jsonify({
        "status": j["status"],
        "log": j["log"],
        "outputs": outputs,
    })


@app.route("/api/file/<job_id>/<filename>")
def file(job_id, filename):
    j = jobs.get(job_id)
    if not j:
        return "not found", 404
    out_dir = Path(j["dir"]) / "outputs"
    safe = Path(filename).name
    if safe not in ("annotated.mp4", "metrics.md", "timeseries.csv", "plot.png"):
        return "forbidden", 403
    return send_from_directory(str(out_dir), safe)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)

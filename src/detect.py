from pathlib import Path
import yaml
from dotenv import load_dotenv
from ultralytics import YOLO
import os

load_dotenv()


def download_roboflow_dataset(slug: str, dest: str = "data/dataset") -> str:
    from roboflow import Roboflow
    api_key = os.environ["ROBOFLOW_API_KEY"]
    rf = Roboflow(api_key=api_key)
    parts = slug.split("/")
    ws, proj = parts[0], parts[1]
    version = int(parts[2]) if len(parts) > 2 else None
    project = rf.workspace(ws).project(proj)
    if version is not None:
        ds = project.version(version).download("yolov8")
    else:
        ds = project.version(1).download("yolov8")
    return ds.location


def train_model(cfg: dict) -> str:
    ds_slug = cfg["training"]["dataset_slug"]
    if not ds_slug:
        raise ValueError("training.dataset_slug must be set in config.yaml")
    ds_path = download_roboflow_dataset(ds_slug)
    model = YOLO(cfg["model"]["name"])
    results = model.train(
        data=str(Path(ds_path) / "data.yaml"),
        epochs=cfg["training"]["epochs"],
        batch=cfg["training"]["batch"],
        imgsz=cfg["model"]["imgsz"],
        project="runs/detect",
        name="train",
    )
    return str(Path("runs/detect/train/weights/best.pt"))


def fish_class_ids(model, cfg):
    """Only explicitly named fish classes may contribute to fish estimates."""
    labels = {str(name).strip().casefold() for name in cfg["model"].get("fish_classes", ["fish"])}
    names = model.names
    items = names.items() if isinstance(names, dict) else enumerate(names)
    ids = [int(i) for i, name in items if str(name).strip().casefold() in labels]
    if not ids:
        raise ValueError(
            "Fish detection is unavailable: this model has no configured fish classes. "
            "Install fish-trained weights at paths.model_weights and set model.fish_classes "
            "to their fish/species labels in config.yaml. General object detections cannot estimate fish."
        )
    return ids


def get_model(cfg: dict) -> YOLO:
    weights = Path(cfg["paths"]["model_weights"])
    if not weights.is_absolute():
        weights = Path(__file__).resolve().parents[1] / weights
    model = YOLO(str(weights)) if weights.exists() else YOLO(cfg["model"]["name"])
    fish_class_ids(model, cfg)
    return model


def validate(model: YOLO, data_yaml: str = None) -> dict:
    if data_yaml and Path(data_yaml).exists():
        metrics = model.val(data=data_yaml)
        return {
            "mAP50": float(metrics.box.map50),
            "mAP50-95": float(metrics.box.map),
        }
    return {"mAP50": None, "mAP50-95": None}


def has_val_data(data_yaml: str = None) -> bool:
    return bool(data_yaml) and Path(data_yaml).exists()

"""Train a fish-only baseline; keep source splits and save held-out evaluation."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import yaml

ROOT = Path(__file__).resolve().parent
SOURCES = ['underwater-fish-v6', 'pond-species-v1']


def prepare():
    dest = ROOT / 'data' / 'fish-combined'
    dest.mkdir(exist_ok=True)
    seen = set()
    counts = {}
    # Held-out images take precedence if a source duplicated an image across splits.
    for split in ('test', 'valid', 'train'):
        images = dest / split / 'images'
        labels = dest / split / 'labels'
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        count = 0
        for source in SOURCES:
            base = ROOT / 'data' / source
            metadata = yaml.safe_load((base / 'data.yaml').read_text())
            names = metadata['names']
            for image in sorted((base / split / 'images').glob('*')):
                if image.suffix.lower() not in ('.jpg', '.jpeg', '.png'): continue
                label = base / split / 'labels' / (image.stem + '.txt')
                if not label.exists():
                    raise ValueError(f'Missing annotation: {label.name}')
                digest = hashlib.sha256(image.read_bytes()).hexdigest()
                if digest in seen: continue
                seen.add(digest)
                rows = []
                for line in label.read_text().splitlines():
                    values = line.split()
                    if len(values) != 5: raise ValueError('Expected detection boxes')
                    cid = int(values[0])
                    if cid < 0 or cid >= len(names): raise ValueError('Invalid class ID')
                    coords = [float(v) for v in values[1:]]
                    if not all(0 <= v <= 1 for v in coords) or min(coords[2:]) <= 0:
                        raise ValueError('Invalid bounding box')
                    rows.append('0 ' + ' '.join(values[1:]))
                filename = source + '_' + image.name
                shutil.copy2(image, images / filename)
                (labels / (Path(filename).stem + '.txt')).write_text('\n'.join(rows))
                count += 1
        counts[split] = count
        if not count: raise ValueError(f'Empty {split} split')
    config = {'path': str(dest), 'train': 'train/images', 'val': 'valid/images',
              'test': 'test/images', 'names': {0: 'fish'}, 'nc': 1}
    path = dest / 'data.yaml'
    path.write_text(yaml.safe_dump(config))
    (dest / 'audit.json').write_text(json.dumps({'counts': counts, 'sources': SOURCES,
        'classes': 'All source species merged to fish; no species identification',
        'deduplication': 'Exact file hashes only; near-duplicate frames may remain'}, indent=2))
    print('Dataset prepared:', counts, flush=True)
    return path


def main():
    os.chdir(ROOT)
    import torch
    from ultralytics import YOLO
    torch.set_num_threads(4)
    data = prepare()
    model = YOLO(str(ROOT / 'yolo11n.pt'))
    model.train(data=str(data), epochs=20, patience=5, imgsz=416, batch=4,
                device='cpu', workers=0, freeze=10, cache=False, seed=42,
                project=str(ROOT / 'runs' / 'detect'), name='fish_baseline',
                exist_ok=False, plots=True, save=True)
    best = Path(model.trainer.best)
    trained = YOLO(str(best))
    metrics = trained.val(data=str(data), split='test', imgsz=416, batch=4,
                          device='cpu', workers=0, plots=True)
    summary = {'weights': str(best), 'test_mAP50': float(metrics.box.map50),
               'test_mAP50_95': float(metrics.box.map),
               'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
               'note': 'Baseline only. Not validated on Nigerian pond footage or for biomass.'}
    (best.parent.parent / 'evaluation.json').write_text(json.dumps(summary, indent=2))
    print('TRAINING COMPLETE', json.dumps(summary), flush=True)
    # Keep a completed candidate separate until its pond performance is reviewed.

if __name__ == '__main__':
    main()

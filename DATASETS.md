# Fish baseline training

Sources:
- UnderWater Fish Detection v6, underwater fish, CC BY 4.0: https://universe.roboflow.com/underwater-fish/underwater-fish-detection-izi1l/dataset/6
- Fish Detection v1, Travaux Professionnel, Public Domain: https://universe.roboflow.com/travaux-professionnel/fish-detection-taxj6-tnm2l/dataset/1

The underwater set supplies submerged fish views. The supplementary set includes Carp, CatFish, Mullet and Tilapia, including photographs out of water. Neither establishes accuracy on Nigerian pond footage. These are initial candidates, not a proven best dataset.

All species labels are merged into one fish class. Source splits are preserved and exact file duplicates excluded; related video frames and augmented copies may still inflate evaluation scores. Independently captured pond footage is needed for a reliable evaluation.

Run `python train_fish.py` with both downloaded datasets in data/. Training uses YOLO11n, 20 epochs maximum with early stopping, 416px images and a frozen backbone for a CPU baseline. Candidate weights and held-out test metrics are saved under runs/detect/. Existing app weights are not overwritten automatically. Datasets and weights stay local and are ignored by Git.

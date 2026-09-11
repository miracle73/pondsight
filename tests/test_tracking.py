import unittest
from types import SimpleNamespace
import numpy as np
from src.track import Tracker
from src.biomass import BiomassEstimator
from src.detect import fish_class_ids

class Tensor:
    def __init__(self, data): self.data = np.array(data)
    def cpu(self): return self
    def numpy(self): return self.data

def result(ids, boxes):
    return SimpleNamespace(boxes=SimpleNamespace(id=Tensor(ids), xyxy=Tensor(boxes)))

class TrackingTests(unittest.TestCase):
    def test_disappearing_and_returning_fish(self):
        tracker = Tracker()
        tracker.update(result([1, 2], [[0,0,3,4], [0,0,6,8]]))
        both = BiomassEstimator().frame_biomass([tracker.tracks[i] for i in tracker.active_ids])
        tracker.update(result([2], [[0,0,6,8]]))
        self.assertEqual(tracker.active_ids, [2])
        self.assertLess(BiomassEstimator().frame_biomass([tracker.tracks[2]]), both)
        tracker.update(SimpleNamespace(boxes=SimpleNamespace(id=None)))
        self.assertEqual(tracker.active_ids, [])
        tracker.update(result([1], [[0,0,3,4]]))
        self.assertEqual(tracker.active_ids, [1])
        self.assertEqual(len(tracker.tracks), 2)
        self.assertIsNone(tracker.total_id_switches())

    def test_current_smoothed_length(self):
        tracker = Tracker(smooth_window=1)
        tracker.update(result([1], [[0,0,3,4]]))
        tracker.update(result([1], [[0,0,6,8]]))
        self.assertEqual(tracker.tracks[1].mean_length_px, 10)

    def test_fish_classes_exclude_other_objects(self):
        cfg = {'model': {'fish_classes': ['fish', 'catfish']}}
        self.assertEqual(fish_class_ids(SimpleNamespace(names={0:'person',1:'fish',2:'catfish'}), cfg), [1,2])
        with self.assertRaisesRegex(ValueError, 'Fish detection is unavailable'):
            fish_class_ids(SimpleNamespace(names={0:'person',1:'car'}), cfg)

if __name__ == '__main__': unittest.main()

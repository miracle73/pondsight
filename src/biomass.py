from dataclasses import dataclass
from src.track import Track


@dataclass
class BiomassEstimator:
    px_per_cm: float = 12.5
    a: float = 0.0089
    b: float = 3.05

    def length_cm(self, length_px: float) -> float:
        return length_px / self.px_per_cm

    def weight_kg(self, length_cm: float) -> float:
        grams = self.a * (length_cm ** self.b)
        return grams / 1000.0

    def per_track_kg(self, track: Track) -> float:
        cm = self.length_cm(track.mean_length_px)
        return self.weight_kg(cm)

    def frame_biomass(self, active_tracks: list) -> float:
        return sum(self.per_track_kg(t) for t in active_tracks)

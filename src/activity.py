from collections import deque
from typing import List, Tuple
import numpy as np
import cv2


class FeedingActivity:
    def __init__(self, smooth_window: int = 15, flow_scale: float = 0.5):
        self.smooth_window = smooth_window
        self.flow_scale = flow_scale
        self.prev_gray = None
        self.history: deque = deque(maxlen=smooth_window)

    def _flow_magnitude(self, gray: np.ndarray, boxes) -> float:
        if self.prev_gray is None or boxes is None or len(boxes) == 0:
            return 0.0
        h, w = gray.shape[:2]
        mag_sum, pix_count = 0.0, 0
        for box in boxes:
            x1, y1, x2, y2 = [int(v) for v in box[:4]]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            roi_prev = self.prev_gray[y1:y2, x1:x2]
            roi_curr = gray[y1:y2, x1:x2]
            roi_prev_r = cv2.resize(roi_prev, None, fx=self.flow_scale, fy=self.flow_scale)
            roi_curr_r = cv2.resize(roi_curr, None, fx=self.flow_scale, fy=self.flow_scale)
            flow = cv2.calcOpticalFlowFarneback(
                roi_prev_r, roi_curr_r, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
            mag_sum += float(mag.mean())
            pix_count += 1
        return mag_sum / pix_count if pix_count else 0.0

    def update(self, gray: np.ndarray, boxes) -> float:
        score = self._flow_magnitude(gray, boxes)
        self.history.append(score)
        self.prev_gray = gray.copy()
        if not self.history:
            return 0.0
        arr = np.array(self.history)
        normed = arr / (arr.max() + 1e-6)
        return float(normed[-1])

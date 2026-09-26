"""Causal motion baseline for Part B. Scores are NOT calibrated probabilities.

Only current/past detections are used. Image-plane footprints approximate road
contact; perspective, occlusion and tracking errors limit collision prediction.
No Part A cache, video filename, sample labels or future registration is read.
"""
from collections import deque
from itertools import combinations
import math
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CLASSES = [0, 1, 2, 3, 5, 7]


def contact(box, person=False):
    x1, y1, x2, y2 = box
    point = np.array([(x1 + x2) / 2, y2], dtype=float)
    radius = np.maximum([2., 2.], [(x2-x1) * (.25 if person else .45),
                                  (y2-y1) * (.06 if person else .12)])
    return point, radius


def time_to_contact(offset, velocity, radius, horizon=5.):
    """Time until two ground-footprint ellipses overlap under constant velocity.

    Current overlap cannot establish impending collision (often occlusion).
    Relative position/velocity cancels common camera translation.
    """
    p, v = np.asarray(offset) / radius, np.asarray(velocity) / radius
    a, b, c = float(v @ v), float(2 * p @ v), float(p @ p - 1)
    if a < .04 or b >= 0 or c <= 0:
        return None
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return None
    entry = (-b - math.sqrt(discriminant)) / (2*a)
    return entry if 0 < entry <= horizon else None


class MotionRisk:
    """Small bounded history; all evidence ends at the supplied timestamp."""
    def __init__(self):
        self.histories = {}
        self.pair_hits = {}
        self.last_time = None
        self.score = 0.
        self.evidence = None

    def update(self, t, detections):
        if not math.isfinite(t) or (self.last_time is not None and t <= self.last_time):
            raise ValueError('Risk timestamps must be finite and strictly increasing')
        dt = .5 if self.last_time is None else t - self.last_time
        self.last_time = t
        self.histories = {k: h for k, h in self.histories.items() if t-h[-1][0] <= 1.1}
        fitted = []
        for item in detections:
            key = (item['class'], item['id'])
            box = np.asarray(item['bbox'], dtype=float)
            if (item['class'] not in CLASSES or not np.isfinite(box).all()
                    or box[2] <= box[0] or box[3] <= box[1]):
                continue
            point, radius = contact(box, item['class'] == 0)
            history = self.histories.setdefault(key, deque(maxlen=6))
            # Reset implausible jumps; avoid turning an ID switch into a crash.
            if history and np.linalg.norm((point-history[-1][1])/radius) > 8*(t-history[-1][0]):
                history.clear()
            history.append((t, point, radius, float(item['score'])))
            while history and t-history[0][0] > 1.6:
                history.popleft()
            if len(history) < 3 or t-history[0][0] < .75:
                continue
            times = np.array([x[0]-t for x in history])
            positions = np.array([x[1] for x in history])
            design = np.column_stack([times, np.ones(len(times))])
            coefficients = np.linalg.lstsq(design, positions, rcond=None)[0]
            residual = np.sqrt(np.mean(np.sum(((positions-design@coefficients)/radius)**2, axis=1)))
            quality = math.exp(-2*residual*residual)
            if quality < .5:
                continue
            fitted.append((key, point, radius, coefficients[0], quality,
                           min(x[3] for x in history)))
        active_pairs = {}
        raw, evidence = 0., None
        for a, b in combinations(fitted, 2):
            if a[0][0] == b[0][0] == 0:
                continue
            # Similar ground position of person and rider/vehicle can be an
            # occlusion; time_to_contact already rejects current overlap.
            ttc = time_to_contact(b[1]-a[1], b[3]-a[3], a[2]+b[2])
            if ttc is None:
                continue
            pair = tuple(sorted([a[0], b[0]]))
            count = self.pair_hits.get(pair, 0) + 1 if dt <= 1.1 else 1
            active_pairs[pair] = count
            reliability = min(a[4], b[4]) * min(1., min(a[5], b[5])/.5)
            candidate = (.6 + .35*(1-ttc/5)) * reliability
            # A single noisy observation must not create an alarm.
            if count < 2:
                candidate = min(candidate, .35)
            if candidate > raw:
                raw = candidate
                evidence = {'pair': pair, 'ttc_sec': round(ttc, 3),
                            'consecutive_samples': count, 'raw_score': round(raw, 4)}
        self.pair_hits = active_pairs
        tau = .25 if raw > self.score else .6
        self.score += (raw-self.score) * (1-math.exp(-dt/tau))
        self.evidence = evidence
        return float(np.clip(self.score, 0, 1))


class CausalRiskEstimator:
    """YOLO26x + ByteTrack at 2 Hz; returns a held score on intervening frames."""
    def __init__(self):
        self.model = None

    def reset(self, meta):
        self.meta = dict(meta)
        self.motion = MotionRisk()
        self.last_inference = -float('inf')
        self.last_input = -float('inf')
        self.last_score = 0.
        # Tracker state must never cross video boundaries. Reuse only weights.
        if self.model is not None:
            for tracker in getattr(self.model.predictor, 'trackers', []):
                tracker.reset()

    def _load_model(self):
        weights = ROOT / 'weights/yolo26x.pt'
        if not weights.is_file():
            raise FileNotFoundError(f'Missing offline weights: {weights}')
        os.environ.setdefault('YOLO_CONFIG_DIR', str(ROOT / 'debug'))
        Path(os.environ['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
        import torch
        from ultralytics import YOLO
        if not torch.cuda.is_available():
            raise RuntimeError('Part B requires CUDA for local YOLO26x inference')
        print(f'Part B: YOLO26x, 2 Hz, GPU {torch.cuda.get_device_name(0)}; '
              f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}', flush=True)
        self.model = YOLO(str(weights))

    def step(self, frame, t_sec):
        t = float(t_sec)
        if not math.isfinite(t) or t < self.last_input:
            raise ValueError('Frames must arrive in timestamp order')
        self.last_input = t
        if t-self.last_inference < .5-1e-6:
            return self.last_score
        if self.model is None:
            self._load_model()
        result = self.model.track(frame, persist=True, tracker='bytetrack.yaml',
                                  device=0, imgsz=2560, quantize=16, conf=.25,
                                  classes=CLASSES, verbose=False)[0]
        from tools.auto_label_video import get_detections, person_inside_vehicle
        h, w = frame.shape[:2]
        items = get_detections(result, w, h)
        items = [item for item in items
                 if item['bbox'][3] < h-3 and not (
                     item['class'] == 0 and person_inside_vehicle(item, items))]
        self.last_score = self.motion.update(t, items)
        self.last_inference = t
        return self.last_score

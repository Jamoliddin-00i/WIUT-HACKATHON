"""Offline event inference from a new recording of the reference camera.

Inference never opens manual labels, sample event predictions, cached tracks,
or video-specific signal timelines. Scene alignment comes from decoded frames.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


class SceneRegistration:
    def __init__(self):
        self.early = None
        self.frames = []
        self.last_time = -float('inf')

    def observe(self, frame, t_sec):
        # Keep a fallback for short recordings; later frames avoid initial shake.
        if self.early is None:
            self.early = cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA)
        if 30 <= t_sec <= 65 and t_sec - self.last_time >= 10 and len(self.frames) < 4:
            self.frames.append(cv2.resize(frame, (1920, 1080), interpolation=cv2.INTER_AREA))
            self.last_time = t_sec

    def scene(self):
        from tools.scene.register import register_frames
        from tools.auto_label_video import scene_from_zones

        frames = self.frames or ([self.early] if self.early is not None else [])
        if not frames:
            raise RuntimeError('No decoded frames available for scene registration')
        registration, _ = register_frames(frames)
        matrix = np.asarray(registration['H_ref_to_video'])
        if (not np.isfinite(matrix).all() or registration['inliers_h'] < 30
                or registration['inlier_spread'] < .25
                or registration['resid_h_px']['p90'] > 10):
            raise RuntimeError('Camera registration unreliable; refusing to apply misaligned road zones')
        print(f"Scene alignment: {registration['inliers_h']} inliers, "
              f"p90 error {registration['resid_h_px']['p90']:.1f}px", flush=True)
        zones = json.loads((ROOT / 'reports/eda/zones.json').read_text(encoding='utf-8'))
        return scene_from_zones(zones, '__runtime__', {
            'videos': {'__runtime__': {'homography': registration}}})


def detect_events(video_path):
    # Configure Ultralytics before importing it. No implicit weight downloads.
    os.environ.setdefault('YOLO_CONFIG_DIR', str(Path(tempfile.gettempdir()) / 'salen-yolo'))
    from tools.auto_label_video import propose

    weights = ROOT / 'weights/yolo26x.pt'
    if not weights.is_file():
        raise FileNotFoundError(f'Missing offline model: {weights}. Run python tools/download_weights.py before submission.')
    registration = SceneRegistration()
    work = os.environ.get('SALEN_WORK_DIR')
    if work:
        Path(work).mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='salen-tracks-', dir=work) as scratch:
        result = propose(Path(video_path), registration.scene, weights,
                         sample_fps=2.0, image_size=2560, max_seconds=None,
                         tracks_dir=Path(scratch), reuse_tracks=False,
                         frame_observer=registration.observe)
    return result['events']

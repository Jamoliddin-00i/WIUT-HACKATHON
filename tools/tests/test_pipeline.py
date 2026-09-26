import unittest
from unittest.mock import patch
import numpy as np

from src.pipeline import SceneRegistration
import solution


class PipelineTests(unittest.TestCase):
    def test_public_entry_point_returns_detector_events(self):
        expected = [[1, 3, 'jaywalking']]
        with patch('src.pipeline.detect_events', return_value=expected) as runner:
            self.assertEqual(solution.detect_events('unseen-camera-recording.MP4'), expected)
            runner.assert_called_once_with('unseen-camera-recording.MP4')

    def test_registration_uses_frames_and_rejects_low_quality(self):
        registration = SceneRegistration()
        registration.observe(np.zeros((108, 192, 3), np.uint8), 0)
        weak = {'H_ref_to_video': np.eye(3).tolist(), 'inliers_h': 5,
                'inlier_spread': .1, 'resid_h_px': {'p90': 2}}
        with patch('tools.scene.register.register_frames', return_value=(weak, None)):
            with self.assertRaisesRegex(RuntimeError, 'registration unreliable'):
                registration.scene()

    def test_short_video_has_registration_fallback(self):
        registration = SceneRegistration()
        registration.observe(np.zeros((108, 192, 3), np.uint8), 0)
        registration.observe(np.zeros((108, 192, 3), np.uint8), 1)
        self.assertEqual(len(registration.frames), 0)
        self.assertIsNotNone(registration.early)


if __name__ == '__main__':
    unittest.main()

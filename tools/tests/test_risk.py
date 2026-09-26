import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from src.risk import CausalRiskEstimator, MotionRisk, time_to_contact


def car(identity, x, y=100):
    return {'id': identity, 'class': 2, 'bbox': [x-10, y-20, x+10, y], 'score': .9}


class RiskTests(unittest.TestCase):
    def test_contact_is_in_future_and_within_horizon(self):
        self.assertAlmostEqual(time_to_contact(np.array([100., 0]), np.array([-20., 0]), np.array([20., 5.])), 4.)
        self.assertIsNone(time_to_contact(np.array([100., 0]), np.array([20., 0]), np.array([20., 5.])))
        self.assertIsNone(time_to_contact(np.array([100., 0]), np.array([-2., 0]), np.array([20., 5.])))

    def test_collision_course_alarms_before_contact(self):
        risk = MotionRisk()
        scores = [risk.update(t, [car(1, 10*t), car(2, 100-10*t)]) for t in np.arange(0, 3, .5)]
        self.assertGreater(scores[-1], .5)
        self.assertLess(max(scores[:3]), .5)
        self.assertGreater(risk.evidence['ttc_sec'], 0)

    def test_parallel_and_separating_traffic_stay_low(self):
        for mode in ['parallel', 'separating', 'adjacent_lane']:
            risk = MotionRisk()
            for t in np.arange(0, 3, .5):
                second = car(2, 100+10*t) if mode != 'adjacent_lane' else car(2, 100-10*t, 130)
                first = car(1, -10*t if mode == 'separating' else 10*t)
                self.assertLess(risk.update(t, [first, second]), .5)

    def test_missing_objects_decay_and_expire(self):
        risk = MotionRisk()
        for t in np.arange(0, 3, .5):
            risk.update(t, [car(1, 10*t), car(2, 100-10*t)])
        self.assertGreater(risk.score, .5)
        self.assertLess(risk.update(4., []), .1)
        self.assertEqual(risk.histories, {})

    def test_track_jump_cannot_create_alarm(self):
        risk = MotionRisk()
        for t in [0, .5, 1]:
            risk.update(t, [car(1, 0), car(2, 500)])
        self.assertLess(risk.update(1.5, [car(1, 0), car(2, 50)]), .5)

    def test_prefix_is_independent_of_later_detections(self):
        common = [[car(1, 10*t), car(2, 100-10*t)] for t in np.arange(0, 2, .5)]
        outputs = []
        for suffix in [[car(1, 300)], [car(1, 0), car(2, 5)]]:
            risk = MotionRisk()
            outputs.append([risk.update(i*.5, items) for i, items in enumerate(common+[suffix])])
        self.assertEqual(outputs[0][:-1], outputs[1][:-1])

    def test_sampling_and_reset(self):
        risk = CausalRiskEstimator()
        risk.model = MagicMock()
        risk.reset({'video_id': 'first'})
        frame = np.zeros((100, 100, 3), dtype=np.uint8)
        with patch('tools.auto_label_video.get_detections', return_value=[]):
            for t in [0, .1, .2, .5, .6, 1.]:
                self.assertEqual(risk.step(frame, t), 0.)
        self.assertEqual(risk.model.track.call_count, 3)
        risk.motion.score = .9
        risk.reset({'video_id': 'second'})
        self.assertEqual(risk.motion.score, 0.)
        self.assertEqual(risk.motion.histories, {})


if __name__ == '__main__':
    unittest.main()

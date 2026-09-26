import unittest
import numpy as np

from tools.pedestrian_rules import follows_edge, reliable_foot, road_outline, trajectory_intervals


class PedestrianRulesTests(unittest.TestCase):
    def test_internal_lane_boundary_is_not_a_kerb(self):
        lanes = [[[0, 0], [.5, 0], [.5, 1], [0, 1]],
                 [[.5, 0], [1, 0], [1, 1], [.5, 1]]]
        edges = road_outline(lanes, 200, 200)
        self.assertFalse(follows_edge((100, 100), (0, 20), edges, 10))
        self.assertTrue(follows_edge((5, 100), (0, 20), edges, 10))

    def test_edge_allowance_requires_parallel_motion_and_proximity(self):
        crossing = [np.asarray([[0, 0], [200, 0], [200, 40], [0, 40]], np.float32)]
        self.assertTrue(follows_edge((100, 45), (30, 0), crossing, 10))
        self.assertTrue(follows_edge((100, 45), (-30, 0), crossing, 10))
        self.assertFalse(follows_edge((100, 45), (0, 30), crossing, 10))
        self.assertFalse(follows_edge((100, 60), (30, 0), crossing, 10))
        self.assertFalse(follows_edge((100, 45), (0, 0), crossing, 10))

    def test_clipped_box_has_no_trustworthy_foot(self):
        self.assertTrue(reliable_foot((100, 100, 150, 300), 1920, 1080))
        self.assertFalse(reliable_foot((100, 900, 150, 1080), 1920, 1080))
        self.assertFalse(reliable_foot((0, 100, 50, 300), 1920, 1080))

    def test_pause_inside_road_episode_keeps_event(self):
        samples = [(i * .5, (x, 100), 100, True)
                   for i, x in enumerate([0, 20, 40, 40, 40, 60])]
        self.assertEqual(trajectory_intervals(samples, .5, 10), [[0, 3]])

    def test_stationary_box_jitter_does_not_make_an_event(self):
        samples = [(i * .5, (x, 100), 100, True)
                   for i, x in enumerate([100, 103, 97, 101, 98, 102])]
        self.assertEqual(trajectory_intervals(samples, .5, 10), [])

    def test_earlier_motion_does_not_extend_stationary_tail(self):
        samples = [(i * .5, (x, 100), 100, True)
                   for i, x in enumerate([0, 20, 40] + [40] * 20)]
        intervals = trajectory_intervals(samples, .5, 20)
        self.assertEqual(len(intervals), 1)
        self.assertLessEqual(intervals[0][1], 2.5)

    def test_observed_return_to_zebra_splits_event(self):
        samples = [(i * .5, (i * 20, 100), 100, i != 3) for i in range(7)]
        self.assertEqual(trajectory_intervals(samples, .5, 10), [[0, 1.5], [2, 3.5]])

    def test_id_jump_does_not_join_unrelated_people(self):
        samples = [(i * .5, (x, 100), 100, True)
                   for i, x in enumerate([0, 20, 40, 700, 720, 740])]
        self.assertEqual(trajectory_intervals(samples, .5, 10), [[0, 1.5], [1.5, 3]])

    def test_missing_frames_do_not_extend_beyond_video(self):
        samples = [(0, (0, 100), 100, True), (.5, (20, 100), 100, True),
                   (1.5, (60, 100), 100, True)]
        self.assertEqual(trajectory_intervals(samples, .5, 1.7), [[0, 1.7]])


if __name__ == '__main__':
    unittest.main()

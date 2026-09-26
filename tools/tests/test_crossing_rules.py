import unittest
from tools.crossing_rules import CrossingEvents


class CrossingTests(unittest.TestCase):
    def run_scene(self, person_x=35, stationary=False, excluded=False):
        crossings = [[[0, .45], [1, .45], [1, .55], [0, .55]]]
        roads = [[[0, 0], [.5, 0], [.5, 1], [0, 1]],
                 [[.5, 0], [1, 0], [1, 1], [.5, 1]]]
        detector = CrossingEvents(crossings, roads, 100, 100, .5, 2.5)
        for i, bottom in enumerate([42, 47, 52, 57, 62]):
            if stationary:
                bottom = 52
            detector.observe(i*.5, [
                {'id': 1, 'class': 2, 'bbox': [10, bottom-20, 30, bottom]},
                {'id': 2, 'class': 0, 'bbox': [person_x-3, 30, person_x+3, 50]},
            ], {2} if excluded else set())
        return detector.events()

    def test_event_covers_entry_to_exit(self):
        self.assertEqual(self.run_scene(), [[.5, 2.0]])

    def test_other_carriageway_does_not_trigger(self):
        self.assertEqual(self.run_scene(person_x=80), [])

    def test_stationary_vehicle_does_not_trigger(self):
        self.assertEqual(self.run_scene(stationary=True), [])

    def test_occupants_do_not_trigger(self):
        self.assertEqual(self.run_scene(excluded=True), [])


if __name__ == '__main__':
    unittest.main()

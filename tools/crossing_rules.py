"""Associate moving vehicles with pedestrians on the same crossing road section."""
from collections import defaultdict
import math

import cv2
import numpy as np


def contains(point, polygon):
    return cv2.pointPolygonTest(polygon, tuple(map(float, point)), False) >= 0


def vehicle_ground_points(box):
    x1, y1, x2, y2 = box
    # Contact region near the wheels; the roof may visually cover a crossing
    # while the vehicle is still behind it.
    y = y2 - .2 * (y2 - y1)
    return [(x1, y), (x2, y), (x2, y2), (x1, y2), ((x1 + x2) / 2, (y + y2) / 2)]


class CrossingEvents:
    def __init__(self, crosswalks, roads, width, height, period, duration):
        self.crossings = [np.asarray(p, np.float32) * [width, height] for p in crosswalks]
        self.crossings = [p.astype(np.float32) for p in self.crossings]
        self.roads = [(np.asarray(p) * [width, height]).astype(np.float32) for p in roads]
        self.width, self.height = width, height
        self.period, self.duration = period, duration
        self.samples = defaultdict(list)
        self.previous = {}

    def observe(self, t, items, excluded_people):
        people = []
        for p in items:
            if p['class'] != 0 or p['id'] in excluded_people:
                continue
            x1, y1, x2, y2 = p['bbox']
            if y2 >= self.height - 3:
                continue
            foot = ((x1+x2)/2, y2)
            crossing_ids = {i for i, poly in enumerate(self.crossings) if contains(foot, poly)}
            road_ids = {i for i, poly in enumerate(self.roads) if contains(foot, poly)}
            if crossing_ids and road_ids:
                people.append((foot, crossing_ids, road_ids))
        for v in items:
            if v['class'] not in {1, 2, 3, 5, 7}:
                continue
            key = (v['class'], v['id'])
            box = v['bbox']
            foot = ((box[0]+box[2])/2, box[3])
            extent = max(box[2]-box[0], box[3]-box[1])
            previous = self.previous.get(key)
            self.previous[key] = (t, foot)
            moving = False
            if previous and 0 < t-previous[0] <= 1.1:
                speed = math.dist(foot, previous[1]) / (t-previous[0])
                moving = .15 * extent <= speed <= 4 * extent
            points = vehicle_ground_points(box)
            road_ids = {i for i, poly in enumerate(self.roads) if contains(foot, poly)}
            for ci, polygon in enumerate(self.crossings):
                on_crossing = any(contains(p, polygon) for p in points)
                conflict = moving and on_crossing and any(
                    ci in crosses and bool(road_ids & roads) and math.dist(foot, p) <= 1.5*extent
                    for p, crosses, roads in people)
                self.samples[(key, ci)].append((t, on_crossing, conflict, foot, extent))

    def events(self):
        intervals = []
        for samples in self.samples.values():
            chunk = []
            for sample in samples + [(self.duration, False, False, (0, 0), 0)]:
                if chunk and (not sample[1] or sample[0]-chunk[-1][0] > 1.1):
                    duration = chunk[-1][0]-chunk[0][0]+self.period
                    travel = math.dist(chunk[0][3], chunk[-1][3])
                    extent = float(np.median([p[4] for p in chunk]))
                    if (any(p[2] for p in chunk) and len(chunk) >= 2
                            and duration <= 15 and travel >= .35*extent):
                        intervals.append([round(chunk[0][0], 3),
                                          round(min(self.duration, chunk[-1][0]+self.period), 3)])
                    chunk = []
                if sample[1]:
                    chunk.append(sample)
        return intervals

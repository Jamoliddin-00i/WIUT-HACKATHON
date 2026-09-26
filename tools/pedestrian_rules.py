"""Pixel-space pedestrian geometry and short trajectory checks.

These checks use video evidence only, never manual event times or video IDs.
"""
from __future__ import annotations

import math
import cv2
import numpy as np


def simplified_polygons(polygons, width, height):
    scale = np.asarray([width, height], np.float32)
    return [cv2.approxPolyDP(np.asarray(p, np.float32) * scale,
                            max(1.0, height / 720), True).reshape(-1, 2)
            for p in polygons]


def road_outline(polygons, width, height):
    """Use the road's outer boundary, not borders between adjacent lanes."""
    mask = np.zeros((height, width), np.uint8)
    for polygon in polygons:
        pixels = np.rint(np.asarray(polygon) * [width, height]).astype(np.int32)
        cv2.fillPoly(mask, [pixels], 255)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    return [cv2.approxPolyDP(c, max(1.0, height / 720), True).reshape(-1, 2)
            for c in contours if cv2.contourArea(c) > 4]


def follows_edge(point, motion, polygons, margin, min_alignment=0.866):
    """Near a polygon edge and moving within 30 degrees of its tangent.

    The allowance scales with apparent person height, rather than a fixed
    fraction of the image, so it is smaller for distant pedestrians.
    """
    point = np.asarray(point, np.float32)
    motion = np.asarray(motion, np.float32)
    length = float(np.linalg.norm(motion))
    if length < 1e-6:
        return False
    for polygon in polygons:
        edges = np.roll(polygon, -1, axis=0) - polygon
        lengths2 = np.sum(edges * edges, axis=1)
        valid = lengths2 > 1e-12
        if not np.any(valid):
            continue
        starts, edges, lengths2 = polygon[valid], edges[valid], lengths2[valid]
        fractions = np.clip(np.sum((point - starts) * edges, axis=1) / lengths2, 0, 1)
        distances = np.linalg.norm(point - (starts + fractions[:, None] * edges), axis=1)
        nearest = int(np.argmin(distances))
        alignment = abs(float(np.dot(motion, edges[nearest]))) / (length * math.sqrt(lengths2[nearest]))
        if distances[nearest] <= margin and alignment >= min_alignment:
            return True
    return False


def reliable_foot(box, width, height):
    """A box clipped at the bottom/side cannot supply a reliable foot point."""
    x1, y1, x2, y2 = box
    border = max(2.0, height / 720)
    return (x1 > border and x2 < width - border and y2 < height - border
            and x2 > x1 and y2 > y1)


def trajectory_intervals(samples, sample_period, duration):
    """Keep supported road episodes, including pauses within the episode.

    samples contain (time, foot_xy, person_height, spatially_eligible). Explicit
    ineligible observations end an episode; missing detections bridge at most
    one second. A large jump starts a new episode to avoid an ID switch joining
    unrelated pedestrians. Tiny jitter without meaningful motion is rejected.
    """
    # A single earlier movement must not keep a stationary false detection
    # active for the rest of the clip. Measure movement across a two-second
    # window; this tolerates short pauses and reduces frame-to-frame box jitter.
    supported = []
    left = right = 0
    for index, sample in enumerate(samples):
        t, point, height, eligible = sample
        while samples[left][0] < t - 1.01:
            left += 1
        right = max(right, index)
        while right + 1 < len(samples) and samples[right + 1][0] <= t + 1.01:
            right += 1
        elapsed = samples[right][0] - samples[left][0]
        movement = math.dist(samples[left][1], samples[right][1])
        moving = elapsed > 0 and movement / elapsed >= 0.1 * height
        supported.append((t, point, height, eligible and moving))
    chunks, chunk = [], []
    for sample in supported:
        t, point, height, eligible = sample
        if chunk:
            previous = chunk[-1]
            gap = t - previous[0]
            distance = math.dist(point, previous[1])
            jump = distance > max(height, previous[2]) * (0.5 + 1.5 * gap)
            if not eligible or gap > max(1.01, sample_period * 2.01) or jump:
                chunks.append(chunk)
                chunk = []
        if eligible:
            chunk.append(sample)
    if chunk:
        chunks.append(chunk)
    intervals = []
    for chunk in chunks:
        if len(chunk) < 3 or chunk[-1][0] - chunk[0][0] < 0.75:
            continue
        points = np.asarray([p[1] for p in chunk])
        extent = float(np.linalg.norm(np.ptp(points, axis=0)))
        person_height = float(np.median([p[2] for p in chunk]))
        if extent < 0.3 * person_height:
            continue
        intervals.append([round(chunk[0][0], 3),
                          round(min(duration, chunk[-1][0] + sample_period), 3)])
    return intervals

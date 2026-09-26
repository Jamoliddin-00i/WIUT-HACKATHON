# Jaywalking review against updated user observations

The updated manual event intervals are preserved exactly in
[`labels/user_truth_2026-09-26.json`](../labels/user_truth_2026-09-26.json).
They are the reference for review, not proof that every boundary or event is
final. The older observations remain in `labels/user_observations_2026-09-25.json`.

The automatic rule already used the **bottom center of the person detection
box** as a rough foot point. We inspected the actual detections in
[`jaywalking_false_candidates_review.jpg`](../../../videos/jaywalking_false_candidates_review.jpg)
and ran the official local YOLO26x pose model on representative frames in
[`pose_ankle_reliability_check.jpg`](../../../videos/pose_ankle_reliability_check.jpg).
`tools/check_ankle_pose.py` reproduces the latter check when the cached tracks,
videos, and `weights/yolo26x-pose.pt` are present.

| Case | Frame finding | Outcome |
| --- | --- | --- |
| C3905, 38 and 60 s | The person detection is actually a car occupant; its box bottom lands inside the vehicle. | Suppressed by same-frame vehicle enclosure. |
| C3896, 42, 55, 103, 140, 277, 328 s | Similar driver or passenger detections inside cars or trucks. | Suppressed by the same filter. |
| C3905, 34 s | The person is riding a motorcycle. The pose model finds two ankles, so ankle visibility alone cannot classify a pedestrian. | Still a false candidate. |
| C3905, 85 s | A person walks at the foreground crossing edge. Two ankles are visible; their mean lies about 0.010 normalized image units outside the mapped crossing. This is close to the drawn boundary. | Still a candidate; inspect the crossing geometry and the video before changing the margin. |
| C3905, 16, 76, 107 s; C3896, 170 s | Pose found both ankles on sampled people in the updated hand-labeled jaywalking intervals. | The vehicle filter retains these cases. |

The earlier occupant-filter replay had 97 **unverified** event proposals across
four clips. C3905 has seven jaywalking proposals; three overlap the user's three
intervals and four do not. C3896 has 14 jaywalking proposals; two overlap the
user's two intervals and 12 do not. The completed four-video labels now include
four jaywalking intervals each for C3897 and C3902. At temporal IoU 0.5, only
one interval matched in each of those clips. The current geometry and person/rider detection
still require work before these proposals can be treated as ground truth.

The subsequent trajectory checks reduce the output to 82 proposals and improve
jaywalking at temporal IoU 0.5 from 5 matches / 57 extras / 8 misses to
8 matches / 39 extras / 5 misses. See `AUTO_PROPOSALS.md` for current behavior
and reproducible evaluation commands. The frame findings above remain useful,
but their counts describe the earlier replay.

Looking at ankles is useful for a second-stage check, but the nine sampled
frames do not justify replacing the current rule with a pose rule. A rider can
have two confident ankle keypoints, and an ankle just outside a polygon can
be caused by the hand-drawn or mapped crossing boundary. The pose weight is
local and is **not** part of `auto_label_video.py` or the 97 proposal replay.

The person-in-vehicle filter is deliberately conservative: it requires at
least 85% of the person box inside a detected car, bus, or truck, with the
box bottom no lower than 75% of the vehicle's height. A pedestrian crossing
in front of a vehicle can overlap its box without being suppressed. Replay
the cached tracks with `--no-occupant-filter` to compare this gate. It uses
box geometry, not a retrained model or changed neural-network weights.

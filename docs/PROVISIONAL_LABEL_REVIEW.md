# Review of Jamoliddin's provisional observations

Reviewed 2026-09-25 against the original videos, frame sequences, the team
`LABELING_GUIDE.md`, and the current v4 head D intervals in
`reports/eda/signals/v4/`. Head D is a timing proxy for the near-carriageway
flow; its physical control of individual movements is unverified. The
submitted intervals are preserved unchanged in
`labels/user_observations_2026-09-25.json`. The observations have no actor IDs,
so the interpretation of each stopped vehicle still needs confirmation from
the annotator. This is a triage review, not verified ground truth or an
evaluator score. Exact boundaries need frame-by-frame review.

The stopped bus at the far-side stop appears to stand in a live lane, with no
separate pull-out bay visible in these frames. The organizer explicitly counts
a bus dwelling in a live lane for at least 10 s as `stopped_vehicle`.

## C3905.MP4

| User interval (s) | Review | Evidence and next check |
| --- | --- | --- |
| 1.268-24.191 `stopped_vehicle` | Likely valid if this means the green bus at the far-side stop. | That bus is already stationary at 0 s and remains at the stop through about 24 s, then leaves by 30 s. Start may need to be 0. A near-side red-light queue would not count. |
| 11.211-16.950 `jaywalking` | Plausible, timing needs review. | A dark-clothed pedestrian walks diagonally off the upper zebra toward the island around 12-17 s; others use the zebra, and the cyclist is a vehicle. Mark only the portion with both feet off the painted crossing. |
| 36.770-62.729 `stopped_vehicle` | Likely valid if this means the bus at the far-side stop. | A bus remains at the stop around 40-63 s. The preceding bus moves through the area. The start/end should follow this particular bus's stop and departure. |
| 72.139-79.746 `jaywalking` | Uncertain. | Most visible people use the lower zebra and islands. A few may step off the stripe edge. Identify the specific pedestrian(s) and label only their time on carriageway outside a crossing. |
| 72.139-111.612 `congestion` | Likely invalid under the organizer rule. | D changes from blinking green to amber at 72.472 s, red at 75.475 s, and does not turn green again until 114.514 s. This interval is almost entirely an ordinary red-light queue; it ends before the next green can test whether the queue fails to clear. |
| 75.609-92.192 `stopped_vehicle` | Likely valid if this means the articulated green bus at the far-side stop. | The bus is stationary at the stop through roughly 78-93 s while other far-side vehicles move. A near-side vehicle waiting at red would be excluded. |
| 81.448-84.017 `near_miss` | Plausible; preserve for review. | The annotator saw a pedestrian jump back from the moving white Cobalt at the right-hand zebra. The right-hand frame sequence shows the car and pedestrians converging. Confirm the evasive step frame by frame, then set start to the first evasive movement and end when they are clear. Mere closeness would not count. |
| 81.782-84.718 `failure_to_yield` | Likely valid, exact boundaries need review. | The white Cobalt moves through the right-hand zebra while pedestrians are entering/occupying its carriageway half. Start when its front enters the stripes; end when its rear leaves. The earlier review mistakenly inspected the central zebra, where vehicles were waiting. |
| 107.007-117.918 `stopped_vehicle` | Uncertain; likely excluded if this means the near-side bus or cars. | The far-side bus stop is empty. Near-side vehicles queue under red/red+amber and begin moving after green at 114.514 s. Ask which vehicle was intended. |

## C3896.MP4

| User interval (s) | Review | Evidence and next check |
| --- | --- | --- |
| 8.542-26.059 `stopped_vehicle` | Likely valid if this means the dark bus at the far-side stop. | It remains at the stop through at least 10-30 s as nearby vehicles move. D is red/red+amber until about 27.127 s, so a near-side queue would not count. |
| 53.820-71.705 `congestion` | Likely invalid. | Vehicles in the near carriageway move through the junction in the sampled frames at 54, 63, and 71 s. D is green or blinking green until 62.562 s, then unknown/amber/red; there is no visible all-lane standstill that persists through a green. |
| 80.414-98.832 `stopped_vehicle` | Likely valid if this means the green bus at the far-side stop. | The bus is stationary there from about 80 through 100 s while other far-side traffic moves. A near-side queue is excluded because D is red from 66.066 to 99.099 s. |

## Automatic proposal comparison

The first YOLO26x/ByteTrack pass proposed only `jaywalking` and
`failure_to_yield`: 10 candidates on C3905 and 33 on C3896. Replaying the
cached tracks with a narrow far-side bus-dwell rule now produces 13 and 38
candidates respectively. It still cannot propose congestion or near misses.
The best
same-class temporal IoU for C3905's first jaywalking observation is only 0.161
(`16.016-17.017` versus `11.211-16.950`); the second has no overlap. The
`failure_to_yield` observation overlaps an automatic `82.082-82.582` fragment
at IoU 0.170. The revised right-hand crossing review supports a real event,
but the automatic fragment is much too short. These are
**agreement diagnostics, not accuracy metrics**, since the user intervals and
the automatic intervals are unverified.

The bus-dwell rule produces C3905 proposals at `0.000-26.026`,
`35.535-64.564`, and `74.074-93.594` s. They overlap the first, second, and
third hand-observed bus intervals at IoU 0.881, 0.894, and 0.850. For C3896 it
produces `6.506-28.027` and `78.579-99.100` s at IoU 0.814 and 0.898. The
sixth user stopped-vehicle interval, C3905 `107.007-117.918`, has no bus-stop
match. This focused match is encouraging but does not validate the additional
bus proposals on C3896, C3897, or C3902. Actor and exact boundary checks remain
necessary.

Next implementation should verify the bus stop's live-lane boundary in the
zone editor and extend stopped-vehicle proposals beyond buses. Use signal
phases and per-lane tracks to reject
red-light queues from `congestion`. Tighten jaywalking with verified zebra and
island polygons, and failure-to-yield with same-carriageway-half occupancy and
vehicle front/rear boundaries. Keep all output provisional until a human
confirms the actor and interval.

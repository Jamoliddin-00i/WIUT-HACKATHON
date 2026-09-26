# Labeling guide

How Team Salen labels the sample videos, so that two people labeling the same
video get the same answer. The official definitions and start/end rules come
first; everything below them is our interpretation for the cases the official
text leaves open. If you meet a case this guide does not cover, label it your
best way, write a note on the event, and add the decision to the log at the end.

## 1. The basics

- **One label = one event**: `[start_sec, end_sec, label]`. Time is seconds
  from the first frame of the video, as shown in the tool.
- **Same class at the same time = one segment.** Two jaywalkers crossing at
  overlapping times get ONE jaywalking segment from the first one stepping on
  the road to the last one leaving it (official FAQ).
- **Same class, one after another**: if the gap between them is **1 s or
  less**, merge into one segment. If the gap is longer, two segments.
- **Different classes may overlap**: a car running the red light while a
  pedestrian jaywalks is two labels.
- **Cut by the video**: an event already happening at the first frame starts
  at `0`; one still happening at the last frame ends at the video duration.
- **Only label what you can see.** If the deciding moment is hidden (behind a
  bus, the gantry), use the first frame where it is visible again. If it is
  hidden for less than 1 s, estimate the moment between the frames you can see.
- **Unsure? Label it and write a note starting with `?`.** We review every `?`
  together. A missing event costs us more than an uncertain one.

## 2. Workflow (same for both of us)

1. **Pass 1, find events.** Watch the whole video at 1x (2x on quiet parts).
   When something happens, pause and add a rough label. Do not skip ahead.
2. **Pass 2, fix boundaries.** For each label, jump to its start, slow to
   0.25x and step frame by frame to the exact moment the start rule describes.
   Same for the end.
3. **Precision target.**
   - Crisp rules (crosses the stop line, steps onto the road, first contact
     visible, enters/leaves the crossing): **within 2 frames**.
   - Fuzzy rules (queue stops moving, onset of evasive action, vehicle comes
     to rest): **within 0.5 s**. There is no single right frame, so do not
     spend minutes on it.
4. **Validate** in the tool before you stop (overlaps, tiny segments).

## 3. Class by class

Official rule in bold, then our interpretation.

### jaywalking
**Pedestrian on the carriageway outside a crossing. Start: steps onto the
road. End: leaves the road.**
- Carriageway = asphalt where vehicles drive. The raised median, kerbs,
  sidewalks and the red-paved traffic islands are NOT carriageway.
- On the zebra stripes = legal, not jaywalking. A person who leaves the zebra
  (walks diagonally away from it) starts jaywalking when **both feet** are
  outside the painted crossing and its narrow edge area. They stop jaywalking
  when a foot is back on the crossing, kerb, median or island.
- If someone follows the zebra across the road but walks just beside its edge
  (roughly within one step), treat that as using the crossing, not jaywalking.
  Judge this from the feet and their path, not the upper body or a single
  frame. If they turn away from the crossing and continue through a traffic
  lane, label the part after they leave that narrow edge area.
- Start = first frame they enter the asphalt away from the crossing's narrow
  edge area. End = first frame both feet are off the carriageway or back on
  the crossing.
- People getting in or out of a vehicle stopped at the kerb: not jaywalking
  unless they walk out into a traffic lane.
- Cyclists and riders on the road are vehicles, not pedestrians.

### red_light
**Vehicle crosses the stop line while its signal is red. Start: front crosses
the stop line. End: vehicle leaves the intersection or the frame.**
- Only when you can see the signal that controls that vehicle. If you cannot
  see its signal, do not label red_light (note `? signal not visible`).
- Amber is not red. Red together with amber (just before green) counts as red.
- The stop line is the painted line before the zebra. Start = frame the front
  bumper passes over it.

### stop_line
**Vehicle stops past the stop line on red without entering the intersection.
Start: vehicle stops. End: signal turns green.**
- The front bumper must be past the stop line (on or over the zebra) when the
  vehicle comes to rest.
- End is when the signal turns green, even if the vehicle has not moved yet.
- If the same vehicle then drives through on red, it is ALSO red_light.

### stopped_vehicle
**Stationary on the carriageway 10 s or more, not in a queue at a signal.
Start: vehicle stops. End: moves again or is removed.**
- Start is the moment it stops, not 10 s later. Only label if it stays at
  least 10 s.
- Only vehicles stopped **in a live traffic lane** count (organizers,
  answer 4).
- IS a stopped vehicle: any vehicle standing in a live lane for 10 s or more
  when not queued at a red light, **including a bus stopped at the bus stop if
  it stands on the carriageway** (organizers, answer 2).
- Not a stopped vehicle: waiting in a signal queue; vehicles **parked at the
  kerb**, in parking spots or bays (the cars at the far left edge of the frame).
  These are ignored completely, even for the whole clip (organizers, answer 4).
- Stopped in a live lane for the whole clip = `[0, duration]`.

### failure_to_yield
**Vehicle drives through a crossing while a pedestrian is on it or stepping
onto it. Start: vehicle enters the crossing. End: vehicle leaves the crossing.**
- Count only pedestrians on the part of the crossing that is on the vehicle's
  own carriageway (the half of the zebra it drives over), or stepping onto it.
- Start = front enters the zebra stripes; end = rear leaves them.
- Applies whatever the signal shows. A pedestrian on the zebra on a red
  pedestrian light is **not jaywalking**; if a car drives through the crossing
  while they are on it, that car is failure_to_yield, otherwise it is nothing
  (organizers, answer 3).

### congestion
**Traffic at a standstill or crawling across all lanes of a direction. Start:
queue stops moving. End: queue clears.**
- An ordinary red-light queue is **not** congestion. Label it only when the
  queue does not clear on the green: the direction stays stopped or crawling
  through a green phase (organizers, answer 1).

### near_miss
**Sharp braking or swerving to avoid a collision; no contact. Start: onset of
evasive action. End: road users are clear of each other.**
- Needs a visible evasive action: a sudden stop, nose dip, swerve, or a
  pedestrian jumping back. Two vehicles merely passing close is not enough.
- Start = first frame the braking or swerve begins. End = first frame they are
  no longer on a collision course.

### accident
**Contact between road users or with a fixed object. Start: first frame contact
is visible. End: all involved stop moving or leave the frame.**

### wrong_way
**Vehicle moves against the traffic direction of its lane. Start: enters the
opposing lane. End: returns to a correct lane or leaves the frame.**
- Lane directions (from the EDA): the far carriageway flows right to left, away
  from the camera; the near carriageway flows towards the camera.
- A short reverse (a few metres, e.g. backing out of a spot) is not wrong_way.

### illegal_u_turn, illegal_turn, solid_line_crossing
- Only label when a prohibition is **visible**: a solid line, a no-turn or
  no-U-turn sign, or a turn from a lane that clearly does not allow it. If you
  cannot see a marking or sign that forbids it, do not label; add a `?` note.
- U-turn: start when the vehicle starts turning, end when it completes the turn.
- Solid line: start when a wheel crosses the line, end when the vehicle is fully
  in the new lane.

### road_obstacle, fire_smoke
- Obstacle: debris, animal or fallen object on the carriageway. Pedestrians
  and vehicles are never obstacles.
- Smoke: real smoke or fire only. Exhaust in cold air and the camera's sudden
  brightness changes are not smoke.

## 4. Calibration before labeling

Before real labeling, both of us label the **same first 60 seconds of C3897**
separately, then compare in the tool. For every difference, decide which
reading of the rule is right and write it in the decision log. This takes
about 15 minutes and prevents an hour of relabeling.

## 5. Questions to the organizers and their answers

Asked 2026-09-24, answered the same day. These are now binding for our labels.

1. Does an ordinary red-light queue count as **congestion**, or only a queue
   that does not clear in the next green?
2. Is a **bus dwelling at the bus stop** a stopped_vehicle?
3. A pedestrian crossing the zebra **on a red pedestrian light** while cars
   pass on green: is that failure_to_yield, jaywalking, or neither?
4. Vehicles **parked at the kerb for the whole clip**: stopped_vehicle
   `[0, duration]`, or ignored?
5. Are the annotation boundaries judged against a frame tolerance, or only by
   temporal IoU as in evaluate.py?

**Answers:**
1. No. A normal red-light queue is not congestion, only when the queue does
   not clear on the green.
2. Yes, if the bus stands on the carriageway.
3. Not jaywalking. If a car drives through the crossing while the pedestrian
   is on it, that car is failure_to_yield; otherwise neither.
4. Ignored. Parked at the kerb is not an event; stopped_vehicle is a vehicle
   stopped in a live lane.
5. Temporal IoU only, as in evaluate.py. No frame tolerance.

## 6. Decision log

Add one line per decision: date, who, the case, what we decided.

- 2026-09-24, team: guide v1 written.
- 2026-09-24, organizers: answers 1 to 5 received; congestion, stopped_vehicle
  and failure_to_yield rules updated to match (guide v2).

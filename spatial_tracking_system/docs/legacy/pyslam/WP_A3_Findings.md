# WP-A3 Baseline Freeze: Findings

Baseline data: `pyslam/tools/baseline_store.json`. Code state: Phase 0
frame-to-keyframe odometry + WP-A0 (direction fix, USAC, recording) +
WP-B4 (graph-layer fixes) + WP-A2 fixtures. This is the number WP-B1's
frame-to-local-map upgrade has to beat, not the old flat-path numbers,
which could not distinguish a correct odometry axis from an inverted
one (see the Phase 1 plan, F2).

## Summary table

| Fixture | Seeds | Median trans drift | Median rot drift | Mirror? | LOST |
|---|---|---|---|---|---|
| square6dof | 5 | 1.19%/m | 0.49deg/m | No, any seed | 0 |
| room_orbit | 5 | 2.76%/m | 0.76deg/m | No, any seed | 0 |
| static_60s | 3 | 0.1-0.3cm final drift | 0.03-0.08deg final | -- | 0 |
| aliasing_rooms | 3 | 2.5-4.2%/m (misleading, see below) | -- | No | 0 |
| corridor_v2 | 1 (220/480 frames, see caveat) | 18.3%/m | 5.5deg/m | No | 7 |

## Three findings that change what WP-B1/B3 should prioritise

**1. corridor_v2's bad drift number is a session-handling gap, not a
tracking-accuracy gap.** 18.3%/m looks like catastrophic odometry
failure. It isn't: per-link RPE on the links that DID form is 0.63cm
mean, 2.4cm max -- as good as any other fixture. Traced directly: 7
LOST events fired in 220 frames, and every single one produced a graph
node with no incoming odometry link (confirmed exactly: 7 LOST == 7
link-less nodes). `force_new_keyframe()` re-anchors local tracking but
the pipeline has no concept of a new SESSION yet -- it keeps writing
`pose_odom` into the same trajectory as if nothing happened, so each
LOST silently inserts a discontinuity that a naive cumulative-drift
metric reads as catastrophic error. **This re-ranks WP-B3 above WP-B1**
for this fixture specifically: the frame-to-local-map tracker in WP-B1
may well reduce LOST frequency, but as long as a LOST event corrupts
the trajectory this badly, better tracking alone won't fix corridor_v2.
WP-B3's session-on-LOST design (start a fresh, independently-scored
trajectory segment rather than silently continuing the old one) needs
to land before corridor_v2's numbers mean anything.

**2. corridor_v2's baseline is incomplete and the LOST rate itself is
suspect.** Only 1 seed, 220 of 480 frames (loop never closes). Cost:
~1s/frame (24 walls vs 6 for the cheapest fixture, plus retrieval cost
that grows with working-memory size -- see the Phase 1 plan's F5
finding), so a full run does not complete inside this environment's
interactive session even once. 7 LOST/220 frames is a high enough rate
that it's worth checking whether it's inherent to the corridor's
geometry (long straight runs -> weak parallax for the F2F tracker,
which is a real and expected weakness this fixture was built to
surface) or an artifact of the still-heuristic keyframe/LOST thresholds
in Phase 0's odometry. WP-B1 should re-run this exact scenario once
implemented and the delta will answer that question directly.

**3. aliasing_rooms produces three cross-room false loop closures that
no threshold can catch, and the underlying cause is worse than
"loop closure got fooled."** Checked whether the three false loops
(inliers 60-412, posterior 0.6-0.85) were noisy: they are not. They
mutually agree with each other to within 0.6-3.2cm, meaning the
appearance evidence for the aliased match is genuinely as good as a
real one, not a low-confidence fluke a stricter threshold would filter.
Traced further and found the frame-to-frame ODOMETRY was fooled first:
`status_log` was empty across the entire run (LOST never fired) and the
odometry link spanning the 20m room-to-room cut reported a normal-
looking 3.9cm step with 160 inliers. Appearance-only tracking cannot
distinguish "revisiting this place" from "looking at an identical but
different place" at any stage of the pipeline when the two scenes are
truly indistinguishable in the sensor's own modality -- this is not a
Phase 1 bug to fix, it's a hard limitation to design around. Added
`PipelineResult.cross_session_merges` (a union-find over the graph) so
a merge crossing a genuine session boundary is at least auditable going
forward, though it does not fire on THIS specific failure (the graph
never actually disconnected, because odometry itself didn't notice
anything was wrong). The one lead worth carrying into Phase 5: IMU
fusion would at least notice zero real acceleration across an apparent
20m visual jump, which vision alone structurally cannot.

## What this means for WP-B1's design

- Prioritise WP-B3 (session-on-LOST) work either alongside or before
  WP-B1's tracking accuracy improvements -- corridor_v2's numbers won't
  be trustworthy until it lands, and it's a smaller, more contained
  piece of work than the full local-map tracker.
- WP-B1 should be validated primarily against square6dof and room_orbit
  first (clean baselines, no session-handling confound), then corridor_v2
  once WP-B3 is in, then a full un-capped corridor_v2 run as a
  background/offline job to get a trustworthy multi-seed number.
- aliasing_rooms is not a WP-B1 target at all -- it's correctly scoped
  to Phase 4/5 per the original plan, and this session's findings only
  sharpen why.

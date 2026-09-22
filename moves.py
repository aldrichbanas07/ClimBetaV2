"""
Turns the raw contact-event stream into a sequence of discrete MOVES, and
describes each move's body position from the tracked pose landmarks.

Why this layer exists
---------------------
contact_detection.py deliberately errs toward over-detection: any limb that
sits within `contact_radius_px` of a hold for `dwell_frames` frames is an
event. On a real attempt that produced ~180 "contacts" for a 16-hold climb -
mostly 0.1-0.2s flybys, a wrist passing a foot hold on its way somewhere
else, and the ankle and toe of the same foot each registering the same hold
separately. A move-by-move analysis over that stream is meaningless, so this
module cleans it up:

  1. ankle + toe of one foot collapse into a single `left_foot`/`right_foot`
     contact point (a foot is one limb, not two).
  2. repeated events on the same (limb, hold) separated by a short gap merge
     into one occupancy interval - the gap count is kept as `regrip_count`,
     since a regrip is itself informative.
  3. intervals shorter than `min_hold_time_s` are dropped as flybys.
  4. where one limb still has two overlapping intervals, the longer one wins.

What remains is, per limb, an ordered list of holds it actually occupied. A
MOVE is one transition in that list: limb releases hold A, establishes on
hold B. Moves from all four limbs are then interleaved by the frame the new
hold was established.

Measurement honesty
-------------------
Everything here is 2D image-plane geometry from a single fixed camera. That
supports real statements about how far the hips traveled, whether they rose
or shifted sideways, how extended a reach was relative to torso length, and
how many other limbs were supporting at the time. It does NOT support any
claim involving depth: distance from the wall, hip rotation/twist, whether
the climber was square or side-on, or where weight was actually loaded.
MEASUREMENT_CAVEATS travels with the output so downstream consumers
(coach.py) can't quietly forget this.
"""

import json
import os

import numpy as np
import pandas as pd

from metrics import compute_frame_metrics


# One physical limb may be tracked by more than one landmark. Order matters:
# the first landmark present in the data is treated as that limb's primary
# contact point (a toe touches a hold; the ankle is just nearby).
LIMB_GROUPS = {
    "left_hand": ("left_wrist",),
    "right_hand": ("right_wrist",),
    "left_foot": ("left_toe", "left_ankle"),
    "right_foot": ("right_toe", "right_ankle"),
}

_GROUP_BY_LANDMARK = {
    landmark: group for group, landmarks in LIMB_GROUPS.items() for landmark in landmarks
}

# For reach extension: a hand's reach is measured from its shoulder, a
# foot's from its hip.
_ANCHOR_LANDMARK = {
    "left_hand": "left_shoulder",
    "right_hand": "right_shoulder",
    "left_foot": "left_hip",
    "right_foot": "right_hip",
}

MEASUREMENT_CAVEATS = [
    "All positions are 2D pixel coordinates from one fixed camera. There is no depth, so nothing here can measure distance from the wall, hip rotation/twist, whether the body was square or side-on, or which limb was actually bearing weight.",
    "Pixel distances are only comparable within this one video - they are not real-world units and mean nothing across different videos or camera setups.",
    "velocity_ratio compares a limb's speed to its own trailing average, so it measures whether a movement was abrupt relative to that climber's own recent pace. It is not a comparison against any reference or ideal ascent.",
    "Hold placeholder_type (jug/crimp/sloper/pinch) is derived from the hold's internal ID (hole_id % 4), not from any real hold-shape data, which the Kilter database does not contain.",
    "A move is inferred from proximity between a tracked keypoint and a calibrated hold position. A hold used with a part of the body that isn't tracked, or occluded from the camera, will be missing entirely.",
]


def _build_pose_lookup(pose_df):
    """
    frame-indexed lookup of every landmark's pixel position, gap-filled by
    interpolation so a few frames of missed detection don't blank out a
    move's body-position features.
    """
    pivot = pose_df.pivot_table(index="frame", columns="landmark", values=["x_px", "y_px"])
    if pivot.empty:
        return pivot
    full_index = range(int(pivot.index.min()), int(pivot.index.max()) + 1)
    return pivot.reindex(full_index).interpolate(method="index", limit_direction="both")


def _point(lookup, landmark, frame):
    """Pixel position of one landmark at one frame, or None if unavailable."""
    x_col, y_col = ("x_px", landmark), ("y_px", landmark)
    if x_col not in lookup.columns or y_col not in lookup.columns:
        return None
    if frame not in lookup.index:
        return None
    x, y = lookup.at[frame, x_col], lookup.at[frame, y_col]
    if pd.isna(x) or pd.isna(y):
        return None
    return (float(x), float(y))


def _center(lookup, left_landmark, right_landmark, frame):
    """Midpoint of a left/right landmark pair, tolerating one side missing."""
    left = _point(lookup, left_landmark, frame)
    right = _point(lookup, right_landmark, frame)
    if left is None and right is None:
        return None
    if left is None:
        return right
    if right is None:
        return left
    return ((left[0] + right[0]) / 2.0, (left[1] + right[1]) / 2.0)


def _hip_center(lookup, frame):
    return _center(lookup, "left_hip", "right_hip", frame)


def _shoulder_center(lookup, frame):
    return _center(lookup, "left_shoulder", "right_shoulder", frame)


def _distance(a, b):
    if a is None or b is None:
        return None
    return float(np.hypot(a[0] - b[0], a[1] - b[1]))


def _path_length(lookup, landmark_fn, start_frame, end_frame):
    """Total distance traveled by a derived point across a frame window."""
    previous = None
    total = 0.0
    seen = False
    for frame in range(start_frame, end_frame + 1):
        current = landmark_fn(lookup, frame)
        if current is None:
            continue
        seen = True
        if previous is not None:
            total += float(np.hypot(current[0] - previous[0], current[1] - previous[1]))
        previous = current
    return total if seen else None


def _interval_length(interval):
    return interval["end_frame"] - interval["start_frame"] + 1


def _overlap(a, b):
    return max(0, min(a["end_frame"], b["end_frame"]) - max(a["start_frame"], b["start_frame"]) + 1)


def clean_contacts(contacts, fps, config):
    """
    Collapse the raw contact stream into non-overlapping occupancy intervals:
    one per (limb, hold) stretch the limb genuinely spent on that hold.

    See the module docstring for the four cleaning stages. Returns a list of
    {limb, hole_id, role, placeholder_type, start_frame, end_frame,
    regrip_count}, ordered by start_frame.
    """
    merge_gap = config.get("merge_gap_frames", 8)
    min_hold_time_s = config.get("min_hold_time_s", 0.25)
    min_frames = max(1, int(round(min_hold_time_s * fps))) if fps else 1

    buckets = {}
    for event in contacts:
        group = _GROUP_BY_LANDMARK.get(event["limb"])
        if group is None:
            continue
        buckets.setdefault((group, event["hole_id"]), []).append(event)

    merged = []
    for (group, hole_id), events in buckets.items():
        events.sort(key=lambda e: e["start_frame"])
        current = None
        for event in events:
            if current is not None and event["start_frame"] - current["end_frame"] <= merge_gap:
                current["end_frame"] = max(current["end_frame"], event["end_frame"])
                current["regrip_count"] += 1
                continue
            if current is not None:
                merged.append(current)
            current = {
                "limb": group,
                "hole_id": hole_id,
                "role": event.get("role"),
                "placeholder_type": event.get("placeholder_type"),
                "start_frame": event["start_frame"],
                "end_frame": event["end_frame"],
                "regrip_count": 0,
            }
        if current is not None:
            merged.append(current)

    # One limb can't be on two holds at once. Longest interval wins; anything
    # overlapping an already-kept interval by more than half its own length is
    # the weaker duplicate (typically ankle-vs-toe on adjacent holds).
    kept = []
    for group in LIMB_GROUPS:
        candidates = sorted(
            (m for m in merged if m["limb"] == group),
            key=_interval_length,
            reverse=True,
        )
        chosen = []
        for candidate in candidates:
            if any(_overlap(candidate, c) > 0.5 * _interval_length(candidate) for c in chosen):
                continue
            chosen.append(candidate)
        chosen = [c for c in chosen if _interval_length(c) >= min_frames]

        # A limb leaving a hold and coming back to the same hold is a
        # readjustment, not a move to somewhere else - otherwise these
        # surface as nonsense "hole 4 -> hole 4" moves with zero reach. The
        # round trip is recorded as a regrip instead. Unconditional on gap
        # length, unlike the earlier merge: however long the limb was off,
        # it still ended up back where it started.
        chosen.sort(key=lambda c: c["start_frame"])
        collapsed = []
        for interval in chosen:
            if collapsed and collapsed[-1]["hole_id"] == interval["hole_id"]:
                collapsed[-1]["end_frame"] = interval["end_frame"]
                collapsed[-1]["regrip_count"] += interval["regrip_count"] + 1
                continue
            collapsed.append(interval)
        kept.extend(collapsed)

    return sorted(kept, key=lambda c: c["start_frame"])


def _primary_landmark(group, metrics_df):
    available = set(metrics_df["landmark"].unique())
    for landmark in LIMB_GROUPS[group]:
        if landmark in available:
            return landmark
    return None


def _velocity_stats(metrics_df, landmark, start_frame, end_frame):
    if landmark is None:
        return None, None
    window = metrics_df[
        (metrics_df["landmark"] == landmark)
        & (metrics_df["frame"] >= start_frame)
        & (metrics_df["frame"] <= end_frame)
    ]
    ratios = window["velocity_ratio"].dropna()
    if ratios.empty:
        return None, None
    return float(ratios.max()), float(ratios.mean())


def _support_points_at(clean_contacts_list, limb, frame):
    """How many OTHER limbs were on a hold at this frame."""
    return sum(
        1
        for c in clean_contacts_list
        if c["limb"] != limb and c["start_frame"] <= frame <= c["end_frame"]
    )


def _round(value, digits=1):
    return None if value is None else round(value, digits)


def _build_move(limb, source, target, clean_list, lookup, metrics_df, calibrated_holds, fps):
    establish_frame = target["start_frame"]
    release_frame = source["end_frame"] if source else establish_frame
    # A limb can re-establish before the detector drops the old interval;
    # clamp so the window is never inverted.
    release_frame = min(release_frame, establish_frame)

    landmark = _primary_landmark(limb, metrics_df)
    peak_ratio, mean_ratio = _velocity_stats(metrics_df, landmark, release_frame, establish_frame)

    hip_at_release = _hip_center(lookup, release_frame)
    hip_at_establish = _hip_center(lookup, establish_frame)
    shoulder_at_establish = _shoulder_center(lookup, establish_frame)

    torso_length = _distance(shoulder_at_establish, hip_at_establish)
    anchor = _point(lookup, _ANCHOR_LANDMARK[limb], establish_frame)
    limb_point = _point(lookup, landmark, establish_frame) if landmark else None
    reach_length = _distance(anchor, limb_point)
    extension_ratio = (
        reach_length / torso_length if reach_length is not None and torso_length else None
    )

    target_xy = calibrated_holds.get(target["hole_id"])
    source_xy = calibrated_holds.get(source["hole_id"]) if source else None

    hip_rise = (
        hip_at_release[1] - hip_at_establish[1]
        if hip_at_release and hip_at_establish
        else None
    )
    hip_shift = (
        hip_at_establish[0] - hip_at_release[0]
        if hip_at_release and hip_at_establish
        else None
    )
    hip_offset = (
        hip_at_establish[0] - target_xy[0] if hip_at_establish and target_xy else None
    )

    return {
        "move_number": None,  # assigned after all limbs are interleaved
        "limb": limb,
        "timestamp_s": _round(establish_frame / fps, 2) if fps else None,
        "duration_s": _round((establish_frame - release_frame) / fps, 2) if fps else None,
        "from_hole_id": source["hole_id"] if source else None,
        "from_role": source["role"] if source else None,
        "to_hole_id": target["hole_id"],
        "to_role": target["role"],
        "to_placeholder_type": target["placeholder_type"],
        "reach_distance_px": _round(_distance(source_xy, target_xy)),
        "dwell_on_previous_hold_s": (
            _round(_interval_length(source) / fps, 2) if source and fps else None
        ),
        "time_on_new_hold_s": _round(_interval_length(target) / fps, 2) if fps else None,
        "regrip_count": target["regrip_count"],
        "support_points_at_start": _support_points_at(clean_list, limb, release_frame),
        "peak_velocity_ratio": _round(peak_ratio, 2),
        "mean_velocity_ratio": _round(mean_ratio, 2),
        "hip_travel_px": _round(_path_length(lookup, _hip_center, release_frame, establish_frame)),
        "hip_rise_px": _round(hip_rise),
        "hip_lateral_shift_px": _round(hip_shift),
        "hip_horizontal_offset_from_target_px": _round(hip_offset),
        "limb_extension_ratio": _round(extension_ratio, 2),
    }


def build_move_sequence(pose_df, contacts, calibrated_holds, config, fps):
    """
    Full pipeline: clean the contact stream, segment it into moves, and
    attach per-move body-position features.

    Returns (moves, clean_contacts_list). Each limb's first interval is
    emitted as a move with from_hole_id=None - that's the starting position,
    not a transition.
    """
    clean_list = clean_contacts(contacts, fps, config)
    if not clean_list:
        return [], []

    lookup = _build_pose_lookup(pose_df)
    metrics_df = compute_frame_metrics(pose_df, config)

    by_limb = {}
    for contact in clean_list:
        by_limb.setdefault(contact["limb"], []).append(contact)

    moves = []
    for limb, intervals in by_limb.items():
        intervals.sort(key=lambda c: c["start_frame"])
        for index, target in enumerate(intervals):
            source = intervals[index - 1] if index > 0 else None
            moves.append(
                _build_move(limb, source, target, clean_list, lookup, metrics_df, calibrated_holds, fps)
            )

    moves.sort(key=lambda m: (m["timestamp_s"] if m["timestamp_s"] is not None else 0))

    # Everything after the last finish-hold establishment is the climber
    # coming off the wall, not climbing - on the test footage that was six
    # moves of feet dropping onto start holds and a hand swinging 330px down
    # to a foot hold, all of which read as terrible technique if analysed.
    # Only applies when a finish hold was actually reached.
    if config.get("truncate_after_finish", True):
        finish_indices = [i for i, m in enumerate(moves) if m["to_role"] == "finish"]
        if finish_indices:
            moves = moves[: finish_indices[-1] + 1]

    for number, move in enumerate(moves, start=1):
        move["move_number"] = number

    return moves, clean_list


def _main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Stand-alone move segmentation runner")
    parser.add_argument("video_path", help="Path to the climb video (used to find cached pose/contacts/holds files)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None, help="JSON output path (default: cache/<video_name>_moves.json)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cache_dir = os.path.dirname(config["db_path"])
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]

    pose_df = pd.read_csv(os.path.join(cache_dir, f"{video_name}_pose.csv"))

    with open(os.path.join(cache_dir, f"{video_name}_contacts.json"), "r", encoding="utf-8") as f:
        contacts = json.load(f)

    with open(os.path.join(cache_dir, f"{video_name}_holds.json"), "r", encoding="utf-8") as f:
        calibrated_holds = {int(k): tuple(v) for k, v in json.load(f).items()}

    time_diffs = pose_df.sort_values("frame")["timestamp_s"].diff()
    frame_diffs = pose_df.sort_values("frame")["frame"].diff()
    per_frame_dt = (time_diffs / frame_diffs).replace([np.inf, -np.inf], np.nan).dropna()
    fps = 1.0 / per_frame_dt.median() if not per_frame_dt.empty else 30.0

    moves, clean_list = build_move_sequence(pose_df, contacts, calibrated_holds, config, fps)

    out_path = args.out or os.path.join(cache_dir, f"{video_name}_moves.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(moves, f, indent=2)

    print(f"{len(contacts)} raw contacts -> {len(clean_list)} held positions -> {len(moves)} moves")
    print(f"Wrote -> {out_path}\n")
    for move in moves:
        source = move["from_hole_id"] if move["from_hole_id"] is not None else "start"
        print(
            f"  #{move['move_number']:>2} t={move['timestamp_s']:>6}s {move['limb']:<11} "
            f"{source} -> hole {move['to_hole_id']} ({move['to_role']})  "
            f"reach={move['reach_distance_px']} hip_travel={move['hip_travel_px']} "
            f"peak_ratio={move['peak_velocity_ratio']} support={move['support_points_at_start']}"
        )


if __name__ == "__main__":
    _main()

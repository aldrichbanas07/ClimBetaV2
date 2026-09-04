"""
Matches pose keypoints to calibrated hold positions, frame by frame, in the
video's own pixel space (no coordinate transform - both pose.py and
calibration.py already produce pixel coordinates for the same video).

A contact event requires the nearest calibrated hold to stay the same and
stay within `contact_radius_px` for at least `dwell_frames` consecutive
frames, so a limb swinging past a hold without using it doesn't register.
"""

import json
import os

import cv2
import numpy as np
import pandas as pd

from calibration import _hex_to_bgr
from hold_type_map import PLACEHOLDER_TYPES


def _load_hold_type_map(path="hold_type_map.json"):
    """
    Load the static hole_id -> placeholder type table. It only covers real
    Aurora hole_ids (built once from the database, see hold_type_map.py) - a
    climb built by kilter_data.build_manual_climb() uses fabricated hole_ids
    that won't be in it. detect_contacts() falls back to the same formula
    (hole_id % 4) for any hole_id missing from this table, so the type is
    still deterministic and consistent even for manually-transcribed climbs.
    """
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(hole_id): hold_type for hole_id, hold_type in raw.items()}


def detect_contacts(pose_df, calibrated_holds, climb, hold_type_map, config):
    """
    Returns an ordered list of contact events:
        {hole_id, role, placeholder_type, limb, start_frame, end_frame, dwell_frames}
    """
    contact_radius_px = config["contact_radius_px"]
    dwell_frames = config["dwell_frames"]
    min_visibility = config.get("min_visibility", 0.5)

    role_by_hole = {p["hole_id"]: p["role"] for p in climb["placements"]}

    hole_ids = list(calibrated_holds.keys())
    hole_xy = np.array([calibrated_holds[h] for h in hole_ids], dtype=float)

    df = pose_df[pose_df["visibility"] >= min_visibility].copy()
    if df.empty or not hole_ids:
        return []

    points = df[["x_px", "y_px"]].to_numpy()
    diffs = points[:, None, :] - hole_xy[None, :, :]
    dists = np.sqrt((diffs**2).sum(axis=2))
    nearest_idx = dists.argmin(axis=1)

    df["nearest_hole_id"] = [hole_ids[i] for i in nearest_idx]
    df["nearest_dist"] = dists[np.arange(len(df)), nearest_idx]
    df["in_contact"] = df["nearest_dist"] <= contact_radius_px

    def finalize(run, limb):
        hole_id = run["hole_id"]
        return {
            "hole_id": hole_id,
            "role": role_by_hole.get(hole_id),
            "placeholder_type": hold_type_map.get(hole_id, PLACEHOLDER_TYPES[hole_id % 4]),
            "limb": limb,
            "start_frame": run["start_frame"],
            "end_frame": run["end_frame"],
            "dwell_frames": run["dwell_frames"],
        }

    events = []
    for limb, group in df.groupby("landmark"):
        group = group.sort_values("frame")
        run = None

        for _, row in group.iterrows():
            frame = int(row["frame"])
            if row["in_contact"]:
                same_run = (
                    run is not None
                    and run["hole_id"] == row["nearest_hole_id"]
                    and frame == run["end_frame"] + 1
                )
                if same_run:
                    run["end_frame"] = frame
                    run["dwell_frames"] += 1
                    continue

                if run is not None and run["dwell_frames"] >= dwell_frames:
                    events.append(finalize(run, limb))
                run = {
                    "hole_id": row["nearest_hole_id"],
                    "start_frame": frame,
                    "end_frame": frame,
                    "dwell_frames": 1,
                }
            else:
                if run is not None and run["dwell_frames"] >= dwell_frames:
                    events.append(finalize(run, limb))
                run = None

        if run is not None and run["dwell_frames"] >= dwell_frames:
            events.append(finalize(run, limb))

    events.sort(key=lambda e: e["start_frame"])
    return events


LIMB_COLORS_BGR = {
    "left_wrist": (0, 0, 255),
    "right_wrist": (0, 128, 255),
    "left_ankle": (255, 0, 0),
    "right_ankle": (255, 128, 0),
    "left_toe": (0, 255, 255),
    "right_toe": (0, 255, 0),
}


def render_contact_overlay(video_path, calibrated_holds, climb, events, out_path):
    """
    Write an annotated copy of `video_path`: every calibrated hold is drawn
    as a small dim circle (colored by its role's LED color) at all times,
    and lights up as a bright ring labeled with the limb currently touching
    it during any active contact event, for visually sanity-checking
    contact detection against the source footage.
    """
    led_color_by_hole = {p["hole_id"]: p.get("role_led_color") for p in climb["placements"]}

    # frame -> list of (hole_id, limb) active at that frame
    active_by_frame = {}
    for event in events:
        for frame in range(event["start_frame"], event["end_frame"] + 1):
            active_by_frame.setdefault(frame, []).append((event["hole_id"], event["limb"]))

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        for hole_id, (x, y) in calibrated_holds.items():
            color = _hex_to_bgr(led_color_by_hole.get(hole_id))
            cv2.circle(frame, (int(x), int(y)), 5, color, 1)

        for hole_id, limb in active_by_frame.get(frame_index, []):
            x, y = calibrated_holds[hole_id]
            color = LIMB_COLORS_BGR.get(limb, (255, 255, 255))
            cv2.circle(frame, (int(x), int(y)), 16, color, 3)
            cv2.putText(
                frame,
                limb,
                (int(x) + 20, int(y)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                2,
            )

        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()


def _main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Stand-alone contact detection runner")
    parser.add_argument("video_path", help="Path to the climb video (used to find cached pose/holds files)")
    parser.add_argument("climb_uuid", help="Climb uuid (cache/<uuid>.json must already exist)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None, help="JSON output path (default: cache/<video_name>_contacts.json)")
    parser.add_argument(
        "--visualize",
        default=None,
        help="Also write an annotated copy of the video with contact events highlighted, to this path",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cache_dir = os.path.dirname(config["db_path"])
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]

    pose_csv_path = os.path.join(cache_dir, f"{video_name}_pose.csv")
    holds_json_path = os.path.join(cache_dir, f"{video_name}_holds.json")
    climb_json_path = os.path.join(cache_dir, f"{args.climb_uuid}.json")

    pose_df = pd.read_csv(pose_csv_path)

    with open(holds_json_path, "r", encoding="utf-8") as f:
        calibrated_holds = {int(k): tuple(v) for k, v in json.load(f).items()}

    with open(climb_json_path, "r", encoding="utf-8") as f:
        climb = json.load(f)

    hold_type_map = _load_hold_type_map()

    events = detect_contacts(pose_df, calibrated_holds, climb, hold_type_map, config)

    out_path = args.out or os.path.join(cache_dir, f"{video_name}_contacts.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(events, f, indent=2)

    print(f"Detected {len(events)} contact events -> {out_path}")
    for event in events:
        print(
            f"  frames {event['start_frame']:>5}-{event['end_frame']:<5} "
            f"({event['dwell_frames']:>3}f) {event['limb']:<12} -> "
            f"hole {event['hole_id']} ({event['role']}, {event['placeholder_type']})"
        )

    if args.visualize:
        render_contact_overlay(args.video_path, calibrated_holds, climb, events, args.visualize)
        print(f"Wrote annotated video -> {args.visualize}")


if __name__ == "__main__":
    _main()

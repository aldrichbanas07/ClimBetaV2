"""
Computes movement metrics from tracked pose keypoints and joins them with
contact events, producing one compact structured JSON summary for the whole
attempt.

Honesty constraint (deliberate, see project notes): there is no reference
"ideal" ascent to compare against, so `velocity_ratio` and cumulative
distance are described here as movement smoothness/consistency metrics -
they say how steady or jerky the climber's movement was relative to their
own recent pace, never whether their path or technique was "correct".
Downstream (coach.py) must keep that same framing.
"""

import json
import os

import numpy as np
import pandas as pd


def compute_frame_metrics(pose_df, config):
    """
    Per landmark, per frame: frame-to-frame speed (px/s), a trailing rolling
    average of that speed, the ratio of current speed to that trailing
    average (the smoothness/consistency signal - near 1 is steady movement,
    spikes mean a sudden burst relative to recent pace), and cumulative
    distance traveled (px).

    Speed uses the real elapsed time between frames (timestamp_s), not frame
    index, since pose.py only emits rows for frames with a detected person -
    gaps from missed detections are handled correctly this way, though any
    movement during a missed-detection gap is necessarily invisible to this
    calculation.
    """
    window = config.get("velocity_window_frames", 15)

    df = pose_df.sort_values(["landmark", "frame"]).copy()
    metrics_frames = []

    for landmark, group in df.groupby("landmark"):
        group = group.sort_values("frame").reset_index(drop=True)

        dx = group["x_px"].diff()
        dy = group["y_px"].diff()
        dt = group["timestamp_s"].diff()
        step_distance = np.sqrt(dx**2 + dy**2)
        speed = step_distance / dt.replace(0, np.nan)

        trailing_avg_speed = speed.rolling(window=window, min_periods=1).mean().shift(1)
        velocity_ratio = speed / trailing_avg_speed.replace(0, np.nan)

        group["step_distance_px"] = step_distance.fillna(0)
        group["cumulative_distance_px"] = group["step_distance_px"].cumsum()
        group["speed_px_s"] = speed
        group["trailing_avg_speed_px_s"] = trailing_avg_speed
        group["velocity_ratio"] = velocity_ratio

        metrics_frames.append(group)

    return pd.concat(metrics_frames, ignore_index=True)


def _approach_velocity(metrics_df, limb, start_frame, window):
    """Mean speed of `limb` over the `window` frames immediately before start_frame."""
    subset = metrics_df[
        (metrics_df["landmark"] == limb)
        & (metrics_df["frame"] < start_frame)
        & (metrics_df["frame"] >= start_frame - window)
    ]
    if subset.empty:
        return None
    value = subset["speed_px_s"].mean()
    return None if pd.isna(value) else float(value)


def build_attempt_summary(pose_df, contacts, climb, config, calibrated_holds=None):
    """
    Returns one compact JSON-serializable dict summarizing the whole
    attempt: climb metadata, overall movement smoothness/consistency by
    limb, and every contact event enriched with its approach velocity and
    dwell time in seconds.

    When `calibrated_holds` is supplied, the raw contact stream is also
    segmented into a move-by-move sequence with per-move body-position
    features (see moves.py) - that sequence is what drives the step-by-step
    coaching analysis, and the raw `contacts` list is dropped from the
    summary in that case, since it is mostly detector noise and would only
    invite the model to narrate flybys as if they were moves.
    """
    metrics_df = compute_frame_metrics(pose_df, config)
    approach_window = config.get("approach_window_frames", 10)

    # fps estimated from consecutive detected-frame timestamps, for
    # converting dwell_frames into seconds
    time_diffs = pose_df.sort_values("frame")["timestamp_s"].diff()
    frame_diffs = pose_df.sort_values("frame")["frame"].diff()
    per_frame_dt = (time_diffs / frame_diffs).replace([np.inf, -np.inf], np.nan).dropna()
    fps = 1.0 / per_frame_dt.median() if not per_frame_dt.empty else None

    by_limb = {}
    for limb, group in metrics_df.groupby("landmark"):
        ratios = group["velocity_ratio"].dropna()
        by_limb[limb] = {
            "total_distance_px": float(group["cumulative_distance_px"].max()),
            "mean_velocity_ratio": float(ratios.mean()) if not ratios.empty else None,
            "std_velocity_ratio": float(ratios.std()) if len(ratios) > 1 else None,
        }

    enriched_contacts = []
    for event in contacts:
        dwell_time_s = event["dwell_frames"] / fps if fps else None
        enriched_contacts.append(
            {
                **event,
                "dwell_time_s": dwell_time_s,
                "approach_velocity_px_s": _approach_velocity(
                    metrics_df, event["limb"], event["start_frame"], approach_window
                ),
            }
        )

    summary = {
        "climb": {
            "uuid": climb["uuid"],
            "name": climb["name"],
            "angle": climb["angle"],
        },
        "video": {
            "duration_s": float(pose_df["timestamp_s"].max()) if not pose_df.empty else None,
            "estimated_fps": fps,
        },
        "movement_smoothness_by_limb": by_limb,
    }

    if calibrated_holds is not None:
        # Imported here rather than at module level: moves.py imports
        # compute_frame_metrics from this module.
        import moves as moves_module

        move_sequence, clean_contacts = moves_module.build_move_sequence(
            pose_df, contacts, calibrated_holds, config, fps
        )
        summary["measurement_caveats"] = moves_module.MEASUREMENT_CAVEATS
        summary["moves"] = move_sequence
        summary["move_counts"] = {
            "raw_contact_events": len(contacts),
            "held_positions_after_cleaning": len(clean_contacts),
            "moves": len(move_sequence),
        }
    else:
        summary["contacts"] = enriched_contacts

    return summary


def _main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Stand-alone metrics runner")
    parser.add_argument("video_path", help="Path to the climb video (used to find cached pose/contacts files)")
    parser.add_argument("climb_uuid", help="Climb uuid (cache/<uuid>.json must already exist)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None, help="JSON output path (default: cache/<video_name>_metrics.json)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cache_dir = os.path.dirname(config["db_path"])
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]

    pose_df = pd.read_csv(os.path.join(cache_dir, f"{video_name}_pose.csv"))

    with open(os.path.join(cache_dir, f"{video_name}_contacts.json"), "r", encoding="utf-8") as f:
        contacts = json.load(f)

    with open(os.path.join(cache_dir, f"{args.climb_uuid}.json"), "r", encoding="utf-8") as f:
        climb = json.load(f)

    with open(os.path.join(cache_dir, f"{video_name}_holds.json"), "r", encoding="utf-8") as f:
        calibrated_holds = {int(k): tuple(v) for k, v in json.load(f).items()}

    summary = build_attempt_summary(pose_df, contacts, climb, config, calibrated_holds)

    out_path = args.out or os.path.join(cache_dir, f"{video_name}_metrics.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Wrote attempt summary -> {out_path}")
    print(f"  video duration: {summary['video']['duration_s']:.1f}s @ ~{summary['video']['estimated_fps']:.1f}fps")
    print("  movement smoothness by limb (mean velocity ratio, ~1 = steady, ignore units on distance):")
    for limb, stats in summary["movement_smoothness_by_limb"].items():
        ratio = stats["mean_velocity_ratio"]
        ratio_str = f"{ratio:.2f}" if ratio is not None else "n/a"
        print(f"    {limb:<12} mean_ratio={ratio_str}  total_distance_px={stats['total_distance_px']:.0f}")
    counts = summary["move_counts"]
    print(
        f"  {counts['raw_contact_events']} raw contact events -> "
        f"{counts['held_positions_after_cleaning']} held positions -> "
        f"{counts['moves']} moves"
    )


if __name__ == "__main__":
    _main()

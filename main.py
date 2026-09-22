"""
CLI entry point: runs the whole pipeline for one video against one climb.

Every stage caches its result under cache/, and every stage is skipped when
its cache file is already there - so re-running after a crash, or after
changing only the coaching prompt, costs nothing for the stages before it.
`--force <stage>` redoes that stage *and everything after it*, because a
later stage built on top of a regenerated earlier one is exactly the kind of
stale mix that produces confusing results rather than an error.

Calibration is the one interactive stage: it opens OpenCV windows and waits
for clicks. So its cache is checked before any expensive work starts - better
to be asked for clicks two seconds in than after a multi-minute pose pass.

The cached pose CSV gets a stronger check than "does the file exist": it is
validated against pose.TRACKED_LANDMARKS. Adding a tracked landmark (as the
torso landmarks were, for the body-position features) leaves behind a CSV
that still loads perfectly and silently yields null features downstream. That
failure is invisible unless something looks for it here.
"""

import argparse
import json
import os

import pandas as pd
import yaml

import calibration
import coach
import contact_detection
import hold_type_map as hold_type_map_module
import kilter_data
import metrics
import pose


# In pipeline order. Forcing one stage forces every stage after it.
STAGES = ("climb", "holds", "pose", "contacts", "metrics", "coach")


def resolve_forced(force_args):
    """Expand the requested stages to include everything downstream of them."""
    if not force_args:
        return set()
    if "all" in force_args:
        return set(STAGES)
    earliest = min(STAGES.index(stage) for stage in force_args)
    return set(STAGES[earliest:])


class _Paths:
    """Every cache file this pipeline reads or writes, for one video."""

    def __init__(self, config, video_path, climb_uuid):
        self.cache_dir = os.path.dirname(config["db_path"]) or "cache"
        name = os.path.splitext(os.path.basename(video_path))[0]
        self.climb = os.path.join(self.cache_dir, f"{climb_uuid}.json")
        self.holds = os.path.join(self.cache_dir, f"{name}_holds.json")
        self.pose = os.path.join(self.cache_dir, f"{name}_pose.csv")
        self.contacts = os.path.join(self.cache_dir, f"{name}_contacts.json")
        self.metrics = os.path.join(self.cache_dir, f"{name}_metrics.json")
        self.frame_metrics = os.path.join(self.cache_dir, f"{name}_frame_metrics.csv")
        self.coaching = os.path.join(self.cache_dir, f"{name}_coaching.json")


def _report(stage, action, path):
    print(f"[{stage:<8}] {action:<7} {path}")


def _pose_csv_is_current(path):
    """
    True only if the cached CSV covers every landmark pose.py now tracks.
    A CSV missing landmarks loads fine but produces null body-position
    features, so it has to be treated as absent rather than reused.
    """
    try:
        cached = set(pd.read_csv(path, usecols=["landmark"])["landmark"].unique())
    except (ValueError, KeyError, pd.errors.EmptyDataError):
        return False
    return set(pose.TRACKED_LANDMARKS).issubset(cached)


def run_pipeline(video_path, climb_uuid, config, forced=frozenset(), run_coach=True):
    """
    Run steps 1-7 end to end, reusing cached stage outputs where they exist.

    Returns (summary, analysis); analysis is None when run_coach is False.
    """
    paths = _Paths(config, video_path, climb_uuid)
    os.makedirs(paths.cache_dir, exist_ok=True)

    # 1: climb layout
    cached_climb = os.path.exists(paths.climb) and "climb" not in forced
    climb = kilter_data.fetch_climb(config, climb_uuid, use_cache="climb" not in forced)
    _report("climb", "using" if cached_climb else "wrote", paths.climb)
    print(f"           {climb['name']} @ {climb['angle']} deg, {len(climb['placements'])} placements")

    # 2: placeholder hold-type table (project-wide, built once)
    if not os.path.exists(hold_type_map_module.OUTPUT_PATH):
        table = hold_type_map_module.build_hold_type_map(config)
        with open(hold_type_map_module.OUTPUT_PATH, "w", encoding="utf-8") as f:
            json.dump(table, f, indent=2, sort_keys=True)
        _report("holdtype", "built", hold_type_map_module.OUTPUT_PATH)
    else:
        _report("holdtype", "using", hold_type_map_module.OUTPUT_PATH)
    hold_type_map = contact_detection._load_hold_type_map(hold_type_map_module.OUTPUT_PATH)

    # 3: calibration (interactive - checked early, before the expensive stages)
    cached_holds = os.path.exists(paths.holds) and "holds" not in forced
    calibrated_holds = calibration.calibrate(
        video_path, climb, paths.cache_dir, force="holds" in forced
    )
    _report("holds", "using" if cached_holds else "wrote", paths.holds)
    print(f"           {len(calibrated_holds)} holds labeled in this video")

    # 4: pose extraction
    if "pose" not in forced and os.path.exists(paths.pose) and _pose_csv_is_current(paths.pose):
        pose_df = pd.read_csv(paths.pose)
        _report("pose", "using", paths.pose)
    else:
        if os.path.exists(paths.pose) and "pose" not in forced:
            print("[pose    ] cached CSV predates the current TRACKED_LANDMARKS - re-extracting")
        pose_df = pose.extract_pose(video_path, config)
        pose_df.to_csv(paths.pose, index=False)
        _report("pose", "wrote", paths.pose)
    print(f"           {pose_df['frame'].nunique()} frames with a detected person")

    # 5: contact detection
    if "contacts" not in forced and os.path.exists(paths.contacts):
        with open(paths.contacts, "r", encoding="utf-8") as f:
            contacts = json.load(f)
        _report("contacts", "using", paths.contacts)
    else:
        contacts = contact_detection.detect_contacts(
            pose_df, calibrated_holds, climb, hold_type_map, config
        )
        with open(paths.contacts, "w", encoding="utf-8") as f:
            json.dump(contacts, f, indent=2)
        _report("contacts", "wrote", paths.contacts)

    # 6 + 6b: attempt summary (movement metrics + move segmentation), plus the
    # raw per-frame metrics alongside it, for checking the numbers the summary
    # is derived from
    if "metrics" not in forced and os.path.exists(paths.metrics):
        with open(paths.metrics, "r", encoding="utf-8") as f:
            summary = json.load(f)
        _report("metrics", "using", paths.metrics)
    else:
        summary = metrics.build_attempt_summary(
            pose_df, contacts, climb, config, calibrated_holds
        )
        with open(paths.metrics, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        _report("metrics", "wrote", paths.metrics)

    if "metrics" in forced or not os.path.exists(paths.frame_metrics):
        metrics.compute_frame_metrics(pose_df, config).to_csv(paths.frame_metrics, index=False)
        _report("metrics", "wrote", paths.frame_metrics)

    counts = summary["move_counts"]
    print(
        f"           {counts['raw_contact_events']} raw contacts -> "
        f"{counts['held_positions_after_cleaning']} held positions -> "
        f"{counts['moves']} moves"
    )

    if not run_coach:
        return summary, None

    # 7: step-by-step coaching analysis
    if "coach" not in forced and os.path.exists(paths.coaching):
        with open(paths.coaching, "r", encoding="utf-8") as f:
            analysis = coach.AttemptAnalysis.model_validate(json.load(f))
        _report("coach", "using", paths.coaching)
    else:
        analysis = coach.get_coaching_feedback(summary, config)
        with open(paths.coaching, "w", encoding="utf-8") as f:
            json.dump(analysis.model_dump(), f, indent=2)
        _report("coach", "wrote", paths.coaching)

    return summary, analysis


def _main():
    parser = argparse.ArgumentParser(
        description="Analyze one climbing video against one Kilter climb, end to end."
    )
    parser.add_argument("video_path", help="Path to the climb video")
    parser.add_argument("climb_uuid", help="Kilter climb uuid")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--force",
        action="append",
        choices=(*STAGES, "all"),
        default=[],
        metavar="STAGE",
        help=(
            "Re-run this stage and every stage after it (repeatable). "
            f"One of: {', '.join(STAGES)}, all"
        ),
    )
    parser.add_argument(
        "--no-coach",
        action="store_true",
        help="Stop after the attempt summary - no Anthropic API call, no cost",
    )
    args = parser.parse_args()

    if not os.path.exists(args.video_path):
        # Worth an explicit check: most stages use this path only to *name*
        # their cache files, so a wrong path otherwise gets a long way in
        # before the one stage that actually opens the video fails.
        parser.error(f"video not found: {args.video_path}")

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    summary, analysis = run_pipeline(
        args.video_path,
        args.climb_uuid,
        config,
        forced=resolve_forced(args.force),
        run_coach=not args.no_coach,
    )

    if analysis is None:
        print("\nStopped before the coaching stage (--no-coach).")
        return

    print()
    coach.print_analysis(analysis)


if __name__ == "__main__":
    _main()

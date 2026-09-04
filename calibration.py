"""
Click-to-label calibration: maps each hold used in a specific climb to its
pixel location in a specific video.

No homography and no attempt to project positions for holds that weren't
directly clicked - a climb typically uses 8-20 holds, so direct labeling in
the video's own pixel space is simpler and more accurate than deriving a
board-wide coordinate transform.

Workflow:
  1. Scrub through the video to pick one representative frame where all of
     the climb's used holds are visible (camera is assumed static for the
     whole attempt, so any frame works as long as holds aren't hidden by the
     climber's body). Kilter boards illuminate the holds used in a climb, so
     a frame from just before the climber steps on (LEDs lit, body not yet
     blocking anything) is usually the easiest to calibrate from.
  2. Click each hold, one at a time, in the order it's prompted. Each prompt
     shows the hold's actual LED color (from the board database) as a
     swatch, to help you find the right physical hold when several are
     close together.
  3. Save {hole_id: [pixel_x, pixel_y]} to cache/<video_name>_holds.json.
"""

import json
import os

import cv2
import numpy as np

ROLE_ORDER = {"start": 0, "middle": 1, "foot": 2, "finish": 3}
MINIMAP_SIZE = 320
MINIMAP_MARGIN = 24


def _hex_to_bgr(hex_color):
    """Convert an 'RRGGBB' hex string (as stored in placement_roles.led_color) to a BGR tuple for cv2."""
    if not hex_color or len(hex_color) != 6:
        return (255, 255, 255)
    r, g, b = (int(hex_color[i : i + 2], 16) for i in (0, 2, 4))
    return (b, g, r)


def _cache_path(video_path, cache_dir="cache"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(cache_dir, f"{video_name}_holds.json")


def load_cached_holds(video_path, cache_dir="cache"):
    path = _cache_path(video_path, cache_dir)
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)
    return {int(hole_id): tuple(xy) for hole_id, xy in raw.items()}


def _order_placements(placements):
    return sorted(
        placements,
        key=lambda p: (ROLE_ORDER.get(p["role"], 99), p["board_y"], p["hole_id"]),
    )


def select_frame(video_path):
    """
    Interactively scrub the video and pick one representative frame.

    Controls: 'd'/right = next frame, 'a'/left = previous frame,
    'D'/'A' = jump 30 frames, 'c'/Enter = confirm this frame, 'q'/Esc = abort.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))

    index = frame_count // 2
    window = "Select a frame (a/d step, A/D jump, c=confirm, q=quit)"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    frame = None
    while True:
        index = max(0, min(frame_count - 1, index))
        capture.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, frame = capture.read()
        if not ok:
            break

        display = frame.copy()
        cv2.putText(
            display,
            f"frame {index}/{frame_count - 1}  a/d step  A/D jump30  c=confirm  q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
        )
        cv2.imshow(window, display)
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("q"), 27):
            capture.release()
            cv2.destroyWindow(window)
            raise KeyboardInterrupt("Frame selection aborted")
        elif key in (ord("c"), 13):
            break
        elif key == ord("d"):
            index += 1
        elif key == ord("a"):
            index -= 1
        elif key == ord("D"):
            index += 30
        elif key == ord("A"):
            index -= 30

    capture.release()
    cv2.destroyWindow(window)
    return frame


def _build_minimap(placements, target_hole_id, labeled_hole_ids, size=MINIMAP_SIZE):
    """
    Draw a schematic reference panel showing every hold's relative position
    (from board_x/board_y), colored by role, with the current target hold
    highlighted. This disambiguates holds that share a role/LED color (e.g.
    several foot holds) by relative position in the pattern, without needing
    a full board-to-video coordinate transform - it's just a proportional
    scatter of the same coordinates already stored on each placement.
    """
    xs = [p["board_x"] for p in placements]
    ys = [p["board_y"] for p in placements]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1)
    span_y = max(max_y - min_y, 1)

    panel = np.full((size, size, 3), 30, dtype=np.uint8)
    cv2.putText(
        panel,
        "reference layout (relative positions)",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
    )

    def to_panel_xy(board_x, board_y):
        px = MINIMAP_MARGIN + (board_x - min_x) / span_x * (size - 2 * MINIMAP_MARGIN)
        # invert y: board_y increases upward, image y increases downward
        py = size - MINIMAP_MARGIN - (board_y - min_y) / span_y * (size - 2 * MINIMAP_MARGIN)
        return int(px), int(py)

    for p in placements:
        px, py = to_panel_xy(p["board_x"], p["board_y"])
        color = _hex_to_bgr(p.get("role_led_color"))
        if p["hole_id"] == target_hole_id:
            continue
        radius = 9 if p["hole_id"] in labeled_hole_ids else 6
        thickness = -1 if p["hole_id"] in labeled_hole_ids else 2
        cv2.circle(panel, (px, py), radius, color, thickness)

    target = next(p for p in placements if p["hole_id"] == target_hole_id)
    tx, ty = to_panel_xy(target["board_x"], target["board_y"])
    target_color = _hex_to_bgr(target.get("role_led_color"))
    cv2.circle(panel, (tx, ty), 16, (255, 255, 255), 3)
    cv2.circle(panel, (tx, ty), 10, target_color, -1)
    cv2.putText(
        panel,
        "<- THIS ONE",
        (min(tx + 20, size - 140), ty + 5),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        2,
    )

    return panel


def label_holds(frame, placements):
    """
    Prompt the user to click each placement's hold, one at a time.

    Controls: click = label current hold, 'u' = undo last label,
    's' = skip current hold (it will be excluded from contact detection),
    'q'/Esc = stop early (already-labeled holds are still returned).
    """
    ordered = _order_placements(placements)
    labeled = {}
    state = {"index": 0, "click": None}

    window = "Click each hold (u=undo, s=skip, q=quit)"
    minimap_window = "Reference layout"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.namedWindow(minimap_window, cv2.WINDOW_NORMAL)

    def on_mouse(event, x, y, flags, userdata):
        if event == cv2.EVENT_LBUTTONDOWN:
            state["click"] = (x, y)

    cv2.setMouseCallback(window, on_mouse)

    while state["index"] < len(ordered):
        placement = ordered[state["index"]]
        display = frame.copy()

        for hole_id, (x, y) in labeled.items():
            cv2.circle(display, (x, y), 6, (0, 255, 0), 2)

        swatch_color = _hex_to_bgr(placement.get("role_led_color"))
        cv2.circle(display, (20, 55), 12, swatch_color, -1)
        cv2.circle(display, (20, 55), 12, (255, 255, 255), 1)

        cv2.putText(
            display,
            f"Click hole_id {placement['hole_id']} ({placement['role']})  "
            f"[{state['index'] + 1}/{len(ordered)}]  u=undo s=skip q=quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 0, 255),
            2,
        )
        cv2.putText(
            display,
            "look for the hold lit this color",
            (40, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 255),
            2,
        )
        minimap = _build_minimap(ordered, placement["hole_id"], set(labeled.keys()))

        cv2.imshow(window, display)
        cv2.imshow(minimap_window, minimap)
        key = cv2.waitKey(20) & 0xFF

        if state["click"] is not None:
            labeled[placement["hole_id"]] = state["click"]
            state["click"] = None
            state["index"] += 1
        elif key == ord("u"):
            if state["index"] > 0:
                state["index"] -= 1
                prev_hole_id = ordered[state["index"]]["hole_id"]
                labeled.pop(prev_hole_id, None)
        elif key == ord("s"):
            state["index"] += 1
        elif key in (ord("q"), 27):
            break

    cv2.destroyWindow(window)
    cv2.destroyWindow(minimap_window)
    return labeled


def calibrate(video_path, climb, cache_dir="cache", force=False):
    """
    Return {hole_id: (pixel_x, pixel_y)} for every labeled hold used in `climb`.

    Reuses cache/<video_name>_holds.json if present, unless force=True.
    """
    if not force:
        cached = load_cached_holds(video_path, cache_dir)
        if cached is not None:
            return cached

    frame = select_frame(video_path)
    labeled = label_holds(frame, climb["placements"])

    os.makedirs(cache_dir, exist_ok=True)
    path = _cache_path(video_path, cache_dir)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({str(hole_id): list(xy) for hole_id, xy in labeled.items()}, f, indent=2)

    return labeled


def _main():
    import argparse
    import json as _json

    import yaml

    import kilter_data

    parser = argparse.ArgumentParser(description="Stand-alone calibration runner")
    parser.add_argument("video_path", help="Path to the climb video")
    parser.add_argument("climb_uuid", help="Climb uuid (cache/<uuid>.json must already exist)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--force", action="store_true", help="Re-calibrate even if cached")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cache_path = os.path.join(os.path.dirname(config["db_path"]), f"{args.climb_uuid}.json")
    with open(cache_path, "r", encoding="utf-8") as f:
        climb = _json.load(f)

    labeled = calibrate(
        args.video_path,
        climb,
        cache_dir=os.path.dirname(config["db_path"]),
        force=args.force,
    )
    print(f"Labeled {len(labeled)}/{len(climb['placements'])} holds:")
    for hole_id, (x, y) in labeled.items():
        print(f"  hole_id {hole_id}: ({x}, {y})")


if __name__ == "__main__":
    _main()

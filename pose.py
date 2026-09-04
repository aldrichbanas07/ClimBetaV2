"""
Runs MediaPipe Pose (per frame) over a climb video and extracts wrist,
ankle, and toe landmark pixel positions.

mediapipe 1.0+ replaced the old `mediapipe.solutions.pose` API with the
Tasks API (`mediapipe.tasks.python.vision.PoseLandmarker`), which needs a
downloadable .task model bundle rather than a bundled model - confirmed by
inspecting the installed mediapipe package directly (1.0.1 has no
`solutions` attribute at all). The model is fetched once from Google's
official MediaPipe model bucket and cached locally.

Landmark coverage: MediaPipe's Pose model (BlazePose, 33 landmarks) has
wrist, ankle, heel, and foot_index (toe) landmarks, but no per-finger
landmarks - that requires the separate, heavier hand-landmark model (via
HolisticLandmarker). Per the "if available" wording in the spec, this
module extracts what Pose actually provides: wrist (as the hand-contact
proxy) and both ankle + foot_index (toe) for feet.
"""

import os

import cv2
import pandas as pd
import requests

MODEL_URLS = {
    "lite": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/latest/pose_landmarker_lite.task",
    "full": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task",
    "heavy": "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task",
}

# name -> mediapipe.tasks.python.vision.PoseLandmark member name
TRACKED_LANDMARKS = {
    "left_wrist": "LEFT_WRIST",
    "right_wrist": "RIGHT_WRIST",
    "left_ankle": "LEFT_ANKLE",
    "right_ankle": "RIGHT_ANKLE",
    "left_toe": "LEFT_FOOT_INDEX",
    "right_toe": "RIGHT_FOOT_INDEX",
}


def ensure_pose_model(config):
    model_path = config["pose_model_path"]
    if os.path.exists(model_path):
        return model_path

    variant = config.get("pose_model_variant", "full")
    url = MODEL_URLS[variant]
    os.makedirs(os.path.dirname(model_path) or ".", exist_ok=True)
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    with open(model_path, "wb") as f:
        f.write(response.content)
    return model_path


def extract_pose(video_path, config):
    """
    Run pose estimation over every frame of `video_path`.

    Returns a tidy (long-form) DataFrame with columns:
        frame, timestamp_s, landmark, x_px, y_px, visibility, presence
    One row per tracked landmark per frame in which a person was detected.
    Frames with no detected person contribute no rows.
    """
    # Imported lazily: mediapipe's Tasks API pulls in a fair amount, and this
    # keeps kilter_data/calibration usable without a full mediapipe install.
    from mediapipe.tasks.python import vision
    from mediapipe.tasks.python.core.base_options import BaseOptions
    import mediapipe as mp

    model_path = ensure_pose_model(config)

    landmark_indices = {
        name: getattr(vision.PoseLandmark, enum_name).value
        for name, enum_name in TRACKED_LANDMARKS.items()
    }

    options = vision.PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=vision.RunningMode.VIDEO,
        num_poses=1,
    )

    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0

    rows = []
    frame_index = 0
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while True:
            ok, frame_bgr = capture.read()
            if not ok:
                break

            timestamp_ms = int(frame_index / fps * 1000)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.pose_landmarks:
                pose = result.pose_landmarks[0]
                height, width = frame_bgr.shape[:2]
                for name, index in landmark_indices.items():
                    lm = pose[index]
                    rows.append(
                        {
                            "frame": frame_index,
                            "timestamp_s": timestamp_ms / 1000.0,
                            "landmark": name,
                            "x_px": lm.x * width,
                            "y_px": lm.y * height,
                            "visibility": lm.visibility,
                            "presence": lm.presence,
                        }
                    )

            frame_index += 1

    capture.release()
    return pd.DataFrame(
        rows,
        columns=["frame", "timestamp_s", "landmark", "x_px", "y_px", "visibility", "presence"],
    )


LANDMARK_COLORS_BGR = {
    "left_wrist": (0, 0, 255),
    "right_wrist": (0, 128, 255),
    "left_ankle": (255, 0, 0),
    "right_ankle": (255, 128, 0),
    "left_toe": (0, 255, 255),
    "right_toe": (0, 255, 0),
}


def render_overlay(video_path, pose_df, out_path, visibility_threshold=0.5):
    """
    Write an annotated copy of `video_path` with each tracked landmark drawn
    as a colored dot (only when visibility >= visibility_threshold, so
    low-confidence extrapolated positions aren't drawn), for visually
    sanity-checking pose extraction against the source footage.
    """
    capture = cv2.VideoCapture(video_path)
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    by_frame = {
        frame: group for frame, group in pose_df.groupby("frame")
    }

    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break

        group = by_frame.get(frame_index)
        if group is not None:
            for _, row in group.iterrows():
                if row["visibility"] < visibility_threshold:
                    continue
                x, y = int(row["x_px"]), int(row["y_px"])
                if 0 <= x < width and 0 <= y < height:
                    color = LANDMARK_COLORS_BGR.get(row["landmark"], (255, 255, 255))
                    cv2.circle(frame, (x, y), 6, color, -1)
                    cv2.putText(
                        frame,
                        row["landmark"],
                        (x + 8, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        color,
                        1,
                    )

        writer.write(frame)
        frame_index += 1

    capture.release()
    writer.release()


def _main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Stand-alone pose extraction runner")
    parser.add_argument("video_path", help="Path to the climb video")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None, help="CSV output path (default: cache/<video_name>_pose.csv)")
    parser.add_argument(
        "--visualize",
        default=None,
        help="Also write an annotated copy of the video with tracked keypoints drawn on it, to this path",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    video_name = os.path.splitext(os.path.basename(args.video_path))[0]

    df = extract_pose(args.video_path, config)

    out_path = args.out
    if out_path is None:
        out_path = os.path.join(os.path.dirname(config["db_path"]), f"{video_name}_pose.csv")
    df.to_csv(out_path, index=False)

    frames_with_detections = df["frame"].nunique()
    print(f"Extracted pose for {frames_with_detections} frames -> {out_path}")
    print(df.head(12))

    if args.visualize:
        render_overlay(args.video_path, df, args.visualize)
        print(f"Wrote annotated video -> {args.visualize}")


if __name__ == "__main__":
    _main()

# ClimBeta

Analyzes a climber's ascent of a specific Kilterboard climb from video,
using the climb's known hold layout (fetched from the Kilter board
database) instead of visual hold detection, and produces coaching feedback
and beta via a cloud LLM!!!!

Target board: Kilter Original 12x12 layout. Assumes a static, fixed camera
for the full duration of the recorded attempt - if the camera moves
mid-climb, this approach doesn't hold up.

## Demo: contact detection overlay

A rendered example showing detected hold contacts (bright ring = a limb
currently touching that hold, labeled by limb name) over a real test climb:


https://github.com/user-attachments/assets/dda35a1b-40b6-467d-a6f9-b5e421dc4259

## Ground truth on the data

The Kilter database (fetched via `boardlib`) only contains hole position,
placement role (start/hand/foot/finish), and board angle. It does **not**
contain hold shape type (jug/crimp/sloper/pinch) - that field does not
exist in the real data. Hold "type" in this project is a deliberate
placeholder (`hole_id % 4`, see `hold_type_map.py`), used only so the same
physical hold gets a consistent label across a session - never presented as
a verified physical hold shape.

## Pipeline

| Step | File | Status |
|---|---|---|
| 1 | `kilter_data.py` - fetch/cache a climb's placements + angle from the Aurora database | done |
| 2 | `hold_type_map.py` - one-time placeholder hold-type table for every hole on the layout | done |
| 3 | `calibration.py` - click-to-label calibration (hole -> pixel position in one video) | done |
| 4 | `pose.py` - MediaPipe Pose extraction (wrist/ankle/toe) per frame | done |
| 5 | `contact_detection.py` - match pose keypoints to calibrated holds into contact events | done |
| 6 | `metrics.py` - movement smoothness metrics joined with contact events | done |
| 6b | `moves.py` - segments the raw contact stream into discrete moves with body-position features | done |
| 7 | `coach.py` - step-by-step coaching analysis via the Anthropic API | done |
| 8 | `main.py` - CLI entry point wiring steps 1-7 together | done |

## The move-by-move analysis

The coaching output is a walkthrough: one entry per move, in order, each
saying what happened, what the body-position numbers show, how the
execution looked (`smooth` / `rushed` / `hesitant` / `fumbled` / `unclear`)
and the numbers behind that rating - then overall strengths, issues and
drills.

That needed a segmentation layer (`moves.py`), because contact detection
deliberately over-detects: on the test footage it produced **177 contact
events for a 16-hold climb**, mostly 0.1-0.2s flybys plus the ankle and toe
of the same foot each registering separately. `moves.py` collapses
ankle+toe into one foot, merges re-contacts on the same hold into one
occupancy (counting the gap as a regrip), drops flybys shorter than
`min_hold_time_s`, and cuts the sequence at the finish hold so the climber
dropping off the wall isn't analysed as climbing. 177 raw events becomes
**34 moves**.

The `execution` rating is an enum of observable movement qualities rather
than a good/bad verdict, and the model has to cite the number behind each
one. That's the honest version of "was this move bad": the data can show a
move was abrupt, slow to commit, or regripped repeatedly - it cannot show
that a different body position would have been better.

## Usage

```
python main.py <video_path> <climb_uuid>
```

That's the whole pipeline. Each stage caches its output under `cache/` and is
skipped when the cache file is there, so a second run is instant and free -
useful because stage 7 is the only one that costs money.

```
--force <stage>   re-run this stage and everything after it (repeatable):
                  climb | holds | pose | contacts | metrics | coach | all
--no-coach        stop after the attempt summary - no API call, no cost
```

`--force` deliberately cascades downstream: a metrics file built on an old
contact stream is worse than no cache at all, because it fails silently
rather than loudly.

Two things it writes beyond the stage caches: `cache/<video>_coaching.json`
(the analysis, also printed) and `cache/<video>_frame_metrics.csv` (raw
per-frame speed, trailing average, velocity ratio and cumulative distance
for every tracked landmark - the numbers the summary is derived from).

Stage 3 is interactive (it opens OpenCV windows for click-to-label
calibration), so its cache is checked before any of the expensive stages
start. Stage 7 needs `ANTHROPIC_API_KEY` set and credit on the account.

## Setup

```
uv venv --python 3.11 .venv
uv pip install --python .venv -r requirements.txt
```

`opencv-contrib-python` (not `opencv-python`) is required - `mediapipe`
depends on the contrib build, and having both installed at once is a known
conflict that silently breaks the GUI windows used by calibration.

Config lives in `config.yaml`: board/layout selection, calibration
tolerances (`contact_radius_px`, `dwell_frames`, `min_visibility`), the pose
model variant, and the Anthropic API key's environment variable name.

## Running the stages standalone

`main.py` covers the normal path; each stage is also runnable on its own, for
inspecting or re-doing one piece without the others.

```
# 1+2: fetch a climb's layout (run once per climb; hold_type_map.json is one-time, project-wide)
python -c "import yaml, kilter_data; c = yaml.safe_load(open('config.yaml')); kilter_data.fetch_climb(c, '<climb_uuid>')"
python hold_type_map.py

# 3: calibrate hold pixel positions for one specific video
python calibration.py <video_path> <climb_uuid>

# 4: extract pose landmarks
python pose.py <video_path> [--visualize <output_mp4>]

# 5: detect hold contacts
python contact_detection.py <video_path> <climb_uuid> [--visualize <output_mp4>]

# 6 (+6b): movement metrics, move segmentation and body-position features,
#          all written into one attempt summary
python metrics.py <video_path> <climb_uuid>

# 6b alone, to inspect just the move segmentation
python moves.py <video_path>

# 7: step-by-step coaching analysis (needs ANTHROPIC_API_KEY and API credit)
python coach.py <video_path>
```

Changing the tracked pose landmarks (`pose.py`) invalidates the cached
`cache/<video_name>_pose.csv` - re-run step 4 before steps 5-7, or the
body-position features come back null. `main.py` detects this case and
re-extracts automatically; running the stages by hand does not.

### When a climb isn't in the cached database snapshot

The local database (`cache/kilter_db.sqlite3`) is a snapshot baked into the
Kilter app's last release, not a live sync - newer community-set climbs
won't be in it, and if the live Aurora API is unreachable from your network,
there's no way to fetch them normally. `kilter_data.build_manual_climb()`
is a fallback for this: transcribe the climb by hand from a screenshot in
the app (which lights up the used holds by role - green/cyan/orange/magenta
for start/middle/foot/finish), and it builds a climb record in the exact
shape `fetch_climb()` would have produced. See `manual_climbs/` for a real
example.

## Explicitly out of scope

- Full-board homography or automatic LED-based calibration
- A RAG/retrieval knowledge base of climbing technique
- Any backend service, task queue, or multi-user database
- Comparison against other climbers' ascents
- Handling camera movement mid-recording

## Is this a climbing ML model?

No. The only trained ML model involved is MediaPipe Pose, and it's a
general-purpose human pose estimator with no climbing-specific knowledge.
Contact detection and metrics are plain deterministic geometry (distance
thresholds, frame counting). The coach (step 7) uses a general-purpose LLM
to narrate structured numbers, not a model trained on climbing technique or
biomechanics - which is exactly why its system prompt has to be explicit
about not asserting confident physical claims the underlying data can't
actually support.

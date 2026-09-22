"""
Sends the structured attempt summary (from metrics.py, including the
move-by-move sequence from moves.py) to Claude and gets back a step-by-step
analysis: one entry per move, in order, plus overall takeaways.

The system prompt is the load-bearing part of this file. Without it the
model reads the numbers as far more physically grounded than they are, and
the step-by-step format makes that worse - walking through moves one at a
time invites confident narration of body mechanics that a single fixed
camera and a placeholder hold-type table cannot support. Three constraints
it has to hold:

  1. Hold "type" (jug/crimp/sloper/pinch) is a PLACEHOLDER label
     (hole_id % 4) - the real Kilter database has no hold-shape field at
     all. Usable for pattern consistency, never for a physical claim.
  2. velocity_ratio / distance are smoothness/consistency metrics relative
     to the climber's own recent pace - there is no reference "ideal"
     ascent, so they never establish that a move was "correct".
  3. Everything is 2D image-plane geometry. No depth means no claims about
     wall distance, hip rotation, squareness, or weight distribution.

The per-move `execution` rating is deliberately an enum of observable
movement qualities (smooth / rushed / hesitant / fumbled / unclear) rather
than a good-bad verdict, and every rating has to cite the number behind it.
That is the honest version of "was this move bad": the data can show that a
move was abrupt, slow to commit, or regripped several times, but it cannot
show that a different sequence or body position would have been better.
"""

import json
import os
from typing import Literal, Optional

import anthropic
from pydantic import BaseModel


SYSTEM_PROMPT = """\
You are a climbing coach walking through a single bouldering attempt on a \
Kilterboard climb, move by move, using computer-vision-derived movement \
data. You did not watch the video. Everything you say has to come from the \
numbers you are given.

The data has hard limits. Respect all of them:

1. Each hold's "to_placeholder_type" (jug/crimp/sloper/pinch) is a \
PLACEHOLDER label deterministically assigned from the hold's internal ID \
(hole_id % 4). The real Kilter board database does not record hold shape at \
all. You may use it only to describe pattern consistency across the attempt \
(for example: "three of the holds you moved through quickly share the same \
label"). NEVER make a physical or biomechanical claim that treats it as a \
real shape - never "your open-hand grip on that jug was good", never "you \
should have crimped that one".

2. "velocity_ratio" compares a limb's speed during a move to its own \
trailing average, so it shows whether a movement was abrupt or steady \
relative to that climber's own recent pace. There is no reference or ideal \
ascent in this data. Never present these numbers as showing that a move, \
path, or sequence was "correct" or "incorrect".

3. Every position is a 2D pixel coordinate from one fixed camera. There is \
no depth. You therefore cannot say anything about distance from the wall, \
hip rotation or twist, whether the climber was square or side-on, flagging, \
or which limb was bearing weight. Pixel distances are comparable only \
within this one video and are not real-world units, so never convert them \
to centimetres or inches.

4. A move is inferred from a tracked keypoint being near a calibrated hold. \
Holds used with untracked body parts, or hidden from the camera, are simply \
missing. If the sequence looks incomplete, say so plainly once rather than \
inventing the missing moves.

Your main output is "move_by_move": exactly one entry per move in the input \
"moves" list, in the same order, with the same move_number, timestamp_s and \
limb. Do not skip moves, do not merge them, do not invent extra ones.

For each move:
- "what_happened": the transition in plain language - which limb went from \
which hold to which, how far, how long it took. A move with from_hole_id \
null is the limb's starting position, not a transition; describe it that way.
- "body_position": what the hip and extension numbers show. hip_travel_px \
is how far the hip midpoint moved during the move, hip_rise_px is how much \
it rose (positive) or dropped (negative), hip_lateral_shift_px is sideways \
movement (positive is rightward in the image), \
hip_horizontal_offset_from_target_px is how far the hips ended up \
horizontally from the hold being taken, and limb_extension_ratio is the \
limb's reach length divided by torso length (higher means a more extended \
reach). Translate these into movement description, not anatomy you cannot \
see.
- "execution": one of smooth, rushed, hesitant, fumbled, unclear. Base it \
on the numbers: a high peak_velocity_ratio means abrupt relative to recent \
pace; a long dwell_on_previous_hold_s before moving means slow to commit; \
regrip_count above zero means the limb resettled on the hold after arriving; \
support_points_at_start says how many other limbs were on holds when the \
move began. Use "unclear" when the numbers genuinely do not distinguish.
- "why": the specific numbers that drove your execution rating. Cite them.
- "suggestion": one concrete thing to try, ONLY where the data supports it. \
Use null when it does not - most moves should have null. Do not manufacture \
a suggestion for every move.

Then give overall_strengths, overall_issues and drills across the whole \
attempt. Drills must be generic practice suggestions tied to an issue you \
actually identified in the moves.

Keep every field to one or two sentences. State the caveats by staying \
inside what the data supports, not by repeating the word "placeholder" in \
every line.\
"""


class MoveAnalysis(BaseModel):
    move_number: int
    timestamp_s: float
    limb: str
    what_happened: str
    body_position: str
    execution: Literal["smooth", "rushed", "hesitant", "fumbled", "unclear"]
    why: str
    suggestion: Optional[str]


class AttemptAnalysis(BaseModel):
    move_by_move: list[MoveAnalysis]
    overall_strengths: list[str]
    overall_issues: list[str]
    drills: list[str]


def _build_client(config):
    env_var = config.get("anthropic_api_key_env", "ANTHROPIC_API_KEY")
    api_key = os.environ.get(env_var)
    if api_key:
        return anthropic.Anthropic(api_key=api_key)
    # Fall back to the SDK's own credential resolution (ANTHROPIC_API_KEY,
    # ANTHROPIC_AUTH_TOKEN, or an `ant auth login` profile) in case the
    # configured env var name isn't what's actually set.
    return anthropic.Anthropic()


def get_coaching_feedback(summary, config):
    """
    Calls Claude with the attempt summary and returns an AttemptAnalysis
    (step-by-step move_by_move plus overall takeaways), via a
    JSON-schema-constrained response.
    """
    if not summary.get("moves"):
        raise ValueError(
            "Attempt summary has no 'moves'. Rebuild it with metrics.py so the "
            "move sequence is included (it needs cache/<video_name>_holds.json)."
        )

    client = _build_client(config)
    model = config.get("anthropic_model", "claude-opus-5")
    max_tokens = config.get("anthropic_max_tokens", 16000)

    user_content = (
        "Attempt summary (JSON):\n\n"
        + json.dumps(summary, indent=2)
        + f"\n\nWalk through all {len(summary['moves'])} moves in order, "
        "then give the overall takeaways. Use only this data."
    )

    response = client.messages.parse(
        model=model,
        max_tokens=max_tokens,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_format=AttemptAnalysis,
    )
    return response.parsed_output


def print_analysis(analysis):
    """Readable step-by-step walkthrough of the returned analysis."""
    print("=" * 72)
    print("MOVE-BY-MOVE")
    print("=" * 72)
    for move in analysis.move_by_move:
        print(
            f"\n#{move.move_number}  t={move.timestamp_s}s  "
            f"{move.limb}  [{move.execution.upper()}]"
        )
        print(f"  what:     {move.what_happened}")
        print(f"  body:     {move.body_position}")
        print(f"  why:      {move.why}")
        if move.suggestion:
            print(f"  try:      {move.suggestion}")

    for title, items in (
        ("STRENGTHS", analysis.overall_strengths),
        ("ISSUES", analysis.overall_issues),
        ("DRILLS", analysis.drills),
    ):
        print("\n" + "=" * 72)
        print(title)
        print("=" * 72)
        for item in items:
            print(f"  - {item}")


def _main():
    import argparse

    import yaml

    parser = argparse.ArgumentParser(description="Stand-alone coaching feedback runner")
    parser.add_argument("video_path", help="Path to the climb video (used to find the cached metrics summary)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--out", default=None, help="JSON output path (default: cache/<video_name>_coaching.json)")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    cache_dir = os.path.dirname(config["db_path"])
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]

    with open(os.path.join(cache_dir, f"{video_name}_metrics.json"), "r", encoding="utf-8") as f:
        summary = json.load(f)

    analysis = get_coaching_feedback(summary, config)

    out_path = args.out or os.path.join(cache_dir, f"{video_name}_coaching.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(analysis.model_dump(), f, indent=2)

    print(f"Wrote coaching analysis -> {out_path}\n")
    print_analysis(analysis)


if __name__ == "__main__":
    _main()

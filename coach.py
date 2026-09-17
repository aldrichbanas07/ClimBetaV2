"""
Sends the structured attempt summary (from metrics.py) plus climb metadata to
Claude and gets back structured coaching feedback: {strengths, issues, drills}.

The system prompt is the load-bearing part of this file. It must make the
model treat two things very carefully, or the output reads as more confident
and more physically grounded than the underlying data actually supports:

  1. Hold "type" (jug/crimp/sloper/pinch) is a PLACEHOLDER label
     (hole_id % 4) - the real Kilter database has no hold-shape field at all.
     The model may only use it for pattern consistency across the attempt
     (e.g. "you moved quickly through several type_b holds"), never for a
     confident physical/biomechanical claim that assumes the label is a real
     shape (e.g. "your open-hand grip on that jug was good").
  2. velocity_ratio / distance are movement smoothness/consistency metrics,
     relative to the climber's own recent pace - there is no reference
     "ideal" ascent, so they must never be framed as measuring whether the
     climber's path or technique was "correct".
"""

import json
import os

import anthropic
from pydantic import BaseModel


SYSTEM_PROMPT = """\
You are a climbing coach reviewing a single bouldering attempt on a \
Kilterboard climb, using computer-vision-derived movement data - you did not \
watch the video yourself, and the data has real limitations you must respect.

Data caveats (do not violate these):

1. Each contact's "placeholder_type" (jug/crimp/sloper/pinch) is a \
PLACEHOLDER label deterministically assigned from the hold's internal ID \
(hole_id % 4). The real Kilter board database does not record hold shape at \
all - this field is not a verified physical hold shape. You may use it only \
to describe pattern consistency across the attempt (for example: "you moved \
quickly through several sloper-labeled holds in a row"). You must NEVER make \
a confident physical or biomechanical claim that assumes this label is a \
real hold shape - for example, never say something like "your open-hand grip \
on that jug was good" or "you should have crimped that hold instead".

2. "velocity_ratio" and distance metrics describe movement smoothness and \
consistency only - how steady or jerky a limb's movement was relative to \
its own recent pace. There is no reference "ideal" ascent to compare \
against. Never describe these numbers as measuring whether the climber's \
path, sequence, or technique was "correct" - only whether it looked smooth \
or abrupt.

3. Ground every claim in the specific numbers provided (hold role, dwell \
time, approach velocity, limb, contact order). Do not invent details about \
grip style, body position, or technique that cannot be derived from this \
data.

Produce three short lists:
- strengths: what went well, grounded in the data (smooth sections, \
confident holds, good pacing)
- issues: what the data suggests could improve (jerky approaches, long \
hesitation, repeated foot adjustments)
- drills: concrete, generic practice suggestions a climber could try, \
tied to an issue you identified

Keep each entry to one or two sentences. Do not add disclaimers about data \
quality to every single entry - state the caveats once in spirit by staying \
within what the data supports, not by repeating "placeholder" in every line.\
"""


class CoachingFeedback(BaseModel):
    strengths: list[str]
    issues: list[str]
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
    Calls Claude with the attempt summary and returns a CoachingFeedback
    (strengths/issues/drills), via a JSON-schema-constrained response.
    """
    client = _build_client(config)
    model = config.get("anthropic_model", "claude-opus-5")

    user_content = (
        "Attempt summary (JSON):\n\n"
        + json.dumps(summary, indent=2)
        + "\n\nGive coaching feedback based only on this data."
    )

    response = client.messages.parse(
        model=model,
        max_tokens=2048,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_content}],
        output_format=CoachingFeedback,
    )
    return response.parsed_output


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

    feedback = get_coaching_feedback(summary, config)

    out_path = args.out or os.path.join(cache_dir, f"{video_name}_coaching.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(feedback.model_dump(), f, indent=2)

    print(f"Wrote coaching feedback -> {out_path}\n")
    print("Strengths:")
    for item in feedback.strengths:
        print(f"  - {item}")
    print("Issues:")
    for item in feedback.issues:
        print(f"  - {item}")
    print("Drills:")
    for item in feedback.drills:
        print(f"  - {item}")


if __name__ == "__main__":
    _main()

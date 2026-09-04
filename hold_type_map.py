"""
Builds hold_type_map.json: a static, deterministic PLACEHOLDER hold-type table
for every hole_id on the configured layout (Kilter Original 12x12).

The Aurora database does not contain hold shape data (jug/crimp/sloper/pinch) -
that field simply does not exist. This script assigns each hole_id a
placeholder type via `hole_id % 4`, purely so the same physical hold always
gets the same label across climbs, which lets metrics/coaching talk about
"pattern consistency" without pretending to know real hold shapes.

Run this ONCE:

    python hold_type_map.py

It refuses to overwrite an existing hold_type_map.json (pass --force to
regenerate) because that file is meant to be hand-edited later when real
hold-shape data replaces the placeholder - a script silently overwriting
those edits would destroy them.
"""

import argparse
import json
import os
import sqlite3

import yaml

import kilter_data

PLACEHOLDER_TYPES = ["jug", "crimp", "sloper", "pinch"]
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "hold_type_map.json")


def build_hold_type_map(config):
    db_path = kilter_data.ensure_database(config)
    with sqlite3.connect(db_path) as connection:
        layout_id = kilter_data._get_layout_id(connection, config["layout_name"])
        hole_ids = [
            row[0]
            for row in connection.execute(
                "SELECT DISTINCT hole_id FROM placements WHERE layout_id = ? ORDER BY hole_id",
                (layout_id,),
            )
        ]
    return {
        str(hole_id): PLACEHOLDER_TYPES[hole_id % 4] for hole_id in hole_ids
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite an existing hold_type_map.json (destroys manual edits)",
    )
    args = parser.parse_args()

    if os.path.exists(OUTPUT_PATH) and not args.force:
        print(
            f"{OUTPUT_PATH} already exists - refusing to overwrite "
            "(it may contain manual edits). Pass --force to regenerate."
        )
        return

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    hold_type_map = build_hold_type_map(config)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(hold_type_map, f, indent=2, sort_keys=True)

    print(f"Wrote {len(hold_type_map)} placeholder hold types to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()

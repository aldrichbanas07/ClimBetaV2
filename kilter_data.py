"""
Fetches a Kilterboard climb's hold layout (placements + roles + board angle)
from the Aurora Climbing backend via boardlib, and caches the result.

boardlib does not expose a "get climb" API call - the Aurora backend serves
placement/hole/role data only inside a bundled sqlite database (extracted
from the Android app's APK). This module downloads that database once
(caching it locally), then reads the following tables directly:

  climbs           - uuid, name, angle, frames (encodes which placements are used)
  placements       - id, layout_id, hole_id (a hole_id fixed to a layout+set)
  holes            - id, x, y (pixel/board coordinates within a layout's product)
  placement_roles  - id, name, full_name (e.g. start/middle/finish/foot)
  layouts          - id, name (e.g. "Kilter Board Original")

A climb's `frames` column is a string like "p1100r15p1103r15..." - repeated
"p<placement_id>r<role_id>" pairs. This was confirmed by inspecting the
downloaded sqlite db directly (see project notes), not assumed from memory.
"""

import datetime
import json
import os
import re
import sqlite3

import boardlib.db.aurora

FRAME_TOKEN_RE = re.compile(r"p(\d+)r(\d+)")


def _db_is_stale(db_path, refresh_days):
    if not os.path.exists(db_path):
        return True
    age = datetime.datetime.now() - datetime.datetime.fromtimestamp(
        os.path.getmtime(db_path)
    )
    return age > datetime.timedelta(days=refresh_days)


def ensure_database(config, force_refresh=False):
    """Download (or reuse) the cached Aurora sqlite database for `config['board']`."""
    db_path = config["db_path"]
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    if force_refresh or _db_is_stale(db_path, config.get("db_refresh_days", 7)):
        boardlib.db.aurora.download_database(config["board"], db_path)
    return db_path


def _get_layout_id(connection, layout_name):
    row = connection.execute(
        "SELECT id FROM layouts WHERE name = ?", (layout_name,)
    ).fetchone()
    if row is None:
        raise ValueError(f"No layout named {layout_name!r} found in the Aurora database")
    return row[0]


def search_climbs(config, name_query, limit=10):
    """Search climbs by (partial, case-insensitive) name on the configured layout."""
    db_path = ensure_database(config)
    with sqlite3.connect(db_path) as connection:
        layout_id = _get_layout_id(connection, config["layout_name"])
        rows = connection.execute(
            """
            SELECT uuid, name, angle
            FROM climbs
            WHERE layout_id = ? AND is_listed = 1 AND name LIKE ?
            ORDER BY name
            LIMIT ?
            """,
            (layout_id, f"%{name_query}%", limit),
        ).fetchall()
    return [{"uuid": uuid, "name": name, "angle": angle} for uuid, name, angle in rows]


def _parse_frames(frames):
    """Parse a climb's `frames` string into a list of (placement_id, role_id)."""
    return [(int(p), int(r)) for p, r in FRAME_TOKEN_RE.findall(frames)]


def fetch_climb(config, climb_uuid, angle_override=None, use_cache=True):
    """
    Fetch a climb's placement list (hole_id + pixel coords + role) and board angle.

    Returns a dict and caches the raw result to cache/<uuid>.json. Subsequent
    calls with use_cache=True load straight from that cache file instead of
    re-querying the database.
    """
    cache_path = os.path.join(os.path.dirname(config["db_path"]), f"{climb_uuid}.json")

    if use_cache and os.path.exists(cache_path):
        with open(cache_path, "r", encoding="utf-8") as f:
            climb = json.load(f)
        if angle_override is not None:
            climb["angle"] = angle_override
        return climb

    db_path = ensure_database(config)
    with sqlite3.connect(db_path) as connection:
        layout_id = _get_layout_id(connection, config["layout_name"])

        climb_row = connection.execute(
            "SELECT uuid, name, angle, frames, layout_id FROM climbs WHERE uuid = ?",
            (climb_uuid,),
        ).fetchone()
        if climb_row is None:
            raise ValueError(f"No climb with uuid {climb_uuid!r} found")
        uuid, name, angle, frames, climb_layout_id = climb_row
        if climb_layout_id != layout_id:
            raise ValueError(
                f"Climb {climb_uuid!r} belongs to layout_id={climb_layout_id}, "
                f"not the configured layout {config['layout_name']!r} (id={layout_id})"
            )

        placements = []
        for placement_id, role_id in _parse_frames(frames):
            placement_row = connection.execute(
                "SELECT hole_id FROM placements WHERE id = ? AND layout_id = ?",
                (placement_id, layout_id),
            ).fetchone()
            if placement_row is None:
                continue
            hole_id = placement_row[0]

            hole_row = connection.execute(
                "SELECT x, y FROM holes WHERE id = ?", (hole_id,)
            ).fetchone()
            if hole_row is None:
                continue
            x, y = hole_row

            role_row = connection.execute(
                "SELECT name, full_name FROM placement_roles WHERE id = ?", (role_id,)
            ).fetchone()
            role_name, role_full_name = role_row if role_row else (None, None)

            placements.append(
                {
                    "hole_id": hole_id,
                    "placement_id": placement_id,
                    "board_x": x,
                    "board_y": y,
                    "role_id": role_id,
                    "role": role_name,
                    "role_full_name": role_full_name,
                }
            )

        climb = {
            "uuid": uuid,
            "name": name,
            "angle": angle if angle_override is None else angle_override,
            "layout_id": layout_id,
            "layout_name": config["layout_name"],
            "placements": placements,
        }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "w", encoding="utf-8") as f:
        json.dump(climb, f, indent=2)

    return climb

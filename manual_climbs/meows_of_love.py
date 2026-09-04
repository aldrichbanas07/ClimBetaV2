"""
Manually transcribed from an in-app screenshot of "meows of love" by
kaleplusplus (V0, 40 degrees), because the climb is newer than the cached
Aurora database snapshot and the live API was unreachable from this network
(see project notes - looked like an ISP/carrier content filter blocking
kilterboardapp.com specifically, on both home broadband and mobile data).

Coordinates are pixel positions read off the screenshot (portrait phone
screen, board scrolled to its rightmost holds), with y inverted (H - y) so
ascending order matches the database's bottom-to-top board_y convention.
Roles were read off each hold's lit LED color: green=start, cyan=middle,
magenta=finish, yellow/orange=foot.

Run once to (re)populate cache/<uuid>.json if it's ever missing:

    python manual_climbs/meows_of_love.py
"""

import os
import sys

import yaml

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import kilter_data

UUID = "adc4780973934bc4868c932936bb14e5"
NAME = "meows of love"
ANGLE = 40

_SCREENSHOT_HEIGHT = 1652
_RAW = [
    ("finish", 711, 492),
    ("finish", 820, 547),
    ("middle", 876, 602),
    ("middle", 820, 712),
    ("middle", 820, 876),
    ("middle", 930, 876),
    ("start", 711, 986),
    ("start", 820, 986),
    ("foot", 820, 1041),
    ("foot", 930, 1041),
    ("foot", 876, 1096),
    ("foot", 766, 1206),
    ("foot", 930, 1206),
    ("foot", 930, 1261),
    ("foot", 820, 1371),
    ("foot", 930, 1371),
]

HOLDS = [
    {"role": role, "x": x, "y": _SCREENSHOT_HEIGHT - y} for role, x, y in _RAW
]


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(repo_root, "config.yaml"), "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    climb = kilter_data.build_manual_climb(config, UUID, NAME, ANGLE, HOLDS)
    print(f"Wrote cache/{UUID}.json with {len(climb['placements'])} placements")


if __name__ == "__main__":
    main()

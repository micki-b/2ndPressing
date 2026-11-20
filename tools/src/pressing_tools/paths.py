"""Where the private record library lives. These paths never go in git.

Set them in `tools/paths.local.json` (git-ignored), e.g.

    {
      "raw": "/Volumes/…/VinylStudio/…",
      "itunes": "/Volumes/…/iTunes/Media/Music",
      "protect_side": "/Volumes/…/VinylStudio/…/Artist~Album_[LP]~side1.m4a"
    }

or with environment variables, which win over the file:

    PRESSING_RAW_DIR        VinylStudio captures: one file per side + its .crd
    PRESSING_ITUNES_DIR     the de-clicked iTunes library
    PRESSING_PROTECT_SIDE   the raw side that protect_eval's test clip comes from

Looked up only when a script needs them, so imports and tests never fail.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2]
LOCAL = TOOLS / "paths.local.json"
_ENV = {
    "raw": "PRESSING_RAW_DIR",
    "itunes": "PRESSING_ITUNES_DIR",
    "protect_side": "PRESSING_PROTECT_SIDE",
}


def get(name: str) -> Path:
    """The configured path `name` (one of raw, itunes, protect_side)."""
    env = _ENV[name]
    if os.environ.get(env):
        return Path(os.environ[env]).expanduser()
    if LOCAL.is_file():
        value = json.loads(LOCAL.read_text()).get(name)
        if value:
            return Path(value).expanduser()
    raise SystemExit(f"Set {env}, or \"{name}\" in {LOCAL}, to the location of your library.")


def raw_dir() -> Path:
    return get("raw")


def itunes_dir() -> Path:
    return get("itunes")

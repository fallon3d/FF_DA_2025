from __future__ import annotations
from typing import Dict, Any, List, Optional
import httpx

from .utils import slot_for_round_pick  # used when slot is missing but (round,pick_no) exist

# -----------------------------
# HTTP basics
# -----------------------------

BASE = "https://api.sleeper.app/v1"
_UA = {"User-Agent": "FF_DA_2025 (github.com/fallon3d/FF_DA_2025)"}


def _get(path: str, timeout: float = 20.0) -> Any:
    """
    Lightweight GET wrapper around Sleeper's public API.
    Raises httpx.HTTPStatusError on non-2xx.
    """
    url = f"{BASE}{path}"
    with httpx.Client(timeout=timeout, headers=_UA) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()

# -----------------------------
# Normalizers
# -----------------------------

def _normalize_picks(picks) -> List[dict]:
    """
    Coerce Sleeper picks into list[dict] reliably:

    Accepts:
      - A dict wrapper with keys like 'picks'/'draft_picks'/'data'
      - A direct list of pick dicts
      - A list that may (rarely) contain bare player_id strings (wrap them)

    Returns a uniform list[dict] where each item at least has:
      - 'player_id' (optional)
      - 'metadata' (dict, possibly empty)
      - optional: 'round', 'pick_no', 'slot', 'roster_id'
    """
    # Unwrap dict wrappers first
    if isinstance(picks, dict):
        for key in ("picks", "draft_picks", "data"):
            val = picks.get(key)
            if isinstance(val, list):
                picks = val
                break
        else:
            return []

    # If not a list by now, bail
    if not isinstance(picks, list):
        return []

    out: List[dict] = []
    for p in picks:
        if isinstance(p, dict):
            # Ensure metadata is a dict
            if not isinstance(p.get("metadata"), dict):
                p = {**p, "metadata": {}}
            out.append(p)
        elif isinstance(p, str):
            # Rare: some sources return bare player_ids
            out.append({"player_id": p, "metadata": {}})
        else:
            # Unknown shape; skip
            continue
    return out

# -----------------------------
# Core Sleeper endpoints
# -----------------------------

def get_league_info(league_id: str) -> Dict[str, Any]:
    """
    League metadata: name, roster_positions, scoring_settings, draft_id, total_rosters, etc.
    """
    return _get(f"/league/{league_id}")

def get_drafts_for_league(league_id: str) -> List[dict]:
    """
    All drafts associated with the league (current + past).
    """
    return _get(f"/league/{league_id}/drafts")

def get_draft(draft_id: str) -> Dict[str, Any]:
    """
    Single draft metadata (settings: teams, rounds; status: pre_draft, in_progress, complete).
    """
    return _get(f"/draft/{draft_id}")

def get_picks(draft_id: str) -> List[dict]:
    """
    Picks for a draft, normalized to list[dict].
    """
    raw = _get(f"/draft/{draft_id}/picks")
    return _normalize_picks(raw)

def get_users(league_id: str) -> List[dict]:
    """
    Users (managers) in the league; includes display_name, user_id, and often roster_id.
    """
    return _get(f"/league/{league_id}/users")

def get_players_nfl() -> Dict[str, Any]:
    """
    Full NFL players dump (large JSON). Keyed by Sleeper player_id (string).
    Each entry can include:
      first_name, last_name, full_name, position, team, injury_status, etc.
    Fetch once per session and cache at the app layer.
    """
    return _get("/players/nfl")

# -----------------------------
# Helpers (URLs, naming, logs)
# -----------------------------

def parse_draft_id_from_url(url_or_id: str) -> Optional[str]:
    """
    Accepts:
      - Full URLs (sleeper.com or sleeper.app), with or without '/nfl' segment
      - Bare draft_id (all digits, length >= 10)
    Returns the draft_id string, or None if not found.
    """
    if not url_or_id:
        return None
    s = url_or_id.strip()

    # Bare id
    if s.isdigit() and len(s) >= 10:
        return s

    parts = s.strip("/").split("/")
    # Look for ".../draft/<maybe nfl>/<id>"
    for i, p in enumerate(parts):
        if p.lower() == "draft":
            # next is possibly 'nfl' then id
            if i + 2 < len(parts) and parts[i + 1].lower() == "nfl":
                return parts[i + 2]
            if i + 1 < len(parts):
                return parts[i + 1]

    # Fallback: last numeric token
    for p in reversed(parts):
        if p.isdigit() and len(p) >= 10:
            return p
    return None


def picked_player_names(picks: List[dict], players_map: Dict[str, Any]) -> List[str]:
    """
    Resolve picked player names from normalized picks.

    Prefers the official players map (player_id → full_name).
    Falls back to pick metadata if the id is missing/unrecognized.
    """
    out: List[str] = []
    for p in _normalize_picks(picks):
        meta = p.get("metadata") or {}
        pid = str(p.get("player_id") or meta.get("player_id") or "").strip()

        name = None
        if pid and pid in players_map:
            info = players_map.get(pid) or {}
            name = info.get("full_name") or f"{info.get('first_name','')} {info.get('last_name','')}".strip()

        if not name:
            name = (
                meta.get("full_name")
                or meta.get("player")
                or f"{meta.get('first_name','')} {meta.get('last_name','')}".strip()
            )

        if name:
            out.append(name.strip())

    return out


def picks_to_internal_log(picks: List[dict], players_map: Dict[str, Any], teams: int) -> List[dict]:
    """
    Convert normalized picks to a consistent, app-friendly log:
      {
        "round": int,
        "pick_no": int,
        "slot": int | None,        # roster slot (aka roster_id) on the clock when this pick was made
        "roster_id": Any | None,   # passthrough from Sleeper, if present
        "metadata": {
            "first_name": str,
            "last_name": str,
            "full_name": str,
            "position": str | None
        }
      }

    If 'slot' is missing, we compute it from (round, pick_no) via serpentine logic.
    """
    log: List[dict] = []
    for p in _normalize_picks(picks):
        meta = p.get("metadata") or {}
        rnd = int(p.get("round", 0) or 0)
        pick_no = int(p.get("pick_no", 0) or 0)
        roster_id = p.get("roster_id")
        slot = p.get("slot") or roster_id

        # Compute slot if missing and we have round/pick_no and team count
        if not slot and rnd and pick_no and teams:
            slot = slot_for_round_pick(rnd, pick_no, teams)

        pid = str(p.get("player_id") or meta.get("player_id") or "")
        pos = meta.get("position")

        full_name = None
        if pid and pid in players_map:
            info = players_map.get(pid) or {}
            full_name = info.get("full_name") or f"{info.get('first_name','')} {info.get('last_name','')}".strip()
            pos = info.get("position") or pos

        if not full_name:
            full_name = (
                meta.get("full_name")
                or meta.get("player")
                or f"{meta.get('first_name','')} {meta.get('last_name','')}".strip()
            )

        first = (full_name or "").split(" ")[0] if full_name else ""
        last = " ".join((full_name or "").split(" ")[1:]) if full_name else ""

        log.append({
            "round": rnd,
            "pick_no": pick_no,
            "slot": int(slot) if slot else None,
            "roster_id": roster_id,
            "metadata": {
                "first_name": first,
                "last_name": last,
                "full_name": full_name,
                "position": pos,
            },
        })

    return log

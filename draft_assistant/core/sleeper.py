# draft_assistant/core/sleeper.py
from __future__ import annotations
import re
import time
from typing import Any, Dict, List, Optional
import requests

BASE = "https://api.sleeper.app/v1"
HTTP_TIMEOUT = 10.0

# Naive in-memory cache (path -> (timestamp, data))
_CACHE: Dict[str, tuple[float, Any]] = {}
TTL_SECONDS = 30.0

def _get(path: str) -> Any:
    now = time.time()
    if path in _CACHE and now - _CACHE[path][0] < TTL_SECONDS:
        return _CACHE[path][1]
    url = f"{BASE}{path}"
    try:
        r = requests.get(url, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        _CACHE[path] = (now, data)
        return data
    except Exception:
        return None

def get_league_info(league_id: str) -> Optional[dict]:
    return _get(f"/league/{league_id}")

def get_drafts_for_league(league_id: str) -> List[dict]:
    return _get(f"/league/{league_id}/drafts") or []

def get_draft(draft_id: str) -> Optional[dict]:
    return _get(f"/draft/{draft_id}")

def get_picks(draft_id: str) -> List[dict]:
    return _get(f"/draft/{draft_id}/picks") or []

def get_users(league_id: str) -> List[dict]:
    return _get(f"/league/{league_id}/users") or []

def get_players_nfl() -> dict:
    """Large payload of all NFL players (cached for 1 hour)."""
    global TTL_SECONDS
    old_ttl = TTL_SECONDS
    TTL_SECONDS = 3600.0
    data = _get("/players/nfl") or {}
    TTL_SECONDS = old_ttl
    return data

# ---------------- URL / ID parsing ----------------
def parse_draft_id_from_url(url_or_id: str) -> Optional[str]:
    """
    Accepts:
      - Raw draft_id (alnum/underscore, 10–24 chars)
      - URLs like:
        https://sleeper.com/draft/123...
        https://sleeper.com/draft/nfl/123...
        https://sleeper.com/draft/board/123...
    """
    s = str(url_or_id).strip()
    if re.fullmatch(r"[A-Za-z0-9_]{10,24}", s):
        return s
    m = re.search(r"/draft/(?:nfl/|board/)?([A-Za-z0-9_]{10,24})", s)
    if m:
        return m.group(1)
    candidates = re.findall(r"([A-Za-z0-9_]{10,24})", s)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    return None

# ---------------- Helpers ----------------
def _slot_from_round_pick(round_number: int, pick_in_round: int, teams: int) -> int:
    """Snake draft slot for a given (round, pick_in_round)."""
    if round_number % 2 == 1:
        return int(pick_in_round)
    return int(teams) - int(pick_in_round) + 1

# ---------------- Picks conversion ----------------
def picks_to_internal_log(picks: List[dict], players_map: dict, teams: int | None = None) -> List[dict]:
    """
    Convert Sleeper picks to our simple structure:
      {round, pick_no, slot, roster_id, team, metadata:{first_name,last_name,position}}
    Prefer Sleeper's 'draft_slot' -> slot/roster_id; if missing, compute via snake math.
    """
    out = []
    for p in picks or []:
        meta = p.get("metadata") or {}
        pid = p.get("player_id")

        # Name/pos from metadata, fallback to players map
        first = (meta.get("first_name") or "").strip()
        last = (meta.get("last_name") or "").strip()
        position = (meta.get("position") or "").strip()
        if (not first and not last) and pid and pid in players_map:
            pm = players_map.get(pid) or {}
            first = (pm.get("first_name") or "").strip() or first
            last = (pm.get("last_name") or "").strip() or last
            position = position or (pm.get("position") or "").strip()
        if not position and pid and pid in players_map:
            position = (players_map.get(pid) or {}).get("position")

        # Round/pick & slot
        rnd = p.get("round")
        pick_no = p.get("pick_no")
        try:
            rnd = int(rnd) if rnd is not None else 0
        except Exception:
            rnd = 0
        try:
            pick_no = int(pick_no) if pick_no is not None else 0
        except Exception:
            pick_no = 0

        slot = p.get("draft_slot")
        try:
            slot = int(slot) if slot is not None else None
        except Exception:
            slot = None

        if slot is None and teams and rnd and pick_no:
            slot = _slot_from_round_pick(rnd, pick_no, int(teams))
        if slot is None:
            slot = 0  # as a last resort

        out.append({
            "round": rnd,
            "pick_no": pick_no,
            "slot": int(slot),
            "roster_id": int(slot),
            "team": p.get("picked_by") or "",
            "metadata": {"first_name": first, "last_name": last, "position": position},
        })
    return out

# ---------------- Picked names helper ----------------
def picked_player_names(picks: List[dict], players_map: dict) -> set[str]:
    """
    Build a set of drafted player names as strings that match our CSV 'PLAYER' field
    as closely as possible: 'First Last' when available, falling back to /players/nfl map.
    """
    out: set[str] = set()
    for p in picks or []:
        meta = p.get("metadata") or {}
        pid = p.get("player_id")

        first = (meta.get("first_name") or "").strip()
        last = (meta.get("last_name") or "").strip()
        full = f"{first} {last}".strip()

        if not full and pid and players_map:
            pm = players_map.get(pid) or {}
            pf = (pm.get("first_name") or "").strip()
            pl = (pm.get("last_name") or "").strip()
            if pf or pl:
                full = f"{pf} {pl}".strip()
            else:
                full = (pm.get("full_name") or pm.get("name") or "").strip()

        if not full:
            full = (meta.get("name") or "").strip()

        if full:
            out.add(full)
    return out

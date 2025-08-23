from __future__ import annotations
from typing import Dict, Any, List, Optional
import httpx
from .utils import norm_name, slot_for_round_pick  # for picks_to_internal_log

BASE = "https://api.sleeper.app/v1"
_UA = {"User-Agent": "FF_DA_2025 (github.com/fallon3d/FF_DA_2025)"}

def _get(path: str, timeout: float = 20.0) -> Any:
    url = f"{BASE}{path}"
    with httpx.Client(timeout=timeout, headers=_UA) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()

# -------- Core endpoints --------

def get_league_info(league_id: str) -> Dict[str, Any]:
    return _get(f"/league/{league_id}")

def get_drafts_for_league(league_id: str) -> List[dict]:
    return _get(f"/league/{league_id}/drafts")

def get_draft(draft_id: str) -> Dict[str, Any]:
    return _get(f"/draft/{draft_id}")

def get_picks(draft_id: str) -> List[dict]:
    return _get(f"/draft/{draft_id}/picks")

def get_users(league_id: str) -> List[dict]:
    return _get(f"/league/{league_id}/users")

def get_players_nfl() -> Dict[str, Any]:
    # Big payload; cache with @st.cache_resource on the caller
    return _get("/players/nfl")

# -------- Helpers for mocks & names --------

def parse_draft_id_from_url(url_or_id: str) -> Optional[str]:
    """
    Accepts:
      - Full URLs (sleeper.com or sleeper.app), with or without '/nfl' segment
      - Bare draft_id (all digits, length >= 10)
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
        if p == "draft":
            # next is possibly 'nfl' then id
            if i + 2 < len(parts) and parts[i+1].lower() == "nfl":
                return parts[i+2]
            if i + 1 < len(parts):
                return parts[i+1]
    # Fallback last numeric token
    for p in reversed(parts):
        if p.isdigit() and len(p) >= 10:
            return p
    return None

def picked_player_names(picks: List[dict], players_map: Dict[str, Any]) -> List[str]:
    """
    Resolve picked player full names using the official players dump when available.
    Falls back to pick metadata names.
    """
    out: List[str] = []
    for p in picks or []:
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
    Convert Sleeper picks to a simplified, consistent log:
      { round, pick_no, slot, roster_id, metadata: {first_name,last_name,position,full_name} }
    If 'slot' (roster_id) is missing, compute from (round, pick_no) + teams.
    """
    log: List[dict] = []
    for p in picks or []:
        meta = p.get("metadata") or {}
        rnd = int(p.get("round", 0) or 0)
        pick_no = int(p.get("pick_no", 0) or 0)
        roster_id = p.get("roster_id")
        slot = p.get("slot") or roster_id
        if not slot and rnd and pick_no and teams:
            slot = slot_for_round_pick(rnd, pick_no, teams)
        pid = str(p.get("player_id") or meta.get("player_id") or "")
        # names
        full_name = None
        if pid and pid in players_map:
            info = players_map.get(pid) or {}
            full_name = info.get("full_name") or f"{info.get('first_name','')} {info.get('last_name','')}".strip()
            pos = info.get("position") or meta.get("position")
        else:
            pos = meta.get("position")
            full_name = (
                meta.get("full_name")
                or meta.get("player")
                or f"{meta.get('first_name','')} {meta.get('last_name','')}".strip()
            )
        log.append({
            "round": rnd,
            "pick_no": pick_no,
            "slot": int(slot) if slot else None,
            "roster_id": roster_id,
            "metadata": {
                "first_name": (full_name or "").split(" ")[0] if full_name else "",
                "last_name": " ".join((full_name or "").split(" ")[1:]) if full_name else "",
                "full_name": full_name,
                "position": pos,
            },
        })
    return log

from __future__ import annotations
from typing import Dict, Any, List, Optional
import re
import httpx

from .utils import slot_for_round_pick  # compute slot when missing

BASE = "https://api.sleeper.app/v1"
_UA = {"User-Agent": "FF_DA_2025 (github.com/fallon3d/FF_DA_2025)"}

def _get(path: str, timeout: float = 20.0) -> Any:
    url = f"{BASE}{path}"
    with httpx.Client(timeout=timeout, headers=_UA) as client:
        r = client.get(url)
        r.raise_for_status()
        return r.json()

# ---------------- Normalizers ----------------

def _normalize_picks(picks) -> List[dict]:
    if isinstance(picks, dict):
        for key in ("picks", "draft_picks", "data"):
            val = picks.get(key)
            if isinstance(val, list):
                picks = val
                break
        else:
            return []
    if not isinstance(picks, list):
        return []
    out: List[dict] = []
    for p in picks:
        if isinstance(p, dict):
            if not isinstance(p.get("metadata"), dict):
                p = {**p, "metadata": {}}
            out.append(p)
        elif isinstance(p, str):
            out.append({"player_id": p, "metadata": {}})
    return out

# ---------------- Core endpoints ----------------

def get_league_info(league_id: str) -> Dict[str, Any]:
    return _get(f"/league/{league_id}")

def get_drafts_for_league(league_id: str) -> List[dict]:
    return _get(f"/league/{league_id}/drafts") or []

def get_draft(draft_id: str) -> Dict[str, Any]:
    return _get(f"/draft/{draft_id}") or {}

def get_picks(draft_id: str) -> List[dict]:
    raw = _get(f"/draft/{draft_id}/picks")
    return _normalize_picks(raw)

def get_users(league_id: str) -> List[dict]:
    return _get(f"/league/{league_id}/users") or []

def get_rosters(league_id: str) -> List[dict]:
    """Map roster_id <-> owner_id; needed to find a user's roster/slot."""
    return _get(f"/league/{league_id}/rosters") or []

def get_players_nfl() -> Dict[str, Any]:
    return _get("/players/nfl") or {}

# ---------------- Helpers (URLs, naming, logs) ----------------

_ALNUM_ID = re.compile(r"[A-Za-z0-9_]{10,24}")

def parse_draft_id_from_url(url_or_id: str) -> Optional[str]:
    if not url_or_id:
        return None
    s = url_or_id.strip()

    # Bare id (alpha-numeric + underscore, 10..24)
    if _ALNUM_ID.fullmatch(s):
        return s

    # URLs: /draft/<id>, /draft/nfl/<id>, /draft/board/<id>
    m = re.search(r"/draft/(?:nfl/|board/)?([A-Za-z0-9_]{10,24})", s)
    if m:
        return m.group(1)

    # Fallback: longest alnum/underscore token
    candidates = re.findall(r"([A-Za-z0-9_]{10,24})", s)
    if candidates:
        candidates.sort(key=len, reverse=True)
        return candidates[0]
    return None

def picked_player_names(picks: List[dict], players_map: Dict[str, Any]) -> List[str]:
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
    log: List[dict] = []
    for p in _normalize_picks(picks):
        meta = p.get("metadata") or {}
        rnd = int(p.get("round", 0) or 0)
        pick_no = int(p.get("pick_no", 0) or 0)
        roster_id = p.get("roster_id")
        slot = p.get("slot") or p.get("draft_slot") or roster_id
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
            "metadata": {"first_name": first, "last_name": last, "full_name": full_name, "position": pos},
        })
    return log

from __future__ import annotations
from typing import Dict, Any, List, Optional
import httpx

# Public, read-only Sleeper API
BASE = "https://api.sleeper.app/v1"

_DEFAULT_HEADERS = {
    "User-Agent": "FF_DA_2025 (https://github.com/fallon3d/FF_DA_2025)"
}

def _get(path: str, timeout: float = 15.0) -> Any:
    """
    Lightweight GET helper. Raises httpx.HTTPStatusError on non-2xx.
    """
    url = f"{BASE}{path}"
    with httpx.Client(timeout=timeout, headers=_DEFAULT_HEADERS) as client:
        resp = client.get(url)
        resp.raise_for_status()
        return resp.json()

# -------------------------------
# League / Draft endpoints
# -------------------------------

def get_league(league_id: str) -> Dict[str, Any]:
    """
    League metadata: roster_positions, scoring_settings, draft_id, total_rosters, etc.
    https://docs.sleeper.com/#getting-started
    """
    return _get(f"/league/{league_id}")

def get_drafts_for_league(league_id: str) -> List[dict]:
    """
    All drafts associated with the league (current + past).
    """
    return _get(f"/league/{league_id}/drafts")

def get_draft(draft_id: str) -> Dict[str, Any]:
    """
    Single draft metadata (status: pre_draft, in_progress, complete).
    """
    return _get(f"/draft/{draft_id}")

def get_draft_picks(draft_id: str) -> List[dict]:
    """
    List of picks for a draft. Each entry includes:
      - player_id, picked_by (user_id), round, pick_no, metadata (names/pos), etc.
    """
    return _get(f"/draft/{draft_id}/picks")

def get_rosters(league_id: str) -> List[dict]:
    """
    Rosters (players) by team for the league.
    """
    return _get(f"/league/{league_id}/rosters")

def get_users_in_league(league_id: str) -> List[dict]:
    """
    Users (managers) in the league with display_name and user_id.
    """
    return _get(f"/league/{league_id}/users")

# -------------------------------
# Players dump
# -------------------------------

def get_players() -> Dict[str, Any]:
    """
    Full NFL players dump (large JSON). Keyed by Sleeper player_id (string).
    Each entry can include:
      first_name, last_name, full_name, position, team, injury_status, etc.
    Fetch once per session and cache in the app with @st.cache_resource.
    """
    return _get("/players/nfl")

# -------------------------------
# Helpers
# -------------------------------

def parse_mock_draft_id_from_url(url: str) -> Optional[str]:
    """
    Accepts typical Sleeper mock URLs:
      - https://sleeper.com/draft/nfl/<draft_id>
      - https://sleeper.app/draft/nfl/<draft_id>
    Falls back to the last long integer-like token if structure differs.
    """
    if not url:
        return None
    parts = url.strip("/").split("/")
    # Look for ".../draft/nfl/<id>"
    for i, p in enumerate(parts):
        if p == "draft" and i + 2 < len(parts):
            # parts[i+1] is expected to be 'nfl'
            return parts[i + 2]
    # Fallback: last integer-looking segment with length >= 12
    for p in reversed(parts):
        if p.isdigit() and len(p) >= 12:
            return p
    return None

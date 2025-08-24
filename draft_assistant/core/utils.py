# draft_assistant/core/utils.py
from __future__ import annotations

import io
import math
from typing import List, Dict, Optional, Tuple

import numpy as np
import pandas as pd

POS_NORMALIZE = {
    "D/ST": "DEF", "DST": "DEF", "TEAM D": "DEF", "TEAM DEF": "DEF", "DEFENSE": "DEF",
}

def norm_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def read_player_table(file_or_path) -> pd.DataFrame:
    if file_or_path is None:
        return None
    try:
        if hasattr(file_or_path, "read"):  # UploadedFile-like
            name = getattr(file_or_path, "name", "")
            if str(name).lower().endswith((".xlsx",".xls")):
                df = pd.read_excel(file_or_path)
            else:
                df = pd.read_csv(file_or_path)
        else:
            path = str(file_or_path)
            if path.lower().endswith((".xlsx",".xls")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
    except Exception:
        return None

    # Minimal normalization of expected columns
    rename_map = {}
    cols_lower = {c.lower(): c for c in df.columns}
    if "player" in cols_lower and "PLAYER" not in df.columns:
        rename_map[cols_lower["player"]] = "PLAYER"
    if "pos" in cols_lower and "POS" not in df.columns:
        rename_map[cols_lower["pos"]] = "POS"
    if "team" in cols_lower and "TEAM" not in df.columns:
        rename_map[cols_lower["team"]] = "TEAM"
    if "adp" in cols_lower and "ADP" not in df.columns:
        rename_map[cols_lower["adp"]] = "ADP"
    if "tier" in cols_lower and "TIER" not in df.columns:
        rename_map[cols_lower["tier"]] = "TIER"
    if "proj_pts" in cols_lower and "PROJ_PTS" not in df.columns:
        rename_map[cols_lower["proj_pts"]] = "PROJ_PTS"

    if rename_map:
        df = df.rename(columns=rename_map)

    if "PLAYER" in df.columns:
        df["PLAYER"] = df["PLAYER"].astype(str).str.strip()
    if "POS" in df.columns:
        df["POS"] = df["POS"].astype(str).str.upper().map(POS_NORMALIZE).fillna(df["POS"])
    if "TEAM" in df.columns:
        df["TEAM"] = df["TEAM"].astype(str).str.upper()

    return df

def snake_position(overall_pick: int, teams: int) -> Tuple[int, int, int]:
    """(round_number, pick_in_round, slot_on_clock) for a snake draft."""
    rnd = (overall_pick - 1) // teams + 1
    pick_in_round = (overall_pick - 1) % teams + 1
    if rnd % 2 == 1:
        slot = pick_in_round
    else:
        slot = teams - pick_in_round + 1
    return rnd, pick_in_round, slot

def starters_from_roster_positions(roster_positions: List[str]) -> Dict[str, int]:
    """Count required starters (approx) for QB/RB/WR/TE/K/DEF; count FLEX as 0.5 RB + 0.5 WR."""
    starters = {"QB":0, "RB":0, "WR":0, "TE":0, "K":0, "DEF":0}
    flex = 0
    for r in roster_positions:
        R = str(r or "").upper().strip()
        R = POS_NORMALIZE.get(R, R)
        if R in starters:
            starters[R] += 1
        elif R == "FLEX":
            flex += 1
    starters["RB"] += 0.5 * flex
    starters["WR"] += 0.5 * flex
    return starters

def slot_to_display_name(slot: int, users: List[dict], rosters: Optional[List[dict]] = None) -> str:
    """Best-effort display name for a slot; fall back to 'Slot N'."""
    # Sleeper /league/{id}/users has no slot mapping; we only know slot number from picks.
    return f"Slot {slot}"

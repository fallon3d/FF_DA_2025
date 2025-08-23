from __future__ import annotations
import io
import re
from typing import Dict, List, Iterable, Union, Optional, Tuple
import numpy as np
import pandas as pd

# Canonical positions
POSS = ["QB", "RB", "WR", "TE", "K", "DEF", "DST"]

# -----------------------------
# Names & normalization
# -----------------------------

def norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def _upper_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    return df

def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Map common CSV/Excel header synonyms to canonical names expected downstream."""
    aliases = {
        "PLAYER": ["NAME","PLAYER_NAME","FULL_NAME","PLAYERID","PLAYER ID"],
        "POS": ["POSITION","POS."],
        "TEAM": ["NFL","TM","TMR","CLUB"],
        "PROJ_PTS": ["PROJ","PROJECTION","PROJECTIONS","PPR","PPR_PTS","FPTS","FANTASY_POINTS"],
        "ADP": ["AVG_DRAFT_POS","AVG PICK","AVG_PICK","ADP_PPR"],
        "ECR": ["RANK_ECR","CONS_RANK","EXPERT_RANK"],
        "TIER": ["TIERS","TIERING"],
        "BYE": ["BYE_WEEK","BYE WK","BYEWK"],
        "INJURY_RISK": ["INJ","INJURY","RISK"],
    }
    present = set(df.columns)
    for canon, alts in aliases.items():
        if canon in present:
            continue
        for a in alts:
            if a in present:
                df[canon] = df[a]
                break
    return df

def ensure_player_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = _upper_cols(df)
    df = _apply_aliases(df)

    # Required minima
    if "PLAYER" not in df.columns or "POS" not in df.columns:
        raise ValueError("Players table needs at least PLAYER and POS columns.")
    # Accept either PROJ_PTS or components; if neither, error
    has_components = any(c in df.columns for c in [
        "PASS_YDS","PASS_TD","PASS_INT","RUSH_YDS","RUSH_TD","REC","REC_YDS","REC_TD","TWO_PT"
    ])
    if "PROJ_PTS" not in df.columns and not has_components:
        raise ValueError("Provide PROJ_PTS or component stats (PASS_YDS/PASS_TD/PASS_INT/RUSH_YDS/RUSH_TD/REC/REC_YDS/REC_TD).")

    # Optional columns
    for c in ["TEAM","ADP","ECR","TIER","BYE","INJURY_RISK","SOS_SEASON","TGT_SHARE","RUSH_SHARE",
              "GOAL_LINE_SHARE","AIR_YARDS","ROUTE_PCT","REDZONE_TGT"]:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize values
    df["PLAYER_KEY"] = df["PLAYER"].map(norm_name)
    df["POS"] = df["POS"].astype(str).str.upper().str.replace("DST","DEF", regex=False)

    # Numeric coercions
    for c in ["PROJ_PTS","ADP","ECR","TIER","BYE","SOS_SEASON","TGT_SHARE","RUSH_SHARE",
              "GOAL_LINE_SHARE","AIR_YARDS","ROUTE_PCT","REDZONE_TGT"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Injury mapping to numeric penalty 0..~0.2
    def _injury_val(x):
        if isinstance(x,(int,float)) and not pd.isna(x): return float(x)
        if not isinstance(x,str): return np.nan
        s = x.strip().lower()
        if s in ("low","l"): return 0.05
        if s in ("med","moderate","m"): return 0.12
        if s in ("high","h"): return 0.20
        try: return float(s)
        except Exception: return np.nan

    df["INJURY_VAL"] = df["INJURY_RISK"].map(_injury_val)
    return df

# -----------------------------
# File I/O — CSV & Excel
# -----------------------------

def read_player_table(path_or_buffer: Union[str, io.BytesIO, io.BufferedReader]) -> pd.DataFrame:
    """
    Read CSV or Excel and normalize headers/columns. Works with file-like objects.
    """
    if isinstance(path_or_buffer, (io.BytesIO, io.BufferedReader)):
        head = path_or_buffer.read(16)
        # reset cursor
        path_or_buffer.seek(0)
        is_excel = head[:2] == b"PK"  # xlsx zip header heuristic
        df = pd.read_excel(path_or_buffer) if is_excel else pd.read_csv(path_or_buffer)
    else:
        if str(path_or_buffer).lower().endswith((".xlsx",".xls")):
            df = pd.read_excel(path_or_buffer)
        else:
            df = pd.read_csv(path_or_buffer)

    return ensure_player_cols(df)

# -----------------------------
# Draft math & roster helpers
# -----------------------------

def snake_position(overall_pick: int, teams: int) -> Tuple[int,int,int]:
    """Return (round, pick_in_round, slot_on_clock) for serpentine draft."""
    rnd = (overall_pick - 1) // teams + 1
    pick_in_round = (overall_pick - 1) % teams + 1
    if rnd % 2 == 1:
        slot = pick_in_round
    else:
        slot = teams - pick_in_round + 1
    return rnd, pick_in_round, slot

def slot_for_round_pick(round_num: int, pick_in_round: int, teams: int) -> int:
    """Compute slot for a (round, pick) in a snake draft."""
    if round_num % 2 == 1:
        return int(pick_in_round)
    return int(teams) - int(pick_in_round) + 1

# -----------------------------
# User/roster mapping (Live)
# -----------------------------

def user_id_by_display_name(users: List[dict], display_name: str) -> Optional[str]:
    if not users or not display_name:
        return None
    target = display_name.strip().lower()
    for u in users:
        if str(u.get("display_name","")).strip().lower() == target:
            return u.get("user_id")
    return None

def roster_id_for_user(rosters: List[dict], user_id: str) -> Optional[int]:
    if not rosters or not user_id:
        return None
    for r in rosters:
        if str(r.get("owner_id")) == str(user_id):
            return r.get("roster_id")
    return None

def user_roster_id(users: List[dict], rosters: List[dict], username: str) -> Optional[int]:
    """Preferred way: display_name -> user_id -> roster_id (slot)."""
    uid = user_id_by_display_name(users, username)
    if not uid:
        return None
    return roster_id_for_user(rosters, uid)

def slot_to_display_name(slot: int, users: List[dict], rosters: Optional[List[dict]] = None) -> str:
    """
    Map slot/roster_id to a display name.
    Tries rosters (owner_id -> user_id -> users.display_name); falls back to users.roster_id; else "Slot N".
    """
    if rosters:
        owner = None
        for r in rosters:
            if str(r.get("roster_id")) == str(slot):
                owner = r.get("owner_id")
                break
        if owner:
            for u in users or []:
                if str(u.get("user_id")) == str(owner):
                    return u.get("display_name") or f"Slot {slot}"

    # Fallback: some installs include roster_id on user objects
    for u in users or []:
        if str(u.get("roster_id")) == str(slot):
            return u.get("display_name") or f"Slot {slot}"

    return f"Slot {slot}"

def remove_players_by_name(df: pd.DataFrame, names: Iterable[str]) -> pd.DataFrame:
    """Filter out players whose normalized names are in 'names'."""
    keys = set(norm_name(n) for n in names if isinstance(n, str))
    if "PLAYER_KEY" not in df.columns:
        df = ensure_player_cols(df)
    return df[~df["PLAYER_KEY"].isin(keys)].copy()

# -----------------------------
# Roster/FLEX baseline helpers (used by evaluation.py)
# -----------------------------

def starters_from_roster_positions(roster_positions: List[str]) -> Dict[str, int]:
    counts = {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0,"FLEX":0}
    for pos in roster_positions or []:
        p = str(pos).upper()
        if p in counts:
            counts[p] += 1
        elif "FLEX" in p:
            counts["FLEX"] += 1
    return counts

def apply_flex_adjustment(
    df: pd.DataFrame, teams: int, starters: Dict[str,int], repl_pts: Dict[str,float]
) -> Dict[str,float]:
    flex = int(starters.get("FLEX", 0) or 0)
    if flex <= 0:
        return repl_pts
    mask = df["POS"].isin(["RB","WR","TE"])
    combo = df[mask].sort_values("EVAL_PTS", ascending=False)
    if combo.empty:
        return repl_pts
    total_core = teams * (starters.get("RB",0)+starters.get("WR",0)+starters.get("TE",0))
    index = min(len(combo)-1, total_core + teams*flex - 1)
    if index < 0:
        return repl_pts
    flex_baseline = float(combo.iloc[index]["EVAL_PTS"])
    for p in ["RB","WR","TE"]:
        repl_pts[p] = min(repl_pts.get(p, flex_baseline), flex_baseline)
    return repl_pts

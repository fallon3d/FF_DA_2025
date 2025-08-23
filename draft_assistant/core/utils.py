from __future__ import annotations
import re
from typing import Dict, List
import numpy as np
import pandas as pd

# Canonical position list used across the app
POSS = ["QB", "RB", "WR", "TE", "K", "DEF"]

# -----------------------------
# Name & column normalization
# -----------------------------

def norm_name(s: str) -> str:
    """
    Normalize a player's name for matching across sources:
    - lowercase
    - alphanumeric only (spaces for separators)
    - collapse whitespace
    """
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Uppercase + trim all column names."""
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    return df


def ensure_player_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required/optional columns exist and are typed correctly.
    Required: PLAYER, POS, PROJ_PTS
    Optional (created if missing): TEAM, ADP, ECR, TIER, BYE, INJURY_RISK,
    SOS_SEASON, TGT_SHARE, RUSH_SHARE, GOAL_LINE_SHARE, AIR_YARDS,
    ROUTE_PCT, REDZONE_TGT
    Also adds:
      - PLAYER_KEY (normalized name)
      - INJURY_VAL (numeric risk 0..~0.2 from labels or raw numbers)
    """
    df = normalize_columns(df)
    required = ["PLAYER", "POS", "PROJ_PTS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Need at least {required}.")

    # Optional columns used by evaluation/suggestions
    optional = [
        "TEAM", "ADP", "ECR", "TIER", "BYE", "INJURY_RISK", "SOS_SEASON",
        "TGT_SHARE", "RUSH_SHARE", "GOAL_LINE_SHARE", "AIR_YARDS",
        "ROUTE_PCT", "REDZONE_TGT",
    ]
    for c in optional:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize values
    df["PLAYER_KEY"] = df["PLAYER"].map(norm_name)
    df["POS"] = df["POS"].astype(str).str.upper().str.replace("DST", "DEF", regex=False)

    # Numeric coercions where applicable
    for c in [
        "PROJ_PTS", "ADP", "ECR", "TIER", "BYE", "SOS_SEASON", "TGT_SHARE",
        "RUSH_SHARE", "GOAL_LINE_SHARE", "AIR_YARDS", "ROUTE_PCT", "REDZONE_TGT",
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    # Map INJURY_RISK -> INJURY_VAL (0..~0.2 typical)
    def _injury_val(x):
        if isinstance(x, (int, float)) and not pd.isna(x):
            return float(x)
        if not isinstance(x, str):
            return np.nan
        s = x.strip().lower()
        if s in ("low", "l"):
            return 0.05
        if s in ("med", "moderate", "m"):
            return 0.12
        if s in ("high", "h"):
            return 0.20
        # Try to parse raw numeric text
        try:
            return float(s)
        except Exception:
            return np.nan

    df["INJURY_VAL"] = df["INJURY_RISK"].map(_injury_val)

    return df

# -----------------------------
# Roster slots / FLEX handling
# -----------------------------

def starters_from_roster_positions(roster_positions: List[str]) -> Dict[str, int]:
    """
    Convert Sleeper roster_positions (e.g., ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF"])
    into a per-position starter count dict (including FLEX count).
    """
    counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DEF": 0, "FLEX": 0}
    for pos in roster_positions or []:
        p = str(pos).upper()
        if p in counts:
            counts[p] += 1
        elif "FLEX" in p:
            counts["FLEX"] += 1
    return counts


def apply_flex_adjustment(
    df: pd.DataFrame,
    teams: int,
    starters: Dict[str, int],
    repl_pts: Dict[str, float],
) -> Dict[str, float]:
    """
    FLEX raises the effective baseline for RB/WR/TE.
    We compute the combined starters (RB+WR+TE plus FLEX slots) and find the
    EVAL_PTS at that combined index; then we cap each RB/WR/TE baseline to at least
    this FLEX baseline to avoid unrealistically low replacement points.
    """
    flex = int(starters.get("FLEX", 0) or 0)
    if flex <= 0:
        return repl_pts

    skill_mask = df["POS"].isin(["RB", "WR", "TE"])
    combo = df[skill_mask].sort_values("EVAL_PTS", ascending=False)
    if combo.empty:
        return repl_pts

    total_core = teams * (starters.get("RB", 0) + starters.get("WR", 0) + starters.get("TE", 0))
    flex_index = min(len(combo) - 1, total_core + teams * flex - 1)
    if flex_index < 0:
        return repl_pts

    flex_baseline = float(combo.iloc[flex_index]["EVAL_PTS"])
    for p in ["RB", "WR", "TE"]:
        repl_pts[p] = min(repl_pts.get(p, flex_baseline), flex_baseline)
    return repl_pts

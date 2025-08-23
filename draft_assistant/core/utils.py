from __future__ import annotations
import re
from typing import Dict, List
import numpy as np
import pandas as pd

POSS = ["QB", "RB", "WR", "TE", "K", "DEF"]

def norm_name(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip().lower()
    s = re.sub(r"[^a-z0-9]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().upper() for c in df.columns]
    return df

def _apply_aliases(df: pd.DataFrame) -> pd.DataFrame:
    """Map common CSV header synonyms to our canonical names."""
    aliases = {
        "PLAYER": ["NAME", "PLAYER_NAME", "FULL_NAME"],
        "POS": ["POSITION", "POS."],
        "PROJ_PTS": ["PROJ", "PROJECTION", "PROJECTIONS", "PPR", "PPR_PTS", "FPTS", "FANTASY_POINTS"],
    }
    colset = set(df.columns)
    for canon, alts in aliases.items():
        if canon in colset:
            continue
        for a in alts:
            if a in colset:
                df[canon] = df[a]
                break
    return df

def ensure_player_cols(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure required/optional columns exist and are typed correctly.
    Required: PLAYER, POS, PROJ_PTS (or component stats so PROJ_PTS can be recomputed later)
    Optional: TEAM, ADP, ECR, TIER, BYE, INJURY_RISK, SOS_SEASON, TGT_SHARE, RUSH_SHARE,
              GOAL_LINE_SHARE, AIR_YARDS, ROUTE_PCT, REDZONE_TGT
    Adds: PLAYER_KEY, INJURY_VAL
    """
    df = normalize_columns(df)
    df = _apply_aliases(df)

    # If POS contains DST, map to DEF
    if "POS" in df.columns:
        df["POS"] = df["POS"].astype(str).str.upper().str.replace("DST", "DEF", regex=False)

    required = ["PLAYER", "POS"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Need at least {required} + projections.")

    # If PROJ_PTS missing, allow component-based recompute later, otherwise raise
    has_components = any(c in df.columns for c in [
        "PASS_YDS","PASS_TD","PASS_INT","RUSH_YDS","RUSH_TD","REC","REC_YDS","REC_TD","TWO_PT"
    ])
    if "PROJ_PTS" not in df.columns and not has_components:
        raise ValueError(
            "Missing PROJ_PTS and no component stat columns found. "
            "Provide PROJ_PTS or any of PASS_YDS, PASS_TD, PASS_INT, RUSH_YDS, RUSH_TD, REC, REC_YDS, REC_TD."
        )
    if "PROJ_PTS" not in df.columns:
        df["PROJ_PTS"] = np.nan  # will be recomputed by evaluation layer if components exist

    # Ensure optional columns exist
    for c in [
        "TEAM","ADP","ECR","TIER","BYE","INJURY_RISK","SOS_SEASON","TGT_SHARE","RUSH_SHARE",
        "GOAL_LINE_SHARE","AIR_YARDS","ROUTE_PCT","REDZONE_TGT"
    ]:
        if c not in df.columns:
            df[c] = np.nan

    # Types
    df["PLAYER_KEY"] = df["PLAYER"].map(norm_name)
    for c in [
        "PROJ_PTS","ADP","ECR","TIER","BYE","SOS_SEASON","TGT_SHARE","RUSH_SHARE",
        "GOAL_LINE_SHARE","AIR_YARDS","ROUTE_PCT","REDZONE_TGT"
    ]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    def _injury_val(x):
        if isinstance(x, (int, float)) and not pd.isna(x):
            return float(x)
        if not isinstance(x, str):
            return np.nan
        s = x.strip().lower()
        if s in ("low","l"): return 0.05
        if s in ("med","moderate","m"): return 0.12
        if s in ("high","h"): return 0.20
        try:
            return float(s)
        except Exception:
            return np.nan

    df["INJURY_VAL"] = df["INJURY_RISK"].map(_injury_val)
    return df

def starters_from_roster_positions(roster_positions: List[str]) -> Dict[str,int]:
    counts = {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0,"FLEX":0}
    for pos in roster_positions or []:
        p = str(pos).upper()
        if p in counts:
            counts[p] += 1
        elif "FLEX" in p:
            counts["FLEX"] += 1
    return counts

def apply_flex_adjustment(df: pd.DataFrame, teams: int, starters: Dict[str,int], repl_pts: Dict[str,float]) -> Dict[str,float]:
    flex = int(starters.get("FLEX", 0) or 0)
    if flex <= 0:
        return repl_pts
    skill_mask = df["POS"].isin(["RB","WR","TE"])
    combo = df[skill_mask].sort_values("EVAL_PTS", ascending=False)
    if combo.empty:
        return repl_pts
    total_core = teams * (starters.get("RB",0)+starters.get("WR",0)+starters.get("TE",0))
    flex_index = min(len(combo)-1, total_core + teams*flex - 1)
    if flex_index < 0:
        return repl_pts
    flex_baseline = float(combo.iloc[flex_index]["EVAL_PTS"])
    for p in ["RB","WR","TE"]:
        repl_pts[p] = min(repl_pts.get(p, flex_baseline), flex_baseline)
    return repl_pts

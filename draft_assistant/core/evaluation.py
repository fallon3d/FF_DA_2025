# draft_assistant/core/evaluation.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# Simple, stable VBD engine:
# - Use PROJ_PTS (or PTS/PROJECTION fallback) as projection.
# - Replacement level = projection of Nth player at position, where
#   N = starters_per_team * teams (+ share of FLEX for RB/WR).
# - VBD = proj - replacement.
# - EVAL_PTS mirrors PROJ_PTS to keep previous UI columns.

POS_NORMALIZE = {
    "D/ST": "DEF", "DST": "DEF", "TEAM D": "DEF", "TEAM DEF": "DEF", "DEFENSE": "DEF",
}

def _pos_norm(p: str) -> str:
    s = str(p or "").upper().strip()
    return POS_NORMALIZE.get(s, s)

def _pick_projection(df: pd.DataFrame) -> pd.Series:
    for c in ["PROJ_PTS", "PROJECTION", "PTS", "PROJECTED_POINTS"]:
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.zeros(len(df)), index=df.index)

def _replacement_index(teams: int, roster_positions: List[str], pos: str) -> int:
    pos = _pos_norm(pos)
    # Count starters per position; give 0.5 share of FLEX to RB and WR
    starters = {"QB":0, "RB":0, "WR":0, "TE":0, "K":0, "DEF":0}
    flex_count = 0
    for r in roster_positions:
        R = _pos_norm(r)
        if R in starters:
            starters[R] += 1
        elif R == "FLEX":
            flex_count += 1
    if flex_count > 0:
        starters["RB"] += 0.5 * flex_count
        starters["WR"] += 0.5 * flex_count
    needed = starters.get(pos, 0)
    if needed <= 0:
        return max(teams, 12)
    return int(round(needed * teams))

def _vbd_per_pos(df: pd.DataFrame, pos: str, repl_rank: int) -> pd.Series:
    pool = df[df["POS"] == pos].copy()
    if pool.empty:
        return pd.Series(np.zeros(len(df)), index=df.index)
    pool = pool.sort_values("PROJ", ascending=False).reset_index()
    idx = min(max(repl_rank-1, 0), len(pool)-1)
    replacement = float(pool.loc[idx, "PROJ"])
    vbd_map = {int(pool.loc[i, "index"]): float(pool.loc[i, "PROJ"]) - replacement for i in range(len(pool))}
    return pd.Series([vbd_map.get(i, 0.0) for i in range(len(df))], index=df.index)

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Normalize core columns
    if "PLAYER" not in out.columns:
        # try common alternatives
        for c in ["Name", "Player", "player"]:
            if c in out.columns:
                out = out.rename(columns={c: "PLAYER"})
                break
    if "POS" not in out.columns:
        for c in ["Position", "Pos"]:
            if c in out.columns:
                out = out.rename(columns={c: "POS"})
                break
    if "TEAM" not in out.columns:
        for c in ["Team", "Tm"]:
            if c in out.columns:
                out = out.rename(columns={c: "TEAM"})
                break
    if "ADP" not in out.columns:
        out["ADP"] = np.nan
    if "TIER" not in out.columns:
        out["TIER"] = np.nan
    # Derived
    out["PLAYER"] = out["PLAYER"].astype(str).str.strip()
    out["POS"] = out["POS"].astype(str).str.strip().str.upper().map(POS_NORMALIZE).fillna(out["POS"])
    out["TEAM"] = out["TEAM"].astype(str).str.strip().str.upper()
    out["PROJ"] = _pick_projection(out).fillna(0.0)
    out["EVAL_PTS"] = out["PROJ"].astype(float)
    return out

def evaluate_players(
    csv_df: pd.DataFrame,
    teams: int,
    roster_positions: List[str],
    current_picks: List[str],
    next_pick_window: Optional[int] = None
) -> pd.DataFrame:
    """
    Returns a normalized, availability-filtered dataframe with VBD computed.
    - current_picks: list of normalized names (norm_name) to remove from pool.
    """
    if csv_df is None or csv_df.empty:
        return pd.DataFrame(columns=["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD"])

    df = _ensure_columns(csv_df)

    # Remove taken
    taken_set = set(current_picks or [])
    df = df[~df["PLAYER"].str.lower().str.replace(r"[^a-z0-9]+", "", regex=True).isin(taken_set)].reset_index(drop=True)

    # Compute VBD per position using simple replacement
    vbd_total = pd.Series(np.zeros(len(df)), index=df.index)
    for pos in ["QB","RB","WR","TE","K","DEF"]:
        repl = _replacement_index(teams, roster_positions, pos)
        vbd_pos = _vbd_per_pos(df, pos, repl)
        vbd_total = vbd_total.add(vbd_pos, fill_value=0.0)

    df["VBD"] = vbd_total.round(2)

    # Order by position-friendly keys
    df = df.sort_values(["VBD", "EVAL_PTS"], ascending=False).reset_index(drop=True)

    # Keep classic columns
    keep = ["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD"]
    for k in keep:
        if k not in df.columns:
            df[k] = np.nan
    return df[keep]

# draft_assistant/core/evaluation.py
from __future__ import annotations

from typing import List, Dict, Tuple, Optional
import numpy as np
import pandas as pd

# Simple, stable VBD engine (pre–deep spreadsheet changes):
# - PROJ_PTS (or fallback) drives EVAL_PTS.
# - Optional light penalties/adjustments can be applied by caller via weights,
#   but by default we keep them minimal for stability.
# - Replacement based on starters per team; FLEX split 50/50 RB/WR.

POS_NORMALIZE = {"D/ST": "DEF", "DST": "DEF", "TEAM D": "DEF", "TEAM DEF": "DEF", "DEFENSE": "DEF"}

SCORING_DEFAULT: Dict[str, float] = {}  # placeholder to preserve signature

def _pos_norm(p: str) -> str:
    s = str(p or "").upper().strip()
    return POS_NORMALIZE.get(s, s)

def _projection(df: pd.DataFrame) -> pd.Series:
    for c in ("PROJ_PTS", "PROJECTION", "PTS", "PROJECTED_POINTS"):
        if c in df.columns:
            return pd.to_numeric(df[c], errors="coerce")
    return pd.Series(np.zeros(len(df)), index=df.index)

def _replacement_index(teams: int, roster_positions: List[str], pos: str) -> int:
    pos = _pos_norm(pos)
    starters = {"QB":0, "RB":0, "WR":0, "TE":0, "K":0, "DEF":0}
    flex = 0
    for r in roster_positions:
        R = _pos_norm(r)
        if R in starters:
            starters[R] += 1
        elif R == "FLEX":
            flex += 1
    starters["RB"] += 0.5 * flex
    starters["WR"] += 0.5 * flex
    need = starters.get(pos, 0)
    if need <= 0:
        return max(teams, 12)
    return int(round(need * teams))

def _vbd_for_pos(df: pd.DataFrame, pos: str, repl_rank: int, use_col: str) -> pd.Series:
    pool = df[df["POS"] == pos].copy()
    if pool.empty:
        return pd.Series(np.zeros(len(df)), index=df.index)
    pool = pool.sort_values(use_col, ascending=False).reset_index()
    idx = min(max(repl_rank-1, 0), len(pool)-1)
    replacement = float(pool.loc[idx, use_col])
    vbd_map = {int(pool.loc[i, "index"]): float(pool.loc[i, use_col]) - replacement for i in range(len(pool))}
    return pd.Series([vbd_map.get(i, 0.0) for i in range(len(df))], index=df.index)

def _ensure_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    # Normalize column names
    ren = {}
    cl = {c.lower(): c for c in out.columns}
    if "player" in cl and "PLAYER" not in out.columns: ren[cl["player"]] = "PLAYER"
    if "pos" in cl and "POS" not in out.columns: ren[cl["pos"]] = "POS"
    if "team" in cl and "TEAM" not in out.columns: ren[cl["team"]] = "TEAM"
    if "adp" in cl and "ADP" not in out.columns: ren[cl["adp"]] = "ADP"
    if "tier" in cl and "TIER" not in out.columns: ren[cl["tier"]] = "TIER"
    if "bye" in cl and "BYE" not in out.columns: ren[cl["bye"]] = "BYE"
    if ren: out = out.rename(columns=ren)

    out["PLAYER"] = out.get("PLAYER", pd.Series([""]*len(out))).astype(str).str.strip()
    out["POS"] = out.get("POS", pd.Series([""]*len(out))).astype(str).str.upper().map(POS_NORMALIZE).fillna(out.get("POS"))
    out["TEAM"] = out.get("TEAM", pd.Series([""]*len(out))).astype(str).str.upper()
    out["TIER"] = pd.to_numeric(out.get("TIER", np.nan), errors="coerce")
    out["ADP"] = pd.to_numeric(out.get("ADP", np.nan), errors="coerce")
    out["BYE"] = pd.to_numeric(out.get("BYE", np.nan), errors="coerce")

    # Projections
    out["PROJ"] = _projection(out).fillna(0.0)

    # Optional passthroughs
    if "INJURY_RISK" in out.columns:
        pass
    else:
        out["INJURY_RISK"] = np.nan

    return out

def evaluate_players(
    csv_df: pd.DataFrame,
    scoring_config: Dict[str, float],
    teams: int,
    roster_positions: List[str],
    weights: Dict[str, float] | None = None,
    current_picks: List[str] | None = None,
    next_pick_window: Optional[int] = None,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    """
    Normalize + filter already drafted + compute VBD.
    Returns (available_df, meta).
    """
    if csv_df is None or csv_df.empty:
        return pd.DataFrame(columns=["PLAYER","TEAM","POS","TIER","ADP","BYE","EVAL_PTS","VBD","INJURY_RISK"]), {}

    df = _ensure_columns(csv_df).copy()

    # Remove taken by normalized name
    taken = set(current_picks or [])
    df = df[~df["PLAYER"].str.lower().str.replace(r"[^a-z0-9]+", "", regex=True).isin(taken)].reset_index(drop=True)

    # EVAL_PTS = PROJ with *very* light adjustments (stable)
    inj_w = float((weights or {}).get("inj_w", 0.0))
    sos_w = float((weights or {}).get("sos_w", 0.0))

    # Injury risk numeric (low=0, med=0.5, high=1.0 if strings; else clamp 0..1)
    def _inj_val(x):
        if pd.isna(x): return 0.0
        try:
            v = float(x); return float(max(0.0, min(1.0, v)))
        except Exception:
            s = str(x).strip().lower()
            if s in ("low","l"): return 0.0
            if s in ("med","moderate","m"): return 0.5
            if s in ("high","h"): return 1.0
            return 0.0

    inj_arr = df["INJURY_RISK"].map(_inj_val).astype(float)

    # SOS_SEASON optional (centered); lighter effect
    if "SOS_SEASON" in df.columns:
        sos_raw = pd.to_numeric(df["SOS_SEASON"], errors="coerce")
        sos_norm = (sos_raw - sos_raw.mean(skipna=True)) / (sos_raw.std(skipna=True) if sos_raw.std(skipna=True) not in (0, np.nan) else 1)
        sos_adj = (-sos_norm.fillna(0.0)) * (sos_w * 1.5)
    else:
        sos_adj = pd.Series(np.zeros(len(df)))

    eval_pts = df["PROJ"] * (1.0 - inj_w * inj_arr) + sos_adj
    df["EVAL_PTS"] = eval_pts.astype(float)

    # VBD on EVAL_PTS
    vbd_total = pd.Series(np.zeros(len(df)), index=df.index)
    for pos in ["QB","RB","WR","TE","K","DEF"]:
        repl = _replacement_index(teams, roster_positions, pos)
        vbd_pos = _vbd_for_pos(df, pos, repl, "EVAL_PTS")
        vbd_total = vbd_total.add(vbd_pos, fill_value=0.0)
    df["VBD"] = vbd_total.round(2)

    keep = ["PLAYER","TEAM","POS","TIER","ADP","BYE","EVAL_PTS","VBD","INJURY_RISK"]
    out = df[keep].sort_values(["VBD","EVAL_PTS"], ascending=False).reset_index(drop=True)

    meta = {"teams": teams, "roster_positions": roster_positions}
    return out, meta

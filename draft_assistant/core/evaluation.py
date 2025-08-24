from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import pandas as pd

# Import both starters map and name normalizer
from .utils import starters_from_roster_positions, norm_name

# ------------- Defaults (kept backward compatible) -------------
SCORING_DEFAULT = {
    "sos_w": 0.03,       # Strength of schedule magnitude
    "def_w": 0.02,       # Opponent defense influence magnitude
    "tend_w": 0.02,      # TEAM_TENDENCY bump magnitude
    "passrate_w": 0.025, # PROE/neutral-pass-rate magnitude
    "vol_w": 0.015,      # Volatility penalty magnitude
    "inj_w": 0.04,       # Injury penalty magnitude
}

# ------------- Small helpers -------------

def _col(df: pd.DataFrame, *names: str) -> Optional[str]:
    lower = {c.lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lower:
            return lower[n.lower()]
    return None

def _to_num(x, default=np.nan) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, str):
            sx = x.strip()
            if sx == "":
                return default
            return float(sx)
        return float(x)
    except Exception:
        return default

def _safe_upper(x: str) -> str:
    return str(x or "").strip().upper()

def _map_volatility(v) -> float:
    s = str(v or "").strip().lower()
    if s in ("low","l","0"):
        return 0.15
    if s in ("medium","med","m","1"):
        return 0.45
    if s in ("high","h","2"):
        return 0.75
    fv = _to_num(v, default=np.nan)
    if pd.notna(fv):
        return float(max(0.0, min(1.0, fv)))
    return 0.45

def _map_injury(v) -> float:
    s = str(v or "").strip().lower()
    if s in ("low","l","0"):
        return 0.15
    if s in ("medium","med","m","1"):
        return 0.45
    if s in ("high","h","2"):
        return 0.75
    fv = _to_num(v, default=np.nan)
    if pd.notna(fv):
        return float(max(0.0, min(1.0, fv)))
    return 0.35

def _map_sos(v) -> float:
    s = _safe_upper(v)
    if "EASY" in s:
        return +1.0
    if "HARD" in s or "TOUGH" in s or "DIFFICULT" in s:
        return -1.0
    if s in ("AVG","AVERAGE","MED","MEDIUM","NEUTRAL"):
        return 0.0
    fv = _to_num(v, default=np.nan)
    if pd.notna(fv):
        return float(np.sign(fv))
    return 0.0

def _z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (s - mu) / sd

def _norm_pos(p: str) -> str:
    p = _safe_upper(p)
    if p in ("DST","D/ST","D-ST","TEAM D","TEAM DEF","DEFENSE"):
        return "DEF"
    return p

# ------------- Replacement / VBD helpers -------------

def _replacement_points(pos_df: pd.DataFrame, starters_total: int) -> float:
    if pos_df.empty:
        return 0.0
    srt = pos_df.sort_values(["EVAL_PTS"], ascending=False).reset_index(drop=True)
    idx = max(0, min(len(srt) - 1, starters_total - 1))
    val = float(srt.iloc[idx]["EVAL_PTS"])
    return val

def _starters_map(teams: int, roster_positions: List[str]) -> Dict[str, int]:
    base = starters_from_roster_positions(roster_positions or ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF"])
    for p in ("QB","TE","K","DEF"):
        base[p] = max(1, int(base.get(p, 0)))
    base["RB"] = max(1, int(base.get("RB", 0)))
    base["WR"] = max(1, int(base.get("WR", 0)))
    return base

# ------------- Main: evaluate_players -------------

def evaluate_players(
    df: pd.DataFrame,
    scoring: Dict[str, float],
    teams: int,
    roster_positions: List[str],
    weights: Dict[str, float] | None = None,
    current_picks: List[str] | None = None,   # <- drafted players (normalized names)
    next_pick_window: int | None = None,
    strategy_name: str | None = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Returns (available_df, replacement_by_pos).
    Adds/updates columns: POS, TEAM, EVAL_PTS, VBD.

    - df is the unified player table (already normalized by your loader).
    - current_picks: list/set of normalized names to EXCLUDE from availability.
                     (We use utils.norm_name on PLAYER to filter.)
    """
    if df is None or df.empty:
        return pd.DataFrame([]), {}

    df = df.copy()

    # Column mapping
    c_player = _col(df, "PLAYER", "Name", "Player") or "PLAYER"
    c_team   = _col(df, "TEAM", "Tm", "Team") or "TEAM"
    c_pos    = _col(df, "POS", "Position") or "POS"

    # ---------- REMOVE DRAFTED PLAYERS (FIX) ----------
    if current_picks:
        taken_keys = set(str(x).strip().lower() for x in current_picks if str(x).strip())
        # Build a normalized key for each row and filter out matches
        df["_NAME_KEY_"] = df[c_player].astype(str).apply(norm_name)
        df = df[~df["_NAME_KEY_"].isin(taken_keys)].drop(columns=["_NAME_KEY_"], errors="ignore")

    # Projections
    c_proj   = _col(df, "PROJ_PTS", "PROJ", "PROJECTED_PTS", "PROJECTION")
    if c_proj is None:
        df["__BASE_PROJ__"] = 0.0
    else:
        df["__BASE_PROJ__"] = pd.to_numeric(df[c_proj], errors="coerce").fillna(0.0)

    # Common helpers
    c_adp    = _col(df, "ADP")
    c_ecr    = _col(df, "ECR", "RANK_ECR")
    c_tier   = _col(df, "TIER")
    c_bye    = _col(df, "BYE", "BYE_WEEK")

    # Advanced fields
    c_sos    = _col(df, "SOS_SEASON", "SOS", "SCHEDULE_STRENGTH")
    c_vol    = _col(df, "VOLATILITY", "VOLATILITY_CAT")
    c_tend   = _col(df, "TEAM_TENDENCY")
    c_npr    = _col(df, "NEUTRAL_PASS_RATE_2024", "NEUTRAL_PASS_RATE")
    c_proe   = _col(df, "PROE_2024", "PROE")
    c_depa   = _col(df, "DST_DEF_EPA_PER_PLAY", "DEF_EPA_PER_PLAY_ALLOWED")
    c_expl   = _col(df, "DST_EXPLOSIVE_PLAYS_ALLOWED", "DEF_EXPLOSIVE_ALLOWED")
    c_rztd   = _col(df, "DST_RZ_TD_ALLOWED_RATE", "DEF_RZ_TD_ALLOWED_RATE")
    c_inj    = _col(df, "INJURY_RISK", "INJURY", "INJURY_TAG")

    # Normalize positions and essential columns
    df["POS"] = df[c_pos].map(_norm_pos) if c_pos in df else "WR"
    df["TEAM"] = df[c_team] if c_team in df else ""

    # --- weights (fallback to SCORING_DEFAULT) ---
    W = dict(SCORING_DEFAULT)
    if isinstance(scoring, dict):
        W.update({k: v for k, v in scoring.items() if isinstance(v, (int, float))})
    if isinstance(weights, dict):
        for k in ("inj_w","sos_w","vol_w","def_w","tend_w","passrate_w"):
            if k in weights and isinstance(weights[k], (int, float)):
                W[k] = float(weights[k])

    # --- schedule / defense / tendency signals ---
    sos_sig = _z(df[c_sos]) if c_sos else pd.Series(np.zeros(len(df)), index=df.index)
    if c_sos:
        sos_sign = df[c_sos].map(_map_sos)
        sos_sig = sos_sign * (1.0 + 0.25 * sos_sig.fillna(0.0))

    depa_sig = _z(df[c_depa]) if c_depa else 0.0
    expl_sig = _z(df[c_expl]) if c_expl else 0.0
    rztd_sig = _z(df[c_rztd]) if c_rztd else 0.0
    def_sig = (depa_sig + expl_sig + rztd_sig) / 3.0 if isinstance(depa_sig, pd.Series) else pd.Series(np.zeros(len(df)), index=df.index)

    passrate_sig = None
    if c_npr and c_proe:
        passrate_sig = 0.5 * _z(df[c_npr]) + 0.5 * _z(df[c_proe])
    elif c_npr:
        passrate_sig = _z(df[c_npr])
    elif c_proe:
        passrate_sig = _z(df[c_proe])
    else:
        passrate_sig = pd.Series(np.zeros(len(df)), index=df.index)

    tend_sig = pd.Series(np.zeros(len(df)), index=df.index)
    if c_tend:
        t = df[c_tend].astype(str).str.upper()
        tend_sig = np.where(t.str.contains("PASS"),  +1.0,
                     np.where(t.str.contains("RUN"), -1.0, 0.0))
        tend_sig = pd.Series(tend_sig, index=df.index)

    vol_idx = df[c_vol].map(_map_volatility) if c_vol else pd.Series(np.zeros(len(df)) + 0.35, index=df.index)
    inj_idx = df[c_inj].map(_map_injury)     if c_inj else pd.Series(np.zeros(len(df)) + 0.25, index=df.index)

    pos = df["POS"]
    base_proj = df["__BASE_PROJ__"].astype(float)

    sched_bump = W["sos_w"] * sos_sig
    def_bump   = W["def_w"] * def_sig

    pass_bump = W["passrate_w"] * passrate_sig
    pass_bump = np.where(pos.isin(["WR","TE","QB"]), pass_bump,
                 np.where(pos == "RB", -0.5 * pass_bump, 0.0))

    tend_bump = W["tend_w"] * tend_sig
    tend_bump = np.where(pos.isin(["WR","TE","QB"]), tend_bump,
                 np.where(pos == "RB", -0.5 * tend_bump, 0.0))

    vol_pen = W["vol_w"] * vol_idx
    inj_pen = W["inj_w"] * inj_idx

    net_bump = np.clip(sched_bump + def_bump + pass_bump + tend_bump, -0.20, 0.20)
    net_pen  = np.clip(vol_pen + inj_pen, 0.0, 0.35)

    adj_proj = base_proj * (1.0 + net_bump) * (1.0 - net_pen)
    df["EVAL_PTS"] = adj_proj.fillna(0.0)

    # --- Compute VBD ---
    starters = _starters_map(int(teams), roster_positions)
    replacement_by_pos: Dict[str, float] = {}

    out_frames = []
    for p in ["QB","RB","WR","TE","K","DEF"]:
        p_df = df[df["POS"] == p].copy()
        if p_df.empty:
            replacement_by_pos[p] = 0.0
            continue
        starters_total = int(starters.get(p, 0)) * int(teams)
        starters_total = max(1, starters_total)
        repl = _replacement_points(p_df, starters_total)
        replacement_by_pos[p] = float(repl)
        p_df["VBD"] = p_df["EVAL_PTS"] - repl
        out_frames.append(p_df)

    if not out_frames:
        return pd.DataFrame([]), replacement_by_pos

    res = pd.concat(out_frames, ignore_index=True)

    # Keep useful columns if present
    c_team = _col(df, "TEAM", "Tm", "Team") or "TEAM"
    c_pos  = _col(df, "POS", "Position") or "POS"
    c_adp  = _col(df, "ADP")
    c_ecr  = _col(df, "ECR", "RANK_ECR")
    c_tier = _col(df, "TIER")
    c_bye  = _col(df, "BYE", "BYE_WEEK")

    for src, dst in [(c_player, "PLAYER"), (c_team, "TEAM"), (c_pos, "POS"),
                     (c_adp, "ADP"), (c_ecr, "ECR"), (c_tier, "TIER"), (c_bye, "BYE")]:
        if src and src != dst and src in res and dst not in res:
            res[dst] = res[src]

    for c in ("ADP","ECR","TIER","BYE","VBD","EVAL_PTS"):
        if c in res:
            res[c] = pd.to_numeric(res[c], errors="coerce")

    res = res.sort_values(["VBD","EVAL_PTS"], ascending=False).reset_index(drop=True)
    return res, replacement_by_pos

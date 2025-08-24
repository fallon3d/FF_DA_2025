from __future__ import annotations
from typing import Dict, List, Tuple, Optional
import pandas as pd
import numpy as np

from draft_assistant.core.utils import (
    ensure_player_cols,
    starters_from_roster_positions,
    apply_flex_adjustment,
)

# ---------------------------------------------------------------------
# League Scoring Defaults (aligned to your settings)
# ---------------------------------------------------------------------
SCORING_DEFAULT: Dict[str, float] = {
    # Passing
    "pass_yd": 0.04,     # 25 yards = 1
    "pass_td": 4.0,
    "pass_int": -1.0,
    "two_pt": 2.0,       # generic 2-pt convs (pass/rush/rec if provided)

    # Rushing
    "rush_yd": 0.10,     # 10 yards = 1
    "rush_td": 6.0,

    # Receiving
    "rec": 1.0,          # PPR
    "rec_yd": 0.10,      # 10 yards = 1
    "rec_td": 6.0,

    # Kicking (coarse buckets; map your projection columns if present)
    "fg_0_39": 3.0,      # covers 0–19, 20–29, 30–39 as +3
    "fg_40_49": 4.0,
    "fg_50_59": 5.0,     # 50+
    "fg_miss": -1.0,
    "pat": 1.0,
    "pat_miss": -1.0,

    # Defense / Special Teams
    "sack": 1.0,
    "int": 2.0,
    "fr": 2.0,
    "safety": 2.0,
    "td": 6.0,
    "blk": 2.0,          # blocked kick
    "ff": 1.0,           # forced fumble

    # Points Allowed tiers
    "pa_0": 10.0,
    "pa_1_6": 7.0,
    "pa_7_13": 4.0,
    "pa_14_20": 1.0,
    "pa_28_34": -1.0,
    "pa_35p": -4.0,
}

# ---------------------------------------------------------------------
# Projection & Context Layer
# ---------------------------------------------------------------------

def recompute_proj_pts_if_components(df: pd.DataFrame, scoring: Dict[str, float]) -> pd.DataFrame:
    """
    If component stat columns exist, rebuild PROJ_PTS from league scoring.
    Otherwise, leave PROJ_PTS as provided.
    Supported (optional): PASS_YDS, PASS_TD, PASS_INT, RUSH_YDS, RUSH_TD,
                          REC, REC_YDS, REC_TD, TWO_PT
    """
    df = df.copy()
    cols = set(df.columns)

    has_rec = any(c in cols for c in ("REC", "REC_YDS", "REC_TD"))
    has_rush = any(c in cols for c in ("RUSH_YDS", "RUSH_TD"))
    has_pass = any(c in cols for c in ("PASS_YDS", "PASS_TD", "PASS_INT"))

    if not (has_rec or has_rush or has_pass):
        return df  # use existing PROJ_PTS

    pts = np.zeros(len(df), dtype=float)

    # Passing
    if "PASS_YDS" in cols: pts += df["PASS_YDS"].fillna(0).astype(float) * scoring.get("pass_yd", 0.04)
    if "PASS_TD" in cols:  pts += df["PASS_TD"].fillna(0).astype(float)  * scoring.get("pass_td", 4.0)
    if "PASS_INT" in cols: pts += df["PASS_INT"].fillna(0).astype(float) * scoring.get("pass_int", -1.0)

    # Rushing
    if "RUSH_YDS" in cols: pts += df["RUSH_YDS"].fillna(0).astype(float) * scoring.get("rush_yd", 0.10)
    if "RUSH_TD" in cols:  pts += df["RUSH_TD"].fillna(0).astype(float)  * scoring.get("rush_td", 6.0)

    # Receiving
    if "REC" in cols:      pts += df["REC"].fillna(0).astype(float)      * scoring.get("rec", 1.0)
    if "REC_YDS" in cols:  pts += df["REC_YDS"].fillna(0).astype(float)  * scoring.get("rec_yd", 0.10)
    if "REC_TD" in cols:   pts += df["REC_TD"].fillna(0).astype(float)   * scoring.get("rec_td", 6.0)

    # Generic two-point conversions (optional as total)
    if "TWO_PT" in cols:   pts += df["TWO_PT"].fillna(0).astype(float)   * scoring.get("two_pt", 2.0)

    df["PROJ_PTS"] = pts
    return df


def apply_context_adjustments(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Apply contextual adjustments on top of PROJ_PTS:
      - Injury penalty (INJURY_VAL ~ 0..0.2 typical), capped to avoid over-penalizing
      - Schedule strength (SOS_SEASON z-score; if higher means harder, penalize)
      - Usage/Upside composite: TGT_SHARE, RUSH_SHARE, GOAL_LINE_SHARE, ROUTE_PCT, REDZONE_TGT
    """
    df = df.copy()
    eval_pts = df["PROJ_PTS"].fillna(0).astype(float).values

    # Injury penalty
    inj = df.get("INJURY_VAL", pd.Series([np.nan] * len(df))).fillna(0.0).astype(float).values
    eval_pts *= (1 - np.clip(inj * float(weights.get("inj_w", 0.5)), 0.0, 0.6))

    # Schedule strength (z-score). If higher SOS = tougher, negative weight.
    sos = df.get("SOS_SEASON", pd.Series([np.nan] * len(df))).astype(float)
    if sos.notna().sum() >= 5 and float(sos.std(ddof=0)) > 1e-9:
        z = (sos - sos.mean()) / sos.std(ddof=0)
        eval_pts *= (1 + (-z.fillna(0).values) * float(weights.get("sos_w", 0.05)))

    # Usage / role-based upside
    usage_cols = ["TGT_SHARE", "RUSH_SHARE", "GOAL_LINE_SHARE", "ROUTE_PCT", "REDZONE_TGT"]
    usage = df[usage_cols].copy()
    for c in usage_cols:
        if c not in usage:
            usage[c] = np.nan
    usage = usage.fillna(usage.median(numeric_only=True))
    # Percentile rank each metric then average to 0..1
    usage_score = usage.rank(pct=True).mean(axis=1).values
    eval_pts *= (1 + usage_score * float(weights.get("usage_w", 0.05)))

    df["EVAL_PTS"] = eval_pts
    return df

# ---------------------------------------------------------------------
# Replacement, VBD & VONA
# ---------------------------------------------------------------------

def _fallback_starters_if_empty(roster_positions: List[str]) -> List[str]:
    """Fallback to a standard lineup if Sleeper did not provide roster_positions."""
    return roster_positions or ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]


def compute_replacement_points(
    df: pd.DataFrame,
    teams: int,
    starters: Dict[str, int]
) -> Dict[str, float]:
    """
    Compute per-position replacement EVAL_PTS:
      - Replacement rank approx = teams * starters_at_pos - 1 (zero-index)
      - If a pos has 0 explicit starters, fall back to ~teams-1
      - Apply FLEX adjustment so RB/WR/TE baselines include the effect of FLEX slots
    """
    # Determine rank (index) of replacement-level player for each position
    repl_ranks: Dict[str, int] = {}
    for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]:
        count = int(starters.get(pos, 0) or 0)
        rank = max(teams * count - 1, teams - 1)  # zero-index
        repl_ranks[pos] = max(rank, 0)

    # Lookup replacement points
    repl_pts: Dict[str, float] = {}
    for pos, rank in repl_ranks.items():
        sub = df[df["POS"] == pos].sort_values("EVAL_PTS", ascending=False)
        if len(sub) == 0:
            repl_pts[pos] = 0.0
        else:
            idx = min(rank, len(sub) - 1)
            repl_pts[pos] = float(sub.iloc[idx]["EVAL_PTS"])

    # FLEX adjustment (raises baseline for RB/WR/TE if FLEX increases the marginal starter)
    repl_pts = apply_flex_adjustment(df, int(teams), starters, repl_pts)
    return repl_pts


def add_vbd_and_vona(
    df: pd.DataFrame,
    teams: int,
    starters: Dict[str, int],
    next_pick_window: Optional[int] = None
) -> pd.DataFrame:
    """
    Add VBD and VONA to the dataframe.
      - VBD  = EVAL_PTS - replacement_pts_at_position
      - VONA ≈ EVAL_PTS - EVAL_PTS_of_pos_player_likely_available_at_next_pick
        (using an estimated window of picks until your next selection)
    """
    df = df.copy()
    repl_pts = compute_replacement_points(df, teams, starters)
    df["VBD"] = df.apply(lambda r: float(r["EVAL_PTS"] - repl_pts.get(r["POS"], 0.0)), axis=1)

    # Window: number of picks until your next turn (snake-aware caller can pass this in)
    window = int(next_pick_window) if (next_pick_window is not None and next_pick_window > 0) else int(teams)

    vona_map: Dict[str, float] = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        sub = df[df["POS"] == pos].sort_values("EVAL_PTS", ascending=False)
        if len(sub) == 0:
            vona_map[pos] = 0.0
        else:
            idx = min(window, len(sub) - 1)
            vona_map[pos] = float(sub.iloc[idx]["EVAL_PTS"])

    df["VONA"] = df.apply(lambda r: float(r["EVAL_PTS"] - vona_map.get(r["POS"], 0.0)), axis=1)
    return df

# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def evaluate_players(
    raw_df: pd.DataFrame,
    scoring: Dict[str, float],
    teams: int,
    roster_positions: List[str],
    weights: Dict[str, float],
    current_picks: List[str],
    next_pick_window: Optional[int] = None,   # <-- supports snake-aware VONA
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Main entry point.
    Returns:
      - avail_df: Available players with EVAL_PTS, VBD, VONA
      - starters: Dict of starting slot counts per position (incl. FLEX)
    """
    # Normalize columns, ensure minimal schema
    df = ensure_player_cols(raw_df)

    # Optionally recompute PROJ_PTS from component stats using league scoring
    df = recompute_proj_pts_if_components(df, scoring or SCORING_DEFAULT)

    # Apply contextual adjustments (injury, SOS, usage)
    df = apply_context_adjustments(df, weights or {"inj_w": 0.5, "sos_w": 0.05, "usage_w": 0.05})

    # Remove taken players (normalized key match)
    taken_keys = set([k for k in (current_picks or []) if isinstance(k, str)])
    avail = df[~df["PLAYER_KEY"].isin(taken_keys)].copy()

    # Determine starters/slots
    roster_positions = _fallback_starters_if_empty(roster_positions or [])
    starters = starters_from_roster_positions(roster_positions)

    # Add VBD & VONA
    avail = add_vbd_and_vona(
        avail,
        teams=int(teams or 12),
        starters=starters,
        next_pick_window=next_pick_window,
    )

    return avail, starters

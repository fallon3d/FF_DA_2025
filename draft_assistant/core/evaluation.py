from __future__ import annotations
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np

from draft_assistant.core.utils import (
    ensure_player_cols,
    starters_from_roster_positions,
    apply_flex_adjustment,
)

# Scoring defaults aligned to your league settings
SCORING_DEFAULT: Dict[str, float] = {
    # Passing
    "pass_yd": 0.04,
    "pass_td": 4.0,
    "pass_int": -1.0,
    "two_pt": 2.0,  # also used for rush/rec two-point conversions when available
    # Rushing
    "rush_yd": 0.10,
    "rush_td": 6.0,
    # Receiving
    "rec": 1.0,
    "rec_yd": 0.10,
    "rec_td": 6.0,
    # Kicking (coarse buckets; if you have component projections you can map them here)
    "fg_0_39": 3.0,     # covers 0–19, 20–29, 30–39 all at +3
    "fg_40_49": 4.0,
    "fg_50_59": 5.0,    # 50+
    "fg_miss": -1.0,
    "pat": 1.0,
    "pat_miss": -1.0,
    # Defense/ST (PA = points allowed tiers)
    "sack": 1.0,
    "int": 2.0,
    "fr": 2.0,
    "safety": 2.0,
    "td": 6.0,
    "blk": 2.0,
    "ff": 1.0,          # forced fumble
    "pa_0": 10.0,
    "pa_1_6": 7.0,
    "pa_7_13": 4.0,
    "pa_14_20": 1.0,
    "pa_28_34": -1.0,
    "pa_35p": -4.0,
}

# -------- Projection & Context Layer -------- #

def recompute_proj_pts_if_components(df: pd.DataFrame, scoring: Dict[str, float]) -> pd.DataFrame:
    """
    If component stat columns exist, rebuild PROJ_PTS from league scoring.
    Otherwise, leave PROJ_PTS as provided by the CSV.
    Supported skill columns (all optional): PASS_YDS, PASS_TD, PASS_INT,
    RUSH_YDS, RUSH_TD, REC, REC_YDS, REC_TD, TWO_PT (generic 2pt convs).
    """
    df = df.copy()
    cols = set(df.columns)

    has_rec = any(c in cols for c in ("REC", "REC_YDS", "REC_TD"))
    has_rush = any(c in cols for c in ("RUSH_YDS", "RUSH_TD"))
    has_pass = any(c in cols for c in ("PASS_YDS", "PASS_TD", "PASS_INT"))

    if not (has_rec or has_rush or has_pass):
        return df  # use existing PROJ_PTS

    pts = np.zeros(len(df))
    # Passing
    if "PASS_YDS" in cols: pts += df["PASS_YDS"].fillna(0) * scoring.get("pass_yd", 0.04)
    if "PASS_TD" in cols:  pts += df["PASS_TD"].fillna(0)  * scoring.get("pass_td", 4.0)
    if "PASS_INT" in cols: pts += df["PASS_INT"].fillna(0) * scoring.get("pass_int", -1.0)
    # Rushing
    if "RUSH_YDS" in cols: pts += df["RUSH_YDS"].fillna(0) * scoring.get("rush_yd", 0.10)
    if "RUSH_TD" in cols:  pts += df["RUSH_TD"].fillna(0)  * scoring.get("rush_td", 6.0)
    # Receiving
    if "REC" in cols:      pts += df["REC"].fillna(0)      * scoring.get("rec", 1.0)
    if "REC_YDS" in cols:  pts += df["REC_YDS"].fillna(0)  * scoring.get("rec_yd", 0.10)
    if "REC_TD" in cols:   pts += df["REC_TD"].fillna(0)   * scoring.get("rec_td", 6.0)
    # Generic two-point conversions (if provided as projected totals)
    if "TWO_PT" in cols:   pts += df["TWO_PT"].fillna(0)   * scoring.get("two_pt", 2.0)

    df["PROJ_PTS"] = pts
    return df


def apply_context_adjustments(df: pd.DataFrame, weights: Dict[str, float]) -> pd.DataFrame:
    """
    Apply immediate contextual adjustments:
      - Injury penalty (downside)
      - Schedule strength (SOS_SEASON z-score; if higher value means harder schedule, this penalizes)
      - Usage/Upside composite (TGT_SHARE, RUSH_SHARE, GOAL_LINE_SHARE, ROUTE_PCT, REDZONE_TGT)
    """
    df = df.copy()
    eval_pts = df["PROJ_PTS"].fillna(0).astype(float).values

    # Injury penalty (INJURY_VAL ~ 0.0-0.2 typical). Cap max penalty at ~60% to avoid nonsense.
    inj = df.get("INJURY_VAL", pd.Series([np.nan]*len(df))).fillna(0.0).astype(float).values
    eval_pts = eval_pts * (1 - np.clip(inj * weights.get("inj_w", 0.5), 0, 0.6))

    # Schedule strength (z-score). Negative weight if higher SOS is worse.
    sos = df.get("SOS_SEASON", pd.Series([np.nan]*len(df))).astype(float)
    if sos.notna().sum() >= 5 and sos.std(ddof=0) > 1e-9:
        z = (sos - sos.mean()) / sos.std(ddof=0)
        eval_pts = eval_pts * (1 + (-z.fillna(0).values) * weights.get("sos_w", 0.05))

    # Usage / role-based upside
    usage_cols = ["TGT_SHARE", "RUSH_SHARE", "GOAL_LINE_SHARE", "ROUTE_PCT", "REDZONE_TGT"]
    usage = df[usage_cols].copy()
    for c in usage_cols:
        if c not in usage:
            usage[c] = np.nan
    usage = usage.fillna(usage.median(numeric_only=True))
    # Percentile rank each metric then average
    usage_score = usage.rank(pct=True).mean(axis=1).values  # 0..1
    eval_pts = eval_pts * (1 + usage_score * weights.get("usage_w", 0.05))

    df["EVAL_PTS"] = eval_pts
    return df


# -------- Replacement, VBD & VONA -------- #

def _fallback_starters_if_empty(roster_positions: List[str]) -> List[str]:
    """Fallback to a standard lineup if Sleeper does not provide roster_positions."""
    return roster_positions or ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]


def compute_replacement_points(
    df: pd.DataFrame,
    teams: int,
    starters: Dict[str, int]
) -> Dict[str, float]:
    """
    Replacement rank ~ teams * starters_at_pos - 1 (zero-index).
    Ensures a reasonable baseline for positions with small counts (QB/TE/K/DEF).
    Applies FLEX adjustment so RB/WR/TE baselines are not unrealistically low.
    """
    # Determine rank (index) of replacement-level player for each position
    repl_ranks: Dict[str, int] = {}
    for pos in ["QB", "RB", "WR", "TE", "K", "DEF"]:
        count = starters.get(pos, 0)
        # Zero-index rank; if count=0 (no explicit slot), still use ~teams-1 as a generic baseline
        rank = max(teams * count - 1, teams - 1)
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

    # FLEX adjustment (raises baseline for RB/WR/TE if FLEX makes the marginal starter stronger)
    repl_pts = apply_flex_adjustment(df, teams, starters, repl_pts)
    return repl_pts


def add_vbd_and_vona(
    df: pd.DataFrame,
    teams: int,
    starters: Dict[str, int],
    next_pick_window: int | None = None
) -> pd.DataFrame:
    """
    Add VBD and VONA:
      - VBD = EVAL_PTS - replacement_pts_at_position
      - VONA ≈ EVAL_PTS - EVAL_PTS_of_pos_player_likely_available_at_next_pick
    """
    df = df.copy()
    repl_pts = compute_replacement_points(df, teams, starters)
    df["VBD"] = df.apply(lambda r: float(r["EVAL_PTS"] - repl_pts.get(r["POS"], 0.0)), axis=1)

    # Window: approx number of picks until your next turn; default ~teams for snake drafts.
    window = int(next_pick_window if next_pick_window is not None else max(teams, 1))
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


# -------- Public API -------- #

def evaluate_players(
    raw_df: pd.DataFrame,
    scoring: Dict[str, float],
    teams: int,
    roster_positions: List[str],
    weights: Dict[str, float],
    current_picks: List[str],
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Main entry point.
    Returns:
      - avail_df: Available players with EVAL_PTS, VBD, VONA
      - starters: Dict of starting slot counts per position (incl. FLEX)
    """
    # Normalize and ensure minimal columns
    df = ensure_player_cols(raw_df)

    # Optionally recompute PROJ_PTS from component stats using league scoring
    df = recompute_proj_pts_if_components(df, scoring or SCORING_DEFAULT)

    # Apply contextual adjustments (injury, SOS, usage)
    df = apply_context_adjustments(df, weights or {"inj_w": 0.5, "sos_w": 0.05, "usage_w": 0.05})

    # Filter out taken players by normalized key
    taken_keys = set([k for k in (current_picks or []) if isinstance(k, str)])
    avail = df[~df["PLAYER_KEY"].isin(taken_keys)].copy()

    # Starters/slots
    roster_positions = _fallback_starters_if_empty(roster_positions or [])
    starters = starters_from_roster_positions(roster_positions)

    # Add VBD & VONA
    avail = add_vbd_and_vona(avail, teams=int(teams or 12), starters=starters)

    return avail, starters

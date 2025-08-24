from __future__ import annotations
from typing import Dict, List
import numpy as np
import pandas as pd

"""
Strategy-aware (but profile-free) pick ranking.

We build a composite SCORE for each available player using:
- VBD (primary signal)
- VONA (value over next available at the same position by your next pick)
- Roster need bonus (light guardrails so you don’t leave holes)
- Tier cliff bonus (prefer players just before a drop at their position)
- ADP value bonus (mild preference for fallers)
- Injury risk penalty (scaled by sidebar weight)

Return the top K with human-readable REASONS so you can pivot easily.
"""

REASON_LIMIT = 3

# -------------------------
# Component signals
# -------------------------

def _need_bonus(row: pd.Series, need_state: Dict[str, int]) -> float:
    """
    Soft push toward positions you’re light on (starters + 1 depth).
    Each missing slot adds a small constant to the score.
    """
    pos = row["POS"]
    need = int(need_state.get(pos, 0))
    return 3.0 * float(need)  # tuneable constant


def _adp_value_bonus(row: pd.Series) -> float:
    """
    Favor ADP fallers a bit. We don’t know your exact pick number here,
    so we use simple thresholds as a mild ranking nudge.
    """
    adp = row.get("ADP", np.nan)
    if pd.isna(adp):
        return 0.0
    # early-round value
    if adp <= 36:
        return 1.0
    # mid-round value pocket
    if adp <= 60:
        return 0.5
    return 0.0


def _risk_penalty(row: pd.Series, risk_w: float) -> float:
    """
    Penalize players with higher INJURY_VAL (0.0–~0.2 typical).
    Multiplied by a weight from the sidebar to reflect your appetite.
    """
    r = row.get("INJURY_VAL", np.nan)
    if pd.isna(r):
        return 0.0
    return -5.0 * risk_w * float(r)  # tuneable slope


def _tier_bonus(row: pd.Series, df: pd.DataFrame) -> float:
    """
    If the next player at the same position is a meaningful drop in EVAL_PTS,
    reward the current player (he’s at a tier edge).
    """
    pos = row["POS"]
    pos_df = df[df["POS"] == pos].sort_values("EVAL_PTS", ascending=False).reset_index(drop=True)
    idx_list = pos_df.index[pos_df["PLAYER_KEY"] == row["PLAYER_KEY"]].tolist()
    if not idx_list:
        return 0.0
    i = idx_list[0]
    if i + 1 < len(pos_df):
        drop = float(pos_df.iloc[i]["EVAL_PTS"] - pos_df.iloc[i + 1]["EVAL_PTS"])
        # Scale the cliff; cap to avoid overdominance
        return min(5.0, max(0.0, drop / 5.0))
    return 0.0


def _ceiling_flag(row: pd.Series) -> float:
    """
    Optional micro-bump for players likely to deliver spike-weeks.
    Heuristic: higher REDZONE_TGT and GOAL_LINE_SHARE -> tiny boost.
    """
    rz = row.get("REDZONE_TGT", np.nan)
    gl = row.get("GOAL_LINE_SHARE", np.nan)
    rz = 0.0 if pd.isna(rz) else float(rz)
    gl = 0.0 if pd.isna(gl) else float(gl)
    # Very small bump; prevents overpowering VBD/VONA
    return 0.1 * (rz > 15) + 0.1 * (gl > 0.35)


# -------------------------
# Reasons (human-readable)
# -------------------------

def reason_strings(row: pd.Series, need_state: Dict[str, int]) -> List[str]:
    reasons: List[str] = []
    # Always show VBD
    reasons.append(f"VBD +{row['VBD']:.1f}")
    # Show VONA if positive
    if row.get("VONA", 0) > 0:
        reasons.append(f"VONA +{row['VONA']:.1f}")
    # Roster need callout
    if need_state.get(row["POS"], 0) > 0:
        reasons.append(f"Roster need at {row['POS']}")
    # ADP note if available
    adp = row.get("ADP", np.nan)
    if not pd.isna(adp):
        reasons.append(f"ADP {int(adp)}")
    # Trim
    if len(reasons) > REASON_LIMIT:
        reasons = reasons[:REASON_LIMIT]
    return reasons


# -------------------------
# Public API
# -------------------------

def suggest(
    avail_df: pd.DataFrame,
    roster_needs: Dict[str, int],
    weights: Dict[str, float],
    topk: int = 8
) -> pd.DataFrame:
    """
    Rank available players by composite score and return the top-K
    with concise rationale strings for quick decision-making.
    """
    df = avail_df.copy()
    if df.empty:
        return df

    scores = []
    inj_w = float(weights.get("inj_w", 0.5))

    for _, r in df.iterrows():
        score = (
            1.00 * float(r.get("VBD", 0.0)) +           # primary signal
            0.50 * float(r.get("VONA", 0.0)) +          # protection vs. position drying up
            _need_bonus(r, roster_needs) +              # guardrail
            _tier_bonus(r, df) +                        # pre-cliff preference
            _adp_value_bonus(r) +                       # mild value nod
            _risk_penalty(r, risk_w=inj_w) +            # risk tolerance
            _ceiling_flag(r)                            # tiny ceiling nudge
        )
        scores.append(score)

    df["SCORE"] = scores
    df = df.sort_values(["SCORE", "VBD", "EVAL_PTS"], ascending=False)

    # Reasons column
    df["REASONS"] = df.apply(lambda row: "; ".join(reason_strings(row, roster_needs)), axis=1)

    cols = ["PLAYER", "TEAM", "POS", "TIER", "EVAL_PTS", "VBD", "VONA", "ADP", "SCORE", "REASONS", "PLAYER_KEY"]
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan

    return df.head(topk)[cols]

from __future__ import annotations
from typing import Dict, List, Optional
import numpy as np
import pandas as pd

# --- Strategy-aware suggestion scoring ---
_STRAT_MUL = {
    # base multipliers by position; will be tweaked by round/needs
    "Zero RB":        {"RB": 0.92, "WR": 1.10, "TE": 1.06, "QB": 1.02},
    "Modified Zero RB":{"RB": 0.96, "WR": 1.08, "TE": 1.05, "QB": 1.02},
    "Hero RB":        {"RB": 1.12, "WR": 1.02, "TE": 1.00, "QB": 1.00},
    "Robust RB":      {"RB": 1.10, "WR": 0.98, "TE": 0.98, "QB": 0.98},
    "Hyper-Fragile RB":{"RB": 0.90, "WR": 1.06, "TE": 1.03, "QB": 1.00},
    "WR-Heavy":       {"RB": 0.95, "WR": 1.12, "TE": 1.06, "QB": 1.02},
    "Pocket QB":      {"RB": 1.00, "WR": 1.02, "TE": 1.00, "QB": 0.92},
    "Bimodal RB":     {"RB": 1.04, "WR": 1.02, "TE": 1.00, "QB": 1.00},
    "Balanced":       {"RB": 1.00, "WR": 1.00, "TE": 1.00, "QB": 1.00},
}

def _z(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    mu, sd = s.mean(skipna=True), s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - mu) / sd

def _need_multiplier(pos: str, need_count: int) -> float:
    # small but meaningful boost for unmet needs
    if need_count <= 0:
        # slight de-emphasis when a slot is already 'covered'
        return 0.94 if pos in ("QB","TE") else 0.97
    # escalate with need count (e.g., still need 2+ WRs)
    base = 1.00 + 0.18 * min(3, int(need_count))
    if pos in ("QB","TE"):
        base += 0.07  # single-slot positions need a little more push
    return float(base)

def _round_phase_multiplier(strategy: str, pos: str, round_number: Optional[int], total_rounds: Optional[int]) -> float:
    if round_number is None or total_rounds is None:
        return 1.0
    r = int(round_number); R = int(total_rounds)
    early = r <= max(3, R // 5)
    mid   = (R // 5) < r <= (3 * R) // 5

    m = 1.0
    if strategy == "Zero RB":
        if early and pos == "RB": m *= 0.88
        if early and pos in ("WR","TE"): m *= 1.10
        if mid and pos == "RB": m *= 1.04
    elif strategy == "Modified Zero RB":
        if early and pos == "RB": m *= 0.93
        if early and pos in ("WR","TE"): m *= 1.08
        if mid and pos == "RB": m *= 1.06
    elif strategy == "Hero RB":
        if early and pos == "RB": m *= 1.14
        if not early and pos == "RB": m *= 0.90
    elif strategy == "Robust RB":
        if early and pos == "RB": m *= 1.10
    elif strategy == "Hyper-Fragile RB":
        if not early and pos == "RB": m *= 0.88
    elif strategy == "WR-Heavy":
        if early and pos == "WR": m *= 1.12
        if early and pos == "RB": m *= 0.94
    elif strategy == "Pocket QB":
        if early and pos == "QB": m *= 0.85
        if mid   and pos == "QB": m *= 1.08
    elif strategy == "Bimodal RB":
        if mid and pos == "RB": m *= 1.08
    return float(m)

def suggest(
    avail_df: pd.DataFrame,
    need_by_pos: Dict[str, int],
    weights: Dict[str, float],
    topk: int = 8,
    strategy_name: Optional[str] = None,
    round_number: Optional[int] = None,
    total_rounds: Optional[int] = None,
) -> pd.DataFrame:
    """
    Return avail_df with a 'score' column, ranked for suggestion.
    The caller (app) handles plain-English explanations & probability to return.
    """
    if avail_df is None or avail_df.empty:
        return pd.DataFrame([])

    df = avail_df.copy()

    # Normalized components (global z)
    z_vbd  = _z(df.get("VBD", 0.0))
    z_eval = _z(df.get("EVAL_PTS", 0.0))

    # Value delta: if both ADP & ECR exist, good if ECR < ADP (draft value)
    ecr = pd.to_numeric(df.get("ECR", np.nan), errors="coerce")
    adp = pd.to_numeric(df.get("ADP", np.nan), errors="coerce")
    val_delta = pd.Series(np.zeros(len(df)), index=df.index)
    if "ECR" in df.columns and "ADP" in df.columns:
        # lower ECR vs ADP -> value
        val_delta = _z(adp - ecr)  # positive if ADP >> ECR (discount vs rank)

    # base score
    score = 0.58 * z_vbd + 0.34 * z_eval + 0.08 * val_delta

    # Apply needs & strategy multipliers
    pos = df["POS"].astype(str).str.upper()
    strat = strategy_name or "Balanced"
    base_mul_map = _STRAT_MUL.get(strat, _STRAT_MUL["Balanced"])

    pos_mul = []
    for i in range(len(df)):
        p = pos.iat[i]
        need_cnt = int(need_by_pos.get(p, 0))
        m = base_mul_map.get(p, 1.0)
        m *= _need_multiplier(p, need_cnt)
        m *= _round_phase_multiplier(strat, p, round_number, total_rounds)
        pos_mul.append(m)

    score = score * pd.Series(pos_mul, index=df.index)

    df["score"] = score.astype(float)
    df = df.sort_values(["score","VBD","EVAL_PTS"], ascending=False).reset_index(drop=True)
    return df.head(topk)

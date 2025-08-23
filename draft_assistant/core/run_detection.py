from __future__ import annotations
from collections import Counter
from typing import List, Dict, Tuple, Iterable, Optional
import pandas as pd
import numpy as np

# -----------------------------
# Position run detection
# -----------------------------

def detect_runs(
    pos_history: List[str],
    lookback: int = 10,
    threshold: int = 3,
) -> List[str]:
    """
    Simple run detector: if >= threshold picks of the same position occurred
    within the last `lookback` selections, emit an alert.
    """
    if not pos_history:
        return []
    recent = [p for p in pos_history[-lookback:] if isinstance(p, str)]
    counts = Counter(recent)
    alerts = []
    for pos, n in counts.items():
        if n >= threshold:
            alerts.append(f"Run detected: {n}/{lookback} recent picks were {pos}.")
    return alerts


def recent_run_lengths(pos_history: List[str]) -> Dict[str, int]:
    """
    For each position, measure the contiguous run length from the end of the history.
    e.g., history [..., RB, RB, WR] -> {'WR': 1}, but also tracks the last-pos run only.
    """
    out: Dict[str, int] = {}
    if not pos_history:
        return out
    last = pos_history[-1]
    length = 0
    for p in reversed(pos_history):
        if p == last:
            length += 1
        else:
            break
    out[last] = length
    return out


# -----------------------------
# Tier evaporation / cliff detection
# -----------------------------

def _safe_int_tier(x) -> Optional[int]:
    try:
        if pd.isna(x):
            return None
        # Some sheets store TIER as float; treat 2.0 -> 2
        v = int(float(x))
        return v
    except Exception:
        return None


def detect_tier_evaporation(
    avail_df: pd.DataFrame,
    positions: Iterable[str] = ("RB", "WR", "TE", "QB"),
    min_left: int = 1,
) -> List[str]:
    """
    Alert when the *best remaining tier* at a position is nearly empty.

    Rules:
      - Determine the lowest (best) tier number still available for each position.
      - If count of players in that tier <= min_left, raise an alert.
      - If possible, include an estimated drop (avg EVAL_PTS diff to next tier).
    """
    alerts: List[str] = []
    if avail_df is None or avail_df.empty or "TIER" not in avail_df.columns:
        return alerts

    df = avail_df.copy()
    df["TIER_INT"] = df["TIER"].map(_safe_int_tier)

    for pos in positions:
        pos_df = df[df["POS"] == pos]
        if pos_df.empty:
            continue
        tiers = sorted(t for t in pos_df["TIER_INT"].dropna().unique())
        if not tiers:
            continue
        best = tiers[0]
        cnt_best = int((pos_df["TIER_INT"] == best).sum())

        # Compute average EVAL_PTS by tier to estimate the cliff
        drop_txt = ""
        if "EVAL_PTS" in pos_df.columns:
            tier_means = (
                pos_df.dropna(subset=["TIER_INT"])
                .groupby("TIER_INT")["EVAL_PTS"]
                .mean()
                .sort_index()
            )
            if best in tier_means.index:
                next_tiers = [t for t in tier_means.index if t > best]
                if next_tiers:
                    next_t = next_tiers[0]
                    drop = float(tier_means.loc[best] - tier_means.loc[next_t])
                    if drop > 0:
                        drop_txt = f" (≈{drop:.1f} pts drop to Tier {next_t})"

        if cnt_best <= min_left:
            suffix = "remaining" if cnt_best == 1 else "remaining"
            alerts.append(f"{pos} Tier {best} nearly gone — {cnt_best} {suffix}{drop_txt}.")

    return alerts


def detect_all_alerts(
    pos_history: List[str],
    avail_df: pd.DataFrame,
    lookback: int = 10,
    threshold: int = 3,
    positions: Iterable[str] = ("RB", "WR", "TE", "QB"),
    min_left: int = 1,
) -> List[str]:
    """
    Convenience wrapper: combine run and tier-evaporation alerts.
    """
    alerts = []
    alerts += detect_runs(pos_history, lookback=lookback, threshold=threshold)
    alerts += detect_tier_evaporation(avail_df, positions=positions, min_left=min_left)
    return alerts

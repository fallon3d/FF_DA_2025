# draft_assistant/core/suggestions.py
from __future__ import annotations

from typing import Dict, Optional
import numpy as np
import pandas as pd

# Stable, simple suggestion ranking:
# - VBD-first with small need bump
# - Suppress early QBs (before R6) unless overwhelming VBD edge
# - Prefer K/DEF only in last rounds (toggle)

QB_SUPPRESS_UNTIL_ROUND = 6
QB_OVERWHELMING_EDGE = 20.0  # allow earlier if QB VBD beats best non-QB by this many points

def _best_non_qb_vbd(df: pd.DataFrame) -> float:
    mx = -1e9
    for p in ("RB","WR","TE"):
        sub = df[df["POS"] == p]
        if not sub.empty:
            v = float(pd.to_numeric(sub["VBD"], errors="coerce").max())
            mx = max(mx, v)
    return 0.0 if mx == -1e9 else mx

def suggest(
    avail_df: pd.DataFrame,
    base_need: Dict[str,int],
    round_number: int,
    total_rounds: int,
    strategy_name: Optional[str] = "Balanced",
    qb_cap: int = 2,
    k_def_last_rounds_only: bool = True,
) -> pd.DataFrame:
    if avail_df is None or avail_df.empty:
        return pd.DataFrame(columns=["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD","WHY"])

    df = avail_df.copy()
    # Early QB suppression
    if round_number < QB_SUPPRESS_UNTIL_ROUND:
        best_non_qb = _best_non_qb_vbd(df)
        df_qb = df[df["POS"] == "QB"].copy()
        if not df_qb.empty:
            top_qb = float(pd.to_numeric(df_qb["VBD"], errors="coerce").max())
            if top_qb < best_non_qb + QB_OVERWHELMING_EDGE:
                df = df[df["POS"] != "QB"]

    # K/DEF late preference
    if k_def_last_rounds_only and round_number < max(1, total_rounds - 1):
        df = df[~df["POS"].isin(["K","DEF"])]

    # Need bump (+2 if we still need that position), light penalty otherwise
    need_adj = []
    for _, r in df.iterrows():
        pos = str(r.get("POS") or "")
        need_adj.append(2.0 if base_need.get(pos, 0) > 0 else -0.5)
    df["__NEED__"] = need_adj

    # Final score: VBD + need
    vbd = pd.to_numeric(df["VBD"], errors="coerce").fillna(0.0)
    df["score"] = vbd + df["__NEED__"]

    # QB cap at display time (don't show more than remaining QB quota)
    if qb_cap <= 0 or base_need.get("QB", 0) <= 0:
        df = pd.concat([df[df["POS"] != "QB"], df[df["POS"] == "QB"].head(0)], ignore_index=True)
    else:
        have_qb = 0  # handled by base_need (remaining desired QBs)
        remaining = max(0, qb_cap - max(0, have_qb))
        non = df[df["POS"] != "QB"]
        qbs = df[df["POS"] == "QB"].head(remaining)
        df = pd.concat([non, qbs], ignore_index=True)

    df = df.sort_values(["score", "VBD", "EVAL_PTS"], ascending=False)

    # Plain-English WHY
    whys = []
    for _, r in df.iterrows():
        why = f"VBD {float(r['VBD']):.1f}"
        if base_need.get(r["POS"], 0) > 0:
            why += f"; you still need {r['POS']}"
        whys.append(why)
    df["WHY"] = whys

    out_cols = ["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD","WHY","score"]
    return df[out_cols]

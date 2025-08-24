from __future__ import annotations
import os
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

# --- Tunables (env overrides supported) ---
QB_SUPPRESS_UNTIL_ROUND = int(os.getenv("FFDA_QB_SUPPRESS_UNTIL_ROUND", "6"))    # hide QB until this round
QB_EARLY_EDGE_GATE      = float(os.getenv("FFDA_QB_EARLY_EDGE_GATE", "22.0"))    # allow early QB only if QB VBD exceeds best non-QB by this
QB_STRICT_POCKET_ROUND  = int(os.getenv("FFDA_QB_STRICT_POCKET_ROUND", "7"))     # even stricter for "Pocket QB"
K_DEF_EARLY_PENALTY_RND = int(os.getenv("FFDA_K_DEF_EARLY_PENALTY_RND", "14"))   # discourage K/DEF before final two rounds
SCARCITY_TIER_SIZE      = int(os.getenv("FFDA_SCARCITY_TIER_SIZE", "2"))         # if <= this many left in tier, add scarcity bump
SCARCITY_BUMP           = float(os.getenv("FFDA_SCARCITY_BUMP", "8.0"))
NEED_BUMP               = float(os.getenv("FFDA_NEED_BUMP", "3.0"))
NO_NEED_PEN             = float(os.getenv("FFDA_NO_NEED_PEN", "1.5"))

def _z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (s - mu) / sd

def _top_vbd(avail_df: pd.DataFrame, pos: str) -> float:
    pool = avail_df[avail_df["POS"] == pos]
    if pool.empty:
        return 0.0
    srt = pool.sort_values(["VBD","EVAL_PTS"], ascending=False)
    return float(pd.to_numeric(srt.iloc[0].get("VBD"), errors="coerce") or 0.0)

def _best_non_qb_edge(avail_df: pd.DataFrame) -> float:
    edges = []
    for p in ("RB","WR","TE"):
        edges.append(_top_vbd(avail_df, p))
    return float(max(edges) if edges else 0.0)

def _apply_position_needs_score(df: pd.DataFrame, base_need: Dict[str,int]) -> pd.DataFrame:
    # small bump if we still need that position; small penalty if we don't
    need_adj = []
    for _, r in df.iterrows():
        pos = str(r.get("POS") or "")
        need_adj.append(NEED_BUMP if base_need.get(pos, 0) > 0 else -NO_NEED_PEN)
    out = df.copy()
    out["__NEED__"] = need_adj
    return out

def _scarcity_bump(df: pd.DataFrame) -> pd.DataFrame:
    """If player is in a small remaining tier (<= SCARCITY_TIER_SIZE), add a bump."""
    out = df.copy()
    out["__SCARCITY__"] = 0.0
    if "TIER" not in df.columns:
        return out
    for pos in ("RB","WR","TE","QB"):
        sub = out[out["POS"] == pos]
        if sub.empty: 
            continue
        for tier, tdf in sub.groupby("TIER"):
            try:
                tier_left = len(tdf)
            except Exception:
                tier_left = 0
            if pd.isna(tier) or tier_left == 0:
                continue
            if tier_left <= SCARCITY_TIER_SIZE:
                out.loc[tdf.index, "__SCARCITY__"] = SCARCITY_BUMP
    return out

def _gate_qb_early(df: pd.DataFrame, avail_df: pd.DataFrame, round_number: int, strategy_name: Optional[str]) -> pd.DataFrame:
    """Suppress QB in early rounds unless the edge is overwhelming."""
    if df.empty:
        return df

    qb_round_gate = QB_SUPPRESS_UNTIL_ROUND
    if str(strategy_name or "").lower() == "pocket qb":
        qb_round_gate = max(qb_round_gate, QB_STRICT_POCKET_ROUND)

    qb_edge = _top_vbd(avail_df, "QB")
    nonqb_edge = _best_non_qb_edge(avail_df)

    qb_allowed_early = (round_number >= qb_round_gate) or (qb_edge >= nonqb_edge + QB_EARLY_EDGE_GATE)

    gated = df.copy()
    gated["__QB_GATE__"] = 0.0
    if not qb_allowed_early:
        gated.loc[gated["POS"] == "QB", "__QB_GATE__"] = -999.0
    return gated

def _discourage_k_def_too_early(df: pd.DataFrame, round_number: int, total_rounds: int) -> pd.DataFrame:
    """Heavily discourage K/DEF until the final two rounds."""
    if df.empty:
        return df
    gated = df.copy()
    gated["__KDEF_PEN__"] = 0.0
    if round_number < max(1, total_rounds - 1):
        if round_number < K_DEF_EARLY_PENALTY_RND:
            gated.loc[gated["POS"].isin(["K","DEF"]), "__KDEF_PEN__"] = -250.0
    return gated

def suggest(
    avail_df: pd.DataFrame,
    base_need: Dict[str,int],
    weights: Dict[str, float] | None = None,
    topk: int = 8,
    strategy_name: Optional[str] = None,
    round_number: int = 1,
    total_rounds: int = 15,
) -> pd.DataFrame:
    """
    Rank candidates for the current pick. Returns df with 'score' column, sorted desc.

    Core rules:
      - VBD-first scoring (with a small EVAL_PTS stabilizer).
      - Tier scarcity bump (if <= SCARCITY_TIER_SIZE remain in their tier).
      - Needs-aware nudge.
      - **Hard early QB suppression** unless the edge is truly massive.
      - **K/DEF discouraged** until late (final two rounds).
    """
    if avail_df is None or avail_df.empty:
        return pd.DataFrame(columns=["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD","score"])

    cand = avail_df.copy()

    # Base scoring: mostly VBD, with a small EVAL_PTS stabilizer
    vbd = pd.to_numeric(cand.get("VBD"), errors="coerce").fillna(0.0)
    eval_z = _z(cand.get("EVAL_PTS")) if "EVAL_PTS" in cand.columns else pd.Series(np.zeros(len(cand)), index=cand.index)
    cand["__BASE__"] = (0.88 * vbd) + (0.12 * eval_z)

    # Needs & scarcity
    cand = _apply_position_needs_score(cand, base_need)
    cand = _scarcity_bump(cand)

    # Gating
    cand = _gate_qb_early(cand, avail_df, round_number, strategy_name)
    cand = _discourage_k_def_too_early(cand, round_number, total_rounds)

    # Final score
    for col in ("__QB_GATE__", "__KDEF_PEN__", "__NEED__", "__SCARCITY__", "__BASE__"):
        if col not in cand.columns:
            cand[col] = 0.0
    cand["score"] = cand["__BASE__"] + cand["__NEED__"] + cand["__SCARCITY__"] + cand["__QB_GATE__"] + cand["__KDEF_PEN__"]

    out_cols = ["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD","score"]
    cand = cand.sort_values("score", ascending=False)
    return cand[out_cols].head(int(topk)).reset_index(drop=True)

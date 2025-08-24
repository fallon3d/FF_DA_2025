from __future__ import annotations
import os
from typing import Dict, Optional
import numpy as np
import pandas as pd

# Tunables (stable)
QB_SUPPRESS_UNTIL_ROUND = int(os.getenv("FFDA_QB_SUPPRESS_UNTIL_ROUND", "6"))
QB_EARLY_EDGE_GATE      = float(os.getenv("FFDA_QB_EARLY_EDGE_GATE", "22.0"))
QB_STRICT_POCKET_ROUND  = int(os.getenv("FFDA_QB_STRICT_POCKET_ROUND", "7"))
K_DEF_EARLY_PENALTY_RND = int(os.getenv("FFDA_K_DEF_EARLY_PENALTY_RND", "14"))
SCARCITY_TIER_SIZE      = int(os.getenv("FFDA_SCARCITY_TIER_SIZE", "2"))
SCARCITY_BUMP           = float(os.getenv("FFDA_SCARCITY_BUMP", "8.0"))
NEED_BUMP               = float(os.getenv("FFDA_NEED_BUMP", "3.0"))
NO_NEED_PEN             = float(os.getenv("FFDA_NO_NEED_PEN", "1.5"))

def _z(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True); sd = s.std(skipna=True)
    if not np.isfinite(sd) or sd == 0: return pd.Series(np.zeros(len(series)), index=series.index)
    return (s - mu) / sd

def _top_vbd(avail_df: pd.DataFrame, pos: str) -> float:
    pool = avail_df[avail_df["POS"] == pos]
    if pool.empty: return 0.0
    srt = pool.sort_values(["VBD","EVAL_PTS"], ascending=False)
    return float(pd.to_numeric(srt.iloc[0].get("VBD"), errors="coerce") or 0.0)

def _best_non_qb_edge(avail_df: pd.DataFrame) -> float:
    return float(max((_top_vbd(avail_df, p) for p in ("RB","WR","TE")), default=0.0))

def _apply_position_needs_score(df: pd.DataFrame, base_need: Dict[str,int]) -> pd.DataFrame:
    out = df.copy()
    out["__NEED__"] = [NEED_BUMP if base_need.get(str(r.get("POS") or ""), 0) > 0 else -NO_NEED_PEN for _, r in df.iterrows()]
    return out

def _scarcity_bump(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy(); out["__SCARCITY__"] = 0.0
    if "TIER" not in df.columns: return out
    for pos in ("RB","WR","TE","QB"):
        sub = out[out["POS"] == pos]
        if sub.empty: continue
        for tier, tdf in sub.groupby("TIER"):
            if pd.isna(tier): continue
            if len(tdf) <= SCARCITY_TIER_SIZE:
                out.loc[tdf.index, "__SCARCITY__"] = SCARCITY_BUMP
    return out

def _gate_qb_early(df: pd.DataFrame, avail_df: pd.DataFrame, round_number: int, strategy_name: Optional[str]) -> pd.DataFrame:
    if df.empty: return df
    qb_round_gate = QB_SUPPRESS_UNTIL_ROUND
    if str(strategy_name or "").lower() == "pocket qb":
        qb_round_gate = max(qb_round_gate, QB_STRICT_POCKET_ROUND)
    qb_edge = _top_vbd(avail_df, "QB"); nonqb_edge = _best_non_qb_edge(avail_df)
    qb_allowed_early = (round_number >= qb_round_gate) or (qb_edge >= nonqb_edge + QB_EARLY_EDGE_GATE)
    gated = df.copy(); gated["__QB_GATE__"] = 0.0
    if not qb_allowed_early: gated.loc[gated["POS"] == "QB", "__QB_GATE__"] = -999.0
    return gated

def _discourage_k_def_too_early(df: pd.DataFrame, round_number: int, total_rounds: int) -> pd.DataFrame:
    if df.empty: return df
    gated = df.copy(); gated["__KDEF_PEN__"] = 0.0
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
    if avail_df is None or avail_df.empty:
        return pd.DataFrame(columns=["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD","score"])

    cand = avail_df.copy()

    # Base: VBD primary, with tiny EVAL_PTS stabilizer
    vbd = pd.to_numeric(cand.get("VBD"), errors="coerce").fillna(0.0)
    eval_z = _z(cand.get("EVAL_PTS")) if "EVAL_PTS" in cand.columns else pd.Series(np.zeros(len(cand)), index=cand.index)
    cand["__BASE__"] = (0.88 * vbd) + (0.12 * eval_z)

    cand = _apply_position_needs_score(cand, base_need)
    cand = _scarcity_bump(cand)

    cand = _gate_qb_early(cand, avail_df, round_number, strategy_name)
    cand = _discourage_k_def_too_early(cand, round_number, total_rounds)

    for col in ("__QB_GATE__", "__KDEF_PEN__", "__NEED__", "__SCARCITY__", "__BASE__"):
        if col not in cand.columns:
            cand[col] = 0.0
    cand["score"] = cand["__BASE__"] + cand["__NEED__"] + cand["__SCARCITY__"] + cand["__QB_GATE__"] + cand["__KDEF_PEN__"]

    out_cols = ["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD","score"]
    cand = cand.sort_values("score", ascending=False)
    return cand[out_cols].head(int(topk)).reset_index(drop=True)

# Helpers used by app (keep here so app can import without circulars)
def _apply_qb_cap(sugg_df: pd.DataFrame, qbs_owned: int, cap: int) -> pd.DataFrame:
    if sugg_df is None or sugg_df.empty: return sugg_df
    remaining = max(0, cap - max(0, qbs_owned))
    qbs = sugg_df[sugg_df["POS"]=="QB"]
    non = sugg_df[sugg_df["POS"]!="QB"]
    if remaining <= 0:
        return non.head(len(sugg_df)).reset_index(drop=True)
    return pd.concat([non, qbs.head(remaining)], ignore_index=True).head(len(sugg_df)).reset_index(drop=True)

def _ensure_k_def_in_suggestions(sugg_df: pd.DataFrame, avail_df: pd.DataFrame, rnd: int, total_rounds: int, include_anytime: bool, need_k: int, need_def: int) -> pd.DataFrame:
    if sugg_df is None or sugg_df.empty: return sugg_df
    have_k = (sugg_df["POS"]=="K").any()
    have_d = (sugg_df["POS"]=="DEF").any()
    force_window = rnd >= total_rounds - 1 or (rnd >= total_rounds - 2 and (need_k > 0 or need_def > 0))
    if (have_k and have_d) and not force_window: return sugg_df
    top_k = avail_df[avail_df["POS"]=="K"].sort_values(["VBD","EVAL_PTS"], ascending=False).head(1)
    top_d = avail_df[avail_df["POS"]=="DEF"].sort_values(["VBD","EVAL_PTS"], ascending=False).head(1)
    base = sugg_df.copy()
    tail_vbd = float(base["VBD"].iloc[min(len(base)-1, 7)]) if "VBD" in base.columns and not base.empty else 0.0
    candidates = []
    if need_def > 0 and not top_d.empty and (force_window or (include_anytime and float(top_d.iloc[0]["VBD"]) >= tail_vbd - 15)):
        candidates.append(top_d.iloc[0])
    if need_k > 0 and not top_k.empty and (force_window or (include_anytime and float(top_k.iloc[0]["VBD"]) >= tail_vbd - 15)):
        candidates.append(top_k.iloc[0])
    if candidates:
        base = pd.concat([base, pd.DataFrame(candidates)], ignore_index=True)
        base = base.drop_duplicates(subset=["PLAYER"], keep="first").sort_values(["VBD","EVAL_PTS"], ascending=False).head(8).reset_index(drop=True)
    return base

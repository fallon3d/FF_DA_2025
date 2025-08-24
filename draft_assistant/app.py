import os
import sys
import math
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

# --- ensure repo root is importable when running from package path ---
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
# --------------------------------------------------------------------

from draft_assistant.core import sleeper
from draft_assistant.core.evaluation import evaluate_players, SCORING_DEFAULT
from draft_assistant.core.suggestions import suggest
from draft_assistant.core.utils import (
    norm_name,
    read_player_table,
    snake_position,
    starters_from_roster_positions,
    slot_to_display_name,
)

st.set_page_config(page_title="FF Draft Assistant — Dynamic Strategy (VBD)", layout="wide")

DEFAULT_CSV_PATH = os.environ.get("FFDA_CSV_PATH", "")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

QB_ROSTER_CAP = int(os.environ.get("FFDA_QB_CAP", "2"))
INCLUDE_K_DEF_EARLY = bool(int(os.environ.get("FFDA_INCLUDE_K_DEF_EARLY", "0")))

# =========================
# Cache
# =========================

@st.cache_resource(show_spinner=False)
def sleeper_players_cache():
    try:
        return sleeper.get_players_nfl()
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def load_local_csv(path: str):
    return read_player_table(path)

# =========================
# Sidebar
# =========================

def sidebar_controls():
    st.sidebar.header("Data")
    src = st.sidebar.radio("Player data source", ["Upload", "Local path"])
    csv_df = None
    if src == "Upload":
        up = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx", "xls"])
        if up is not None:
            csv_df = read_player_table(up)
    else:
        path = st.sidebar.text_input("CSV/Excel path", value=DEFAULT_CSV_PATH)
        if st.sidebar.button("Load path") and path:
            csv_df = load_local_csv(path)

    st.sidebar.header("Weights")
    inj_w = st.sidebar.slider("Injury penalty weight", 0.0, 1.0, 0.5, 0.05)
    sos_w = st.sidebar.slider("Schedule strength weight", 0.0, 0.3, 0.05, 0.01)
    usage_w = st.sidebar.slider("Usage/Upside weight", 0.0, 0.5, 0.05, 0.01)
    weights = {"inj_w": inj_w, "sos_w": sos_w, "usage_w": usage_w}

    st.sidebar.header("Sleeper (Live)")
    league_id = st.sidebar.text_input("League ID", value="")
    username = st.sidebar.text_input("Your Sleeper username (or display name)", value="Fallon3D")
    seat = st.sidebar.number_input("Your draft slot (1–Teams; 0=auto)", min_value=0, max_value=20, value=0)

    poll_secs = st.sidebar.slider("Auto-refresh seconds", 3, 30, 5, 1)
    auto_live = st.sidebar.toggle("Auto-refresh (Live tab)", value=False)

    st.sidebar.header("K/DEF")
    include_k_def_anytime = st.sidebar.checkbox(
        "Allow K & DEF to appear anytime if value is high",
        value=INCLUDE_K_DEF_EARLY
    )

    if st.sidebar.button("Reset dynamic memory"):
        for k in ["prev_strategy"]:
            st.session_state.pop(k, None)
        st.sidebar.success("Strategy will be re-evaluated fresh on your next pick.")

    return csv_df, weights, league_id, username, int(seat), poll_secs, auto_live, include_k_def_anytime

# =========================
# Helpers
# =========================

def compute_next_pick_window(teams: int, seat: int, current_overall_pick: int) -> int:
    if not (1 <= seat <= teams):
        return teams
    rnd = (current_overall_pick - 1) // teams + 1
    pos = (current_overall_pick - 1) % teams + 1
    my_pos_this = seat if rnd % 2 == 1 else (teams - seat + 1)
    if my_pos_this >= pos:
        return my_pos_this - pos
    my_pos_next = (teams - seat + 1) if rnd % 2 == 1 else seat
    return (teams - pos) + my_pos_next

def _pos_from_meta(p: dict) -> str:
    return str((p.get("metadata") or {}).get("position") or "").upper().replace("DST","DEF")

def _team_pos_counts_from_log(pick_log: List[dict], teams: int) -> Dict[int, Dict[str,int]]:
    counts = {slot: {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0} for slot in range(1, teams+1)}
    for p in pick_log or []:
        try:
            slot = int(p.get("slot") or 0)
        except Exception:
            slot = 0
        if not (1 <= slot <= teams):
            continue
        pos = _pos_from_meta(p)
        if pos in counts[slot]:
            counts[slot][pos] += 1
    return counts

def _slots_between(current_overall: int, my_next_overall: int, teams: int) -> List[int]:
    slots = []
    pick = current_overall
    while pick < my_next_overall:
        _, _, slot = snake_position(pick, teams)
        if slot not in slots:
            slots.append(slot)
        pick += 1
    return slots

def _needs_by_slot(team_counts: Dict[int,Dict[str,int]], starters: Dict[str,int]) -> Dict[int,Dict[str,int]]:
    needs = {}
    for slot, cmap in team_counts.items():
        needs[slot] = {k: 0 for k in ["QB","RB","WR","TE","K","DEF"]}
        for pos in ["QB","RB","WR","TE","K","DEF"]:
            base = starters.get(pos, 0)
            if pos in ("QB","TE","K","DEF"):
                base = max(1, base)
            needs[slot][pos] = max(0, base - cmap.get(pos, 0))
    return needs

def _english_injury(x) -> str:
    if pd.isna(x): return "low"
    try:
        x = float(x)
    except Exception:
        s = str(x).strip().lower()
        if s in ("low","l"): return "low"
        if s in ("med","moderate","m"): return "moderate"
        if s in ("high","h"): return "high"
        return "low"
    if x <= 0.07: return "low"
    if x <= 0.14: return "moderate"
    return "high"

def _make_it_back_probability(row: pd.Series, picks_until_next: int, demand_ratio: float, current_overall: int) -> float:
    scale = 6.0
    adp = row.get("ADP")
    if pd.isna(adp) or adp <= 0:
        adp = current_overall + 12.0
    adp_gap = float(adp) - float(current_overall)
    x = (picks_until_next - adp_gap) / scale
    demand_multiplier = min(1.4, max(0.7, 0.7 + demand_ratio * 0.7))
    p_taken = 1.0 / (1.0 + math.exp(-x))
    p_taken *= demand_multiplier
    p_taken = max(0.0, min(1.0, p_taken))
    return float(max(0.0, min(1.0, 1.0 - p_taken)))

def _reason_plain_english(row: pd.Series, need_for_pos: int, prob_back: float, next_picks: int) -> str:
    bits = []
    bits.append(f"Strong {row['POS']} value (VBD {row.get('VBD',0):.1f}).")
    if not pd.isna(row.get("TIER")):
        bits.append(f"Tier {int(row['TIER'])}.")
    if need_for_pos > 0:
        bits.append(f"You still need {need_for_pos} at {row['POS']}.")
    risk_txt = "likely" if prob_back >= 0.65 else ("50/50" if prob_back >= 0.35 else "unlikely")
    bits.append(f"{risk_txt} to make it {next_picks} picks back ({prob_back*100:.0f}%).")
    inj = _english_injury(row.get("INJURY_RISK"))
    if inj != "low":
        bits.append(f"Injury risk {inj}.")
    if not pd.isna(row.get("BYE")) and int(row.get("BYE") or 0) > 0:
        bits.append(f"Bye {int(row['BYE'])}.")
    return " ".join(bits)

def _recent_runs(pick_log: List[dict], window: int = 8) -> Dict[str,int]:
    tail = (pick_log or [])[-window:]
    counts = {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0}
    for p in tail:
        pos = _pos_from_meta(p)
        if pos in counts:
            counts[pos] += 1
    return counts

def _tier_depth(avail_df: pd.DataFrame, pos: str) -> Tuple[int, float]:
    pool = avail_df[avail_df["POS"] == pos]
    if pool.empty:
        return 0, 0.0
    top = pool.sort_values(["VBD","EVAL_PTS"], ascending=False)
    t = top.iloc[0].get("TIER")
    if pd.isna(t):
        return min(9, len(top)), float(top.iloc[0].get("VBD", 0.0))
    depth = int(pool[pool["TIER"] == t].shape[0])
    edge = float(top.iloc[0].get("VBD", 0.0))
    return depth, edge

# ---------- slot detection ----------
def _resolve_user_id(users: List[dict], username_or_display: str) -> Optional[str]:
    want = (username_or_display or "").strip().lower()
    if not want:
        return None
    for u in users or []:
        if str(u.get("username","")).lower() == want:
            return u.get("user_id")
        if str(u.get("display_name","")).lower() == want:
            return u.get("user_id")
    return None

def _detect_my_slot(users: List[dict], draft_meta: dict, pick_log: List[dict], seat_override: int, username: str) -> int:
    # 1) manual override
    if int(seat_override) > 0:
        return int(seat_override)

    # 2) via draft.draft_order (user_id -> slot)
    my_uid = _resolve_user_id(users, username)
    draft_order = draft_meta.get("draft_order") if isinstance(draft_meta, dict) else None
    if my_uid and isinstance(draft_order, dict):
        slot = draft_order.get(my_uid)
        if slot:
            try:
                return int(slot)
            except Exception:
                pass

    # 3) via pick log (first pick with team == my user_id)
    if my_uid:
        for p in pick_log or []:
            if str(p.get("team") or "") == str(my_uid):
                try:
                    s = int(p.get("slot", 0))
                    if s > 0:
                        return s
                except Exception:
                    pass

    # fallback
    return 1

# ---------- NEW: direct raw counter fallback for owned counts ----------
def _count_owned_for_slot_raw(raw_picks: List[dict], players_map: dict, my_slot: int) -> Dict[str,int]:
    """
    Fallback counter: if team map returns zeros, count my owned positions
    directly from the raw Sleeper picks using draft_slot/roster_id.
    """
    counts = {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0}
    if my_slot <= 0:
        return counts

    for p in raw_picks or []:
        # prefer draft_slot; fallback to roster_id
        slot = p.get("draft_slot", p.get("roster_id", 0))
        try:
            slot = int(slot)
        except Exception:
            slot = 0
        if slot != my_slot:
            continue

        meta = p.get("metadata") or {}
        pos = (meta.get("position") or "").strip().upper()
        if not pos:
            pid = p.get("player_id")
            if pid and players_map:
                pm = players_map.get(pid) or {}
                pos = (pm.get("position") or "").strip().upper()

        # normalize defense labels
        if pos in ("DST","D/ST","DEFENSE","TEAM D","TEAM DEF"):
            pos = "DEF"

        if pos in counts:
            counts[pos] += 1

    return counts

# =========================
# K/DEF + QB cap helpers
# =========================

def _ensure_k_def_in_suggestions(
    sugg_df: pd.DataFrame,
    avail_df: pd.DataFrame,
    rnd: int,
    total_rounds: int,
    include_anytime: bool,
    need_k: int,
    need_def: int,
) -> pd.DataFrame:
    if sugg_df is None or sugg_df.empty:
        return sugg_df
    have_k = (sugg_df["POS"]=="K").any()
    have_d = (sugg_df["POS"]=="DEF").any()

    force_window = rnd >= total_rounds - 1 or (rnd >= total_rounds - 2 and (need_k > 0 or need_def > 0))
    want_force = force_window

    if have_k and have_d and not want_force:
        return sugg_df

    top_k = avail_df[avail_df["POS"]=="K"].sort_values(["VBD","EVAL_PTS"], ascending=False).head(1)
    top_d = avail_df[avail_df["POS"]=="DEF"].sort_values(["VBD","EVAL_PTS"], ascending=False).head(1)

    base = sugg_df.copy()
    tail_vbd = float(base["VBD"].iloc[min(len(base)-1, 7)]) if "VBD" in base.columns and not base.empty else 0.0

    candidates = []
    if need_def > 0 and not top_d.empty and (want_force or (include_anytime and float(top_d.iloc[0]["VBD"]) >= tail_vbd - 15)):
        candidates.append(top_d.iloc[0])
    if need_k > 0 and not top_k.empty and (want_force or (include_anytime and float(top_k.iloc[0]["VBD"]) >= tail_vbd - 15)):
        candidates.append(top_k.iloc[0])

    if candidates:
        base = pd.concat([base, pd.DataFrame(candidates)], ignore_index=True)
        base = base.drop_duplicates(subset=["PLAYER"], keep="first").sort_values(["VBD","EVAL_PTS"], ascending=False).head(8).reset_index(drop=True)
    return base

def _apply_qb_cap(sugg_df: pd.DataFrame, qbs_owned: int, cap: int) -> pd.DataFrame:
    if sugg_df is None or sugg_df.empty: return sugg_df
    remaining = max(0, cap - max(0, qbs_owned))
    qbs = sugg_df[sugg_df["POS"]=="QB"]
    non = sugg_df[sugg_df["POS"]!="QB"]
    if remaining <= 0:
        return non.head(len(sugg_df)).reset_index(drop=True)
    return pd.concat([non, qbs.head(remaining)], ignore_index=True).head(len(sugg_df)).reset_index(drop=True)

# =========================
# Dynamic multi-strategy chooser
# =========================

STRATS = [
    "Zero RB", "Modified Zero RB", "Hero RB", "Robust RB",
    "Hyper-Fragile RB", "WR-Heavy", "Pocket QB", "Bimodal RB", "Balanced"
]
STRAT_TARGETS = {
    "Zero RB":         {"RB":5, "WR":7, "TE":1, "QB":1, "K":1, "DEF":1},
    "Modified Zero RB":{"RB":4, "WR":7, "TE":1, "QB":1, "K":1, "DEF":1},
    "Hero RB":         {"RB":5, "WR":6, "TE":1, "QB":1, "K":1, "DEF":1},
    "Robust RB":       {"RB":6, "WR":5, "TE":1, "QB":1, "K":1, "DEF":1},
    "Hyper-Fragile RB":{"RB":4, "WR":7, "TE":1, "QB":1, "K":1, "DEF":1},
    "WR-Heavy":        {"RB":4, "WR":8, "TE":1, "QB":1, "K":1, "DEF":1},
    "Pocket QB":       {"RB":5, "WR":6, "TE":1, "QB":1, "K":1, "DEF":1},
    "Bimodal RB":      {"RB":5, "WR":6, "TE":1, "QB":1, "K":1, "DEF":1},
    "Balanced":        {"RB":5, "WR":6, "TE":1, "QB":1, "K":1, "DEF":1},
}

def _make_base_plan(name: str, rnd: int, total_rounds: int, my_counts: Dict[str,int]) -> Dict[int, str]:
    end_round = min(total_rounds, rnd + 5)
    plan: Dict[int, str] = {}
    for i in range(rnd, end_round + 1):
        if name == "Zero RB":
            plan[i] = "WR/TE" if i <= rnd+3 else ("RB upside" if i <= rnd+5 else "QB")
        elif name == "Modified Zero RB":
            plan[i] = "WR/TE" if i <= rnd+2 else ("RB upside" if i <= rnd+5 else "QB")
        elif name == "Hero RB":
            plan[i] = "RB" if i == rnd and my_counts.get("RB",0) < 1 else ("WR/TE" if i <= rnd+3 else "QB")
        elif name == "Robust RB":
            plan[i] = "RB" if (i in (rnd, rnd+1) and my_counts.get("RB",0) < 2) else ("WR/TE" if i <= rnd+4 else "QB")
        elif name == "Hyper-Fragile RB":
            if my_counts.get("RB",0) < 2 and i <= rnd+2: plan[i] = "RB"
            else: plan[i] = "WR/TE"
        elif name == "WR-Heavy":
            plan[i] = "WR" if i <= rnd+2 else ("TE/RB" if i <= rnd+4 else "QB")
        elif name == "Pocket QB":
            plan[i] = "WR/TE/RB" if i <= rnd+6 else "QB"
        elif name == "Bimodal RB":
            plan[i] = "WR/TE" if i <= rnd+2 else ("RB" if i in (rnd+3, rnd+4) else "WR/TE")
        else:
            plan[i] = "Best Value (RB/WR/TE)" if i <= rnd+3 else "QB"
    return plan

def _reserve_last_rounds_for_k_def(
    plan: Dict[int,str],
    total_rounds: int,
    my_counts: Dict[str,int],
    targets: Dict[str,int]
) -> Dict[int,str]:
    need_def = max(0, targets.get("DEF",1) - my_counts.get("DEF",0))
    need_k   = max(0, targets.get("K",1)   - my_counts.get("K",0))
    if total_rounds < 1:
        return plan.copy()
    new_plan = plan.copy()
    last = total_rounds
    second_last = total_rounds - 1
    if need_def > 0 and second_last >= 1:
        new_plan[second_last] = "DEF"
    if need_k > 0 and last >= 1:
        new_plan[last] = "K"
    return new_plan

def score_strategies(
    avail_df: pd.DataFrame,
    my_counts: Dict[str,int],
    starters: Dict[str,int],
    rnd: int,
    teams: int,
    seat: int,
    picks_until_next: int,
    recent_run_counts: Dict[str,int],
    total_rounds: int,
) -> List[Dict[str,object]]:
    if avail_df is None or avail_df.empty:
        return [{"name":"Balanced","score":0.0,"why":"Empty board","plan":{}}]

    depth = {}; edge = {}
    for p in ["RB","WR","TE","QB"]:
        d, e = _tier_depth(avail_df, p); depth[p]=d; edge[p]=e

    wr_edge = edge.get("WR",0.0); te_edge = edge.get("TE",0.0); rb_edge = edge.get("RB",0.0); qb_edge = edge.get("QB",0.0)
    wr_depth = depth.get("WR",0); te_depth = depth.get("TE",0); rb_depth = depth.get("RB",0)

    run_boost_wr = recent_run_counts.get("WR",0)
    run_boost_rb = recent_run_counts.get("RB",0)

    early_turn = seat in (1, teams)
    long_wrap = picks_until_next >= teams // 2

    results = []
    for name in STRATS:
        score = 0.0
        why_bits = []

        if name == "Zero RB":
            score += max(0.0, (max(wr_edge, te_edge) - rb_edge)) * 0.9
            score += (wr_depth + te_depth) * 0.6
            if rnd <= 3: score += 10
            if run_boost_wr >= 3: score -= 6
            if my_counts.get("RB",0) >= 1: score -= 5
            why_bits.append("Pass-catchers show early VBD edge; delay RB.")
        elif name == "Modified Zero RB":
            score += max(0.0, (max(wr_edge, te_edge) - rb_edge)) * 0.7
            score += (wr_depth + te_depth) * 0.5
            score += max(0, rb_depth - 3) * 0.8
            if rnd <= 3: score += 7
            if my_counts.get("RB",0) == 0 and long_wrap: score += 4
            why_bits.append("Pass-catchers now; mid-round RB pocket later.")
        elif name == "Hero RB":
            score += max(0.0, (rb_edge - max(wr_edge, te_edge))) * 1.1
            if my_counts.get("RB",0) == 0 and rnd <= 2: score += 10
            if run_boost_rb >= 2: score -= 4
            why_bits.append("One elite RB projects a large VBD lead.")
        elif name == "Robust RB":
            score += max(0.0, (rb_edge - max(wr_edge, te_edge))) * 0.9
            if early_turn or long_wrap: score += 6
            if my_counts.get("RB",0) <= 1 and rnd <= 3: score += 7
            if run_boost_rb >= 3: score -= 6
            why_bits.append("Multiple bell-cow RBs at good prices.")
        elif name == "Hyper-Fragile RB":
            if my_counts.get("RB",0) >= 2 and rnd <= 6: score += 12
            if my_counts.get("RB",0) == 3: score += 6
            if rb_edge <= max(wr_edge, te_edge): score += 5
            why_bits.append("After 2–3 RBs, shift to WR/TE for depth.")
        elif name == "WR-Heavy":
            score += max(0.0, (wr_edge - max(rb_edge, te_edge))) * 1.0
            score += wr_depth * 0.7
            if rnd <= 3: score += 8
            if run_boost_wr >= 3: score -= 6
            why_bits.append("WR tiers deeper and stronger than RB.")
        elif name == "Pocket QB":
            if rnd <= 5 and qb_edge < max(wr_edge, rb_edge, te_edge) + 5:
                score += 10
            score += max(0.0, qb_edge - (max(wr_edge, rb_edge, te_edge) - 5)) * 0.4
            why_bits.append("Pass QB until the pocket beats other positions.")
        elif name == "Bimodal RB":
            score += max(0, rb_depth - 4) * 1.0
            if rnd in (4,5,6): score += 6
            why_bits.append("Two mid-round RBs can outscore early+late combo.")
        else:
            score += (wr_edge + rb_edge + te_edge) * 0.2
            why_bits.append("Board fairly even; take best value with needs.")
            score -= 6.0

        base = _make_base_plan(name, rnd, total_rounds, my_counts)
        plan = _reserve_last_rounds_for_k_def(base, total_rounds, my_counts, STRAT_TARGETS[name])
        results.append({"name": name, "score": float(score), "why": "; ".join(why_bits), "plan": plan})

    results.sort(key=lambda x: (x["score"], x["name"] == "Balanced"), reverse=True)
    return results

def render_strategy_panel(current: Dict[str,object], targets: Dict[str,int], total_rounds: int):
    with st.container(border=True):
        st.markdown(f"**Current Strategy (dynamic):** {current['name']}")
        st.caption(current["why"])
        plan = current.get("plan") or {}
        if plan:
            rounds = sorted([r for r in plan.keys() if r <= total_rounds])
            if rounds:
                head = rounds[:6]
                dfp = pd.DataFrame([{"Round": r, "Ideal pick": plan[r]} for r in head])
                st.table(dfp)
        if targets:
            st.caption(
                f"Roster targets — RB {targets.get('RB','?')}, WR {targets.get('WR','?')}, "
                f"TE {targets.get('TE','?')}, QB {targets.get('QB','?')}, "
                f"DEF {targets.get('DEF','?')}, K {targets.get('K','?')}."
            )

def render_strategy_health(my_counts: Dict[str,int], targets: Dict[str,int], plan: Dict[int,str], total_rounds: int):
    owned = {p: int(my_counts.get(p, 0)) for p in ["QB","RB","WR","TE","DEF","K"]}
    tgt   = {p: int(targets.get(p, 0))   for p in ["QB","RB","WR","TE","DEF","K"]}
    remain= {p: max(0, tgt[p] - owned[p]) for p in owned}
    df = pd.DataFrame([
        {"POS": p, "Owned": owned[p], "Target": tgt[p], "Remaining": remain[p]}
        for p in ["QB","RB","WR","TE","DEF","K"]
    ])
    with st.container(border=True):
        st.markdown("**Strategy Health**")
        st.table(df)
        # sanity checks
        msgs = []
        if remain["DEF"] > 0 and (plan.get(total_rounds-1) != "DEF" and plan.get(total_rounds) != "DEF"):
            msgs.append("DEF not yet reserved in last rounds—will force in suggestions.")
        if remain["K"] > 0 and (plan.get(total_rounds) != "K"):
            msgs.append("K not yet reserved in last rounds—will force in suggestions.")
        for m in msgs:
            st.warning(m)

# =========================
# Live tab (Suggested Picks on top; no Player Board)
# =========================

def live_tab(csv_df, weights, league_id, username, seat_override, poll_secs, auto_live, include_k_def_anytime):
    st.subheader("Live Draft (Sleeper)")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🔄 Pull latest from Sleeper (Live)"):
            st.rerun()
    with c2:
        if auto_live:
            st.markdown(f"<meta http-equiv='refresh' content='{int(poll_secs)}'>", unsafe_allow_html=True)
            st.caption(f"Auto-refresh every {poll_secs}s.")

    if not league_id:
        st.info("Enter your Sleeper League ID in the sidebar.")
        return

    try:
        league = sleeper.get_league_info(league_id)
        users = sleeper.get_users(league_id) or []
        rosters = sleeper.get_rosters(league_id) or []
        drafts = sleeper.get_drafts_for_league(league_id) or []
    except Exception as e:
        st.error(f"Failed to load league/draft metadata: {e}")
        return

    teams = int(league.get("total_rosters", 12) or 12)
    roster_positions = league.get("roster_positions") or ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF"]
    starters = starters_from_roster_positions(roster_positions)
    rounds_total = int(league.get("settings", {}).get("rounds", 15) or 15)

    draft_id = ""
    for d in drafts:
        if (d.get("status") or "") in ("pre_draft", "in_progress"):
            draft_id = d.get("draft_id", "")
            break
    if not draft_id and drafts:
        draft_id = drafts[0].get("draft_id", "")

    if not draft_id:
        st.warning("No draft found for this league yet.")
        return

    league_name = league.get("name") or league_id
    st.write(f"**League:** {league_name}  |  **Draft ID:** `{draft_id}`")

    try:
        raw_picks = sleeper.get_picks(draft_id) or []
        players_map = sleeper_players_cache()
        pick_log = sleeper.picks_to_internal_log(raw_picks, players_map, teams) or []
        draft_meta = sleeper.get_draft(draft_id) or {}
    except Exception as e:
        st.error(f"Failed to load picks/players: {e}")
        return

    total_picks = len(pick_log)
    next_overall = total_picks + 1
    rnd, pick_in_rnd, slot_on_clock = snake_position(next_overall, teams)
    team_display = slot_to_display_name(slot_on_clock, users, rosters)

    # >>> Robust slot resolution <<<
    my_slot = _detect_my_slot(users, draft_meta, pick_log, seat_override, username)
    you_on_clock = (slot_on_clock == my_slot)

    st.markdown(
        f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — **{team_display}** on the clock."
        + (" 🎯 _(That’s you)_" if you_on_clock else "")
    )
    # NEW: small caption to confirm which slot source is used
    manual_used = (seat_override is not None and int(seat_override) > 0)
    st.caption(f"Using draft slot: {my_slot} ({'manual' if manual_used else 'auto-detected'})")

    if csv_df is None or csv_df.empty:
        st.warning("Upload/load your player file in the sidebar.")
        return

    # Availability (remove drafted by name)
    picked_names = sleeper.picked_player_names(raw_picks, players_map)
    taken_keys = [norm_name(n) for n in picked_names]
    picks_until_next = compute_next_pick_window(teams, my_slot, next_overall)
    avail_df, _ = evaluate_players(
        csv_df, SCORING_DEFAULT, teams, roster_positions, weights,
        current_picks=taken_keys, next_pick_window=picks_until_next
    )

    # Owned counts from normalized log — keyed by my (resolved) slot
    team_counts = _team_pos_counts_from_log(pick_log, teams)
    my_counts = team_counts.get(my_slot, {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0})

    # >>> NEW FALLBACK: if zeros but there are picks, use raw counter with my manual slot <<<
    if sum(my_counts.values()) == 0 and len(raw_picks) > 0 and my_slot > 0:
        my_counts = _count_owned_for_slot_raw(raw_picks, players_map, my_slot)

    my_next_overall = next_overall + picks_until_next
    between_slots = _slots_between(next_overall, my_next_overall, teams)
    needs = _needs_by_slot(team_counts, starters)
    demand_ratio = {
        p: (sum(1 for s in between_slots if needs.get(s, {}).get(p, 0) > 0) / max(1, len(between_slots)))
        for p in ["QB","RB","WR","TE","K","DEF"]
    }
    runs = _recent_runs(pick_log, window=8)

    # -------- Dynamic strategy (recomputed each pick) --------
    strat_ranked = score_strategies(avail_df, my_counts, starters, rnd, teams, my_slot, picks_until_next, runs, rounds_total)
    current = strat_ranked[0]
    targets = STRAT_TARGETS.get(current["name"], STRAT_TARGETS["Balanced"])

    # ===== Suggested Picks FIRST (top of screen) =====
    base_need = {"QB":0,"RB":0,"WR":0,"TE":0}
    for pos in base_need:
        want = max(1, targets.get(pos, 0))
        base_need[pos] = max(0, want - my_counts.get(pos, 0))

    sugg = suggest(avail_df, base_need, weights, topk=8)

    # QB cap
    qb_have = int(my_counts.get("QB", 0))
    sugg = _apply_qb_cap(sugg, qb_have, QB_ROSTER_CAP)

    # Ensure K/DEF appear as needed (esp. last two rounds)
    need_k = max(0, targets.get("K",1) - my_counts.get("K",0))
    need_def = max(0, targets.get("DEF",1) - my_counts.get("DEF",0))
    sugg = _ensure_k_def_in_suggestions(
        sugg, avail_df, rnd, rounds_total, include_k_def_anytime, need_k, need_def
    )

    # Pivot for this pick if run/tier cliff erases edge
    if not sugg.empty:
        top_pos = sugg.iloc[0]["POS"]
        depth_top, edge_top = _tier_depth(avail_df, top_pos)
        alt_best = avail_df[avail_df["POS"] != top_pos].sort_values(["VBD","EVAL_PTS"], ascending=False).head(1)
        if not alt_best.empty:
            alt_edge = float(alt_best.iloc[0].get("VBD", 0.0))
            if depth_top <= 1 or (edge_top - alt_edge) <= 5:
                st.info("Run/tier cliff detected — pivoting to best overall VBD for this pick.")
                sugg = sugg.sort_values(["VBD","EVAL_PTS"], ascending=False).head(8)

    # Display Suggested Picks (top)
    st.markdown("### Suggested Picks (Top 8)")
    disp_rows = []
    for _, row in sugg.iterrows():
        pos = row["POS"]
        need_for_pos = max(0, targets.get(pos, starters.get(pos, 0)) - my_counts.get(pos, 0))
        prob_back = _make_it_back_probability(
            row,
            picks_until_next=picks_until_next,
            demand_ratio=float(demand_ratio.get(pos, 0.0)),
            current_overall=next_overall
        )
        reason = _reason_plain_english(row, need_for_pos, prob_back, picks_until_next)
        disp_rows.append({
            "PLAYER": row["PLAYER"],
            "POS": pos,
            "TEAM": row.get("TEAM"),
            "TIER": row.get("TIER"),
            "ADP": row.get("ADP"),
            "EVAL_PTS": f"{row.get('EVAL_PTS',0):.1f}",
            "VBD": f"{row.get('VBD',0):.1f}",
            "Make it back": f"{prob_back*100:.0f}%",
            "Why this pick": reason
        })
    disp_df = pd.DataFrame(disp_rows)
    if disp_df.empty:
        st.info("No candidates available.")
    else:
        st.dataframe(disp_df, use_container_width=True, height=420)

    # Strategy panel (below suggestions)
    render_strategy_panel(current, targets, rounds_total)
    render_strategy_health(my_counts, targets, current.get("plan", {}), rounds_total)

    # Debug
    with st.expander("Debug (Live)"):
        st.caption(f"Picks fetched (normalized): {len(pick_log)}")
        st.caption(f"Resolved my_slot: {my_slot} | On clock slot: {slot_on_clock}")
        st.caption(f"Picks until next: {picks_until_next} | Recent runs: {runs}")
        st.caption(f"Owned counts (my team): {my_counts}")

# =========================
# Mock tab (kept board here)
# =========================

def mock_tab(csv_df, weights, include_k_def_anytime):
    st.subheader("Mock Draft (Practice)")
    url_or_id = st.text_input(
        "Sleeper Mock URL or draft_id",
        value="",
        help="Paste a URL like https://sleeper.com/draft/nfl/123... or just the 123... id."
    )
    c1, c2, c3 = st.columns([1,1,1])
    reload_btn = c1.button("Load / Re-sync Mock")
    clear_btn = c2.button("Reset Practice")
    force_btn = c3.button("🔄 Pull latest picks (Mock)")

    qb_have_practice = st.number_input("QBs drafted (practice)", min_value=0, max_value=QB_ROSTER_CAP, value=0, step=1)

    if clear_btn:
        st.session_state.pop("mock_state", None)
        st.session_state.pop("prev_strategy", None)
        st.success("Practice state cleared.")

    if csv_df is None or csv_df.empty:
        st.info("Upload/load your player file in the sidebar.")
        return

    if reload_btn and url_or_id.strip():
        draft_id = sleeper.parse_draft_id_from_url(url_or_id.strip())
        st.caption(f"Parsed draft_id: `{draft_id or 'None'}`")
        if not draft_id:
            st.error("Could not parse a draft_id from your input.")
        else:
            try:
                dmeta = sleeper.get_draft(draft_id) or {}
                teams = int(dmeta.get("settings", {}).get("teams", dmeta.get("teams", 12)) or 12)
                rounds = int(dmeta.get("settings", {}).get("rounds", 15) or 15)
                picks = sleeper.get_picks(draft_id) or []
                players_map = sleeper_players_cache()
                picked_names = sleeper.picked_player_names(picks, players_map)
                taken_keys = [norm_name(n) for n in picked_names]
                roster_positions = ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF"]
                starters = starters_from_roster_positions(roster_positions)
                avail_df, _ = evaluate_players(
                    csv_df, SCORING_DEFAULT, teams, roster_positions, weights, current_picks=taken_keys
                )
                st.session_state.mock_state = {
                    "draft_id": draft_id,
                    "teams": teams,
                    "rounds": rounds,
                    "picks": picks,
                    "available": avail_df.reset_index(drop=True),
                    "starters": starters,
                }
                st.success(f"Mock {draft_id} loaded — {len(picks)} picks synced, teams={teams}, rounds={rounds}.")
            except Exception as e:
                st.error(f"Mock load failed: {e}")

    if "mock_state" not in st.session_state:
        st.info("Load a mock to begin.")
        return

    S = st.session_state.mock_state
    teams = int(S["teams"]); rounds = int(S["rounds"])
    picks = S["picks"]; starters = S["starters"]

    next_overall = len(picks) + 1
    rnd, pick_in_rnd, slot_on_clock = snake_position(next_overall, teams)
    st.write(f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — Slot {slot_on_clock}")

    if force_btn:
        try:
            new_picks = sleeper.get_picks(S["draft_id"]) or []
            if len(new_picks) != len(picks):
                S["picks"] = new_picks
                picks = new_picks
                players_map = sleeper_players_cache()
                picked_names = sleeper.picked_player_names(picks, players_map)
                taken_keys = [norm_name(n) for n in picked_names]
                roster_positions = ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF"]
                avail_df, _ = evaluate_players(
                    csv_df, SCORING_DEFAULT, teams, roster_positions, weights, current_picks=taken_keys
                )
                S["available"] = avail_df.reset_index(drop=True)
                st.session_state.mock_state = S
            st.rerun()
        except Exception as e:
            st.error(f"Refresh picks failed: {e}")

    players_map = sleeper_players_cache()
    picked_names = sleeper.picked_player_names(picks, players_map)
    taken_keys = [norm_name(n) for n in picked_names]
    roster_positions = ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF"]
    avail_df, _ = evaluate_players(
        csv_df, SCORING_DEFAULT, teams, roster_positions, weights, current_picks=taken_keys, next_pick_window=teams
    )
    S["available"] = avail_df.reset_index(drop=True)
    st.session_state.mock_state = S

    # Strategy (dynamic) for mock too
    my_counts = {"QB":qb_have_practice,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0}
    runs = _recent_runs(sleeper.picks_to_internal_log(picks, players_map, teams), window=8)
    strat_ranked = score_strategies(avail_df, my_counts, starters, rnd, teams, slot_on_clock, teams, runs, rounds)
    current = strat_ranked[0]
    targets = STRAT_TARGETS.get(current["name"], STRAT_TARGETS["Balanced"])

    # Suggestions
    base_need = {"QB":1,"RB":2,"WR":2,"TE":1}
    for pos in base_need:
        base_need[pos] = max(1, targets.get(pos, base_need[pos]))
    sugg = suggest(S["available"], base_need, weights, topk=8)
    sugg = _apply_qb_cap(sugg, qb_have_practice, QB_ROSTER_CAP)

    need_k = max(0, targets.get("K",1) - my_counts.get("K",0))
    need_def = max(0, targets.get("DEF",1) - my_counts.get("DEF",0))
    sugg = _ensure_k_def_in_suggestions(
        sugg, S["available"], rnd, rounds, include_k_def_anytime, need_k, need_def
    )

    # Display
    st.markdown("### Suggested Picks (Top 8)")
    disp_rows = []
    for _, row in sugg.iterrows():
        pos = row["POS"]
        prob_back = _make_it_back_probability(
            row, picks_until_next=teams, demand_ratio=0.5, current_overall=next_overall
        )
        reason = _reason_plain_english(row, need_for_pos=1 if pos in ("QB","TE","K","DEF") else 2, prob_back=prob_back, next_picks=teams)
        disp_rows.append({
            "PLAYER": row["PLAYER"],
            "POS": pos,
            "TEAM": row.get("TEAM"),
            "TIER": row.get("TIER"),
            "ADP": row.get("ADP"),
            "EVAL_PTS": f"{row.get('EVAL_PTS',0):.1f}",
            "VBD": f"{row.get('VBD',0):.1f}",
            "Make it back": f"{prob_back*100:.0f}%",
            "Why this pick": reason
        })
    st.dataframe(pd.DataFrame(disp_rows), use_container_width=True, height=420)

    # Strategy panel/health (mock)
    render_strategy_panel(current, targets, rounds)
    render_strategy_health(my_counts, targets, current.get("plan", {}), rounds)

    # Keep board in mock tab
    st.markdown("### Player Board (Available)")
    show_cols = ["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD","INJURY_RISK","SOS_SEASON"]
    st.dataframe(S["available"][show_cols].sort_values(["VBD","EVAL_PTS"], ascending=False), use_container_width=True)

# =========================
# Board tab
# =========================

def board_tab(csv_df, weights):
    st.subheader("Player Board & Filters")
    if csv_df is None or csv_df.empty:
        st.info("Upload/load your player file in the sidebar.")
        return

    teams = st.number_input("Teams", 8, 16, 12, 1)
    roster_positions = st.text_input("Roster positions (comma)", value="QB,RB,RB,WR,WR,TE,FLEX,K,DEF")
    roster_positions = [p.strip().upper() for p in roster_positions.split(",") if p.strip()]

    avail_df, _ = evaluate_players(
        csv_df, SCORING_DEFAULT, int(teams), roster_positions, weights, current_picks=[]
    )

    pos = st.multiselect("Position", ["QB","RB","WR","TE","K","DEF"], default=["RB","WR","TE","QB","K","DEF"])
    team = st.text_input("Team filter (e.g., KC, SF)")
    tier_max = st.number_input("Max Tier (optional)", 0, 20, 20, 1)

    filt = avail_df[avail_df["POS"].isin(pos)].copy()
    if team:
        filt = filt[filt["TEAM"].fillna("").str.upper().str.contains(team.strip().upper(), na=False)]
    if tier_max < 20:
        filt = filt[(filt["TIER"].isna()) | (filt["TIER"] <= tier_max)]

    st.dataframe(filt.sort_values(["VBD","EVAL_PTS"], ascending=False), use_container_width=True)
    csv = filt.to_csv(index=False).encode()
    st.download_button("Download filtered board CSV", csv, "filtered_board.csv", "text/csv")

# =========================
# Main
# =========================

def main():
    st.title("Fantasy Football Draft Assistant — Dynamic Strategy, VBD, & Pivots (K/DEF aware)")
    csv_df, weights, league_id, username, seat, poll_secs, auto_live, include_k_def_anytime = sidebar_controls()

    tabs = st.tabs(["Live Draft", "Mock Draft", "Player Board"])
    with tabs[0]:
        live_tab(csv_df, weights, league_id, username, seat, poll_secs, auto_live, include_k_def_anytime)
    with tabs[1]:
        mock_tab(csv_df, weights, include_k_def_anytime)
    with tabs[2]:
        board_tab(csv_df, weights)

if __name__ == "__main__":
    main()

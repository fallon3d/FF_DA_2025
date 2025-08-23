import os
import sys
import math
from typing import Dict, List, Tuple
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
    user_roster_id,          # users + rosters + username -> roster_id (slot)
    slot_to_display_name,
)

st.set_page_config(page_title="FF Draft Assistant — VBD + Smart Strategy", layout="wide")

DEFAULT_CSV_PATH = os.environ.get("FFDA_CSV_PATH", "")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# === Behavior toggles / caps
QB_ROSTER_CAP = int(os.environ.get("FFDA_QB_CAP", "2"))
INCLUDE_K_DEF_EARLY = bool(int(os.environ.get("FFDA_INCLUDE_K_DEF_EARLY", "0")))  # 0 = mostly late, 1 = anytime if value

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
    username = st.sidebar.text_input("Your Sleeper username (optional)", value="Fallon3D")
    seat = st.sidebar.number_input("Your draft slot (1–Teams; 0=auto)", min_value=0, max_value=20, value=0)

    poll_secs = st.sidebar.slider("Auto-refresh seconds", 3, 30, 5, 1)
    auto_live = st.sidebar.toggle("Auto-refresh (Live tab)", value=False)

    st.sidebar.header("K/DEF")
    include_k_def_anytime = st.sidebar.checkbox(
        "Allow K & DEF to appear anytime if value is high",
        value=INCLUDE_K_DEF_EARLY
    )

    if st.sidebar.button("Reset strategy lock"):
        for k in ["strategy_name","strategy_why","strategy_plan","strategy_targets","strategy_locked_at_overall"]:
            st.session_state.pop(k, None)
        st.sidebar.success("Strategy will be chosen at your next first pick.")

    return csv_df, weights, league_id, username, int(seat), poll_secs, auto_live, include_k_def_anytime

# =========================
# Helpers: VONA window, reasons, demand, runs/pivots
# =========================

def compute_next_pick_window(teams: int, seat: int, current_overall_pick: int) -> int:
    """Snake-draft distance (in selections) from the current overall pick to your next pick."""
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

def _team_pos_counts_from_picks(picks: List[dict], teams: int) -> Dict[int, Dict[str,int]]:
    counts = {slot: {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0} for slot in range(1, teams+1)}
    for p in picks or []:
        try:
            slot = int(p.get("slot") or p.get("roster_id") or 0)
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
            need = max(0, base - cmap.get(pos, 0))
            needs[slot][pos] = int(need)
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
    # ADP vs distance, scaled by demand among teams picking before you
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

# =========================
# Strategy selection + pivot logic
# =========================

_STRAT_TARGETS = {
    "Zero RB":        {"RB":5, "WR":7, "TE":1, "QB":1},
    "Modified Zero RB":{"RB":4, "WR":7, "TE":1, "QB":1},
    "Hero RB":        {"RB":5, "WR":6, "TE":1, "QB":1},
    "Robust RB":      {"RB":6, "WR":5, "TE":1, "QB":1},
    "Hyper-Fragile RB":{"RB":4, "WR":7, "TE":1, "QB":1},
    "WR-Heavy":       {"RB":4, "WR":8, "TE":1, "QB":1},
    "Pocket QB":      {"RB":5, "WR":6, "TE":1, "QB":1},
    "Balanced":       {"RB":5, "WR":6, "TE":1, "QB":1},
}

def _choose_strategy(avail_df: pd.DataFrame, rnd: int, seat: int, teams: int, picks_until_next: int) -> Tuple[str, str, Dict[int, str]]:
    """
    Choose the strategy AT YOUR FIRST PICK ONLY based on VBD edges and table position.
    """
    if avail_df is None or avail_df.empty:
        return "Balanced", "Board is flat; default to balanced value picks.", {}

    # top VBD per position
    top_rb = avail_df[avail_df["POS"]=="RB"].nlargest(3, "VBD")
    top_wr = avail_df[avail_df["POS"]=="WR"].nlargest(3, "VBD")
    top_te = avail_df[avail_df["POS"]=="TE"].nlargest(2, "VBD")
    top_qb = avail_df[avail_df["POS"]=="QB"].nlargest(2, "VBD")

    v_rb1 = float(top_rb.iloc[0]["VBD"]) if len(top_rb)>0 else 0.0
    v_rb2 = float(top_rb.iloc[1]["VBD"]) if len(top_rb)>1 else 0.0
    v_wr1 = float(top_wr.iloc[0]["VBD"]) if len(top_wr)>0 else 0.0
    v_wr2 = float(top_wr.iloc[1]["VBD"]) if len(top_wr)>1 else 0.0
    v_te1 = float(top_te.iloc[0]["VBD"]) if len(top_te)>0 else 0.0
    v_qb1 = float(top_qb.iloc[0]["VBD"]) if len(top_qb)>0 else 0.0

    BIG = 25.0
    MED = 15.0

    early_turn = seat in (1, teams)
    long_wrap = picks_until_next >= teams // 2

    if v_rb1 - max(v_wr1, v_te1) >= BIG:
        strat = "Hero RB"
        why = "One RB has a clear VBD edge at your slot; anchor RB now, then hammer WR/TE."
    elif (v_rb1 + v_rb2) - (v_wr1 + v_wr2) >= MED and (early_turn or long_wrap):
        strat = "Robust RB"
        why = "Two RBs project as weekly touch leaders; double up early before the room can respond."
    elif max(v_wr1, v_te1) - v_rb1 >= MED:
        strat = "Modified Zero RB"
        why = "Pass-catchers hold the wider early value gap; we’ll harvest RB chaos later."
    elif (v_wr1 + v_wr2) - (v_rb1 + v_rb2) >= MED:
        strat = "WR-Heavy"
        why = "WR tiers are wide vs. RB; locking target share early boosts weekly ceiling."
    else:
        strat = "Balanced"
        why = "No extreme edge at your slot; take the best value while meeting roster needs."

    # Ideal positions next ~6 rounds (outline)
    plan: Dict[int,str] = {}
    for i in range(rnd, rnd+6):
        if strat == "Hero RB":
            plan[i] = "RB" if i == rnd else ("WR/TE" if i <= rnd+3 else "QB/K/DEF")
        elif strat == "Robust RB":
            plan[i] = "RB" if i in (rnd, rnd+1) else ("WR/TE" if i <= rnd+4 else "QB/K/DEF")
        elif strat == "Modified Zero RB":
            plan[i] = "WR/TE" if i <= rnd+3 else ("RB upside" if i <= rnd+5 else "QB/K/DEF")
        elif strat == "WR-Heavy":
            plan[i] = "WR" if i <= rnd+2 else ("TE/RB" if i <= rnd+4 else "QB/K/DEF")
        else:  # Balanced
            plan[i] = "Best Value (RB/WR/TE)" if i <= rnd+3 else "QB/K/DEF"

    return strat, why, plan

def _pivot_check(avail_df: pd.DataFrame, target_labels: List[str]) -> Tuple[bool, str]:
    """
    Pivot if the target position's TIER is collapsing or VBD edge is erased.
    - If top candidate's tier for any target label has <=1 remaining, pivot.
    - If top candidate VBD for target is within 5 of alternative positions while alternative tier is 3+ deep, pivot.
    """
    if avail_df is None or avail_df.empty or not target_labels:
        return False, ""
    # Build per-pos top + tier depth
    msg_bits = []
    pivot = False
    for pos in target_labels:
        cand = avail_df[avail_df["POS"]==pos].sort_values(["VBD","EVAL_PTS"], ascending=False).head(1)
        if cand.empty:
            continue
        t = cand.iloc[0].get("TIER")
        v = float(cand.iloc[0].get("VBD", 0.0))
        tier_size = int(avail_df[(avail_df["POS"]==pos) & (avail_df["TIER"]==t)].shape[0]) if not pd.isna(t) else 9
        if tier_size <= 1:
            pivot = True
            msg_bits.append(f"{pos} tier about to vanish")
        # compare with best alternative
        alt = avail_df[~(avail_df["POS"].isin(target_labels))].sort_values(["VBD","EVAL_PTS"], ascending=False).head(1)
        if not alt.empty:
            v_alt = float(alt.iloc[0].get("VBD", 0.0))
            alt_pos = str(alt.iloc[0].get("POS"))
            alt_tier = alt.iloc[0].get("TIER")
            alt_size = int(avail_df[(avail_df["POS"]==alt_pos) & (avail_df["TIER"]==alt_tier)].shape[0]) if not pd.isna(alt_tier) else 9
            if (v_alt - v) >= -5 and alt_size >= 3:
                pivot = True
                msg_bits.append(f"{pos} VBD edge erased vs {alt_pos}")
    return pivot, "; ".join(msg_bits)

def _ensure_k_def_in_suggestions(sugg_df: pd.DataFrame, avail_df: pd.DataFrame, rnd: int, total_rounds: int, include_anytime: bool) -> pd.DataFrame:
    if sugg_df is None or sugg_df.empty: return sugg_df
    want_force = (rnd >= total_rounds-1)
    have_k = (sugg_df["POS"]=="K").any()
    have_d = (sugg_df["POS"]=="DEF").any()
    if have_k and have_d:
        return sugg_df
    top_k = avail_df[avail_df["POS"]=="K"].sort_values(["VBD","EVAL_PTS"], ascending=False).head(1)
    top_d = avail_df[avail_df["POS"]=="DEF"].sort_values(["VBD","EVAL_PTS"], ascending=False).head(1)
    base = sugg_df.copy()
    tail_vbd = float(base["VBD"].iloc[min(len(base)-1, 7)]) if "VBD" in base.columns and not base.empty else 0.0
    candidates = []
    if not top_k.empty and (want_force or (include_anytime and float(top_k.iloc[0]["VBD"]) >= tail_vbd - 15)):
        candidates.append(top_k.iloc[0])
    if not top_d.empty and (want_force or (include_anytime and float(top_d.iloc[0]["VBD"]) >= tail_vbd - 15)):
        candidates.append(top_d.iloc[0])
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
# LIVE TAB
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

    # League + draft metadata
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

    # Picks + players
    try:
        picks = sleeper.get_picks(draft_id) or []
        players_map = sleeper_players_cache()
    except Exception as e:
        st.error(f"Failed to load picks/players: {e}")
        return

    total_picks = len(picks)
    next_overall = total_picks + 1
    rnd, pick_in_rnd, slot_on_clock = snake_position(next_overall, teams)
    team_display = slot_to_display_name(slot_on_clock, users, rosters)

    # Your slot
    auto_slot = user_roster_id(users, rosters, username) or 0
    my_slot = int(seat_override) if int(seat_override) > 0 else (auto_slot or 1)
    you_on_clock = (slot_on_clock == my_slot)

    st.markdown(
        f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — **{team_display}** on the clock."
        + (" 🎯 _(That’s you)_" if you_on_clock else "")
    )

    if csv_df is None or csv_df.empty:
        st.warning("Upload/load your player file in the sidebar.")
        return

    # Build availability
    picked_names = sleeper.picked_player_names(picks, players_map)
    taken_keys = [norm_name(n) for n in picked_names]
    picks_until_next = compute_next_pick_window(teams, my_slot, next_overall)
    avail_df, _ = evaluate_players(
        csv_df, SCORING_DEFAULT, teams, roster_positions, weights,
        current_picks=taken_keys, next_pick_window=picks_until_next
    )

    # Team needs & demand model
    team_counts = _team_pos_counts_from_picks(picks, teams)
    my_counts = team_counts.get(my_slot, {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0})

    my_next_overall = next_overall + picks_until_next
    between_slots = _slots_between(next_overall, my_next_overall, teams)
    needs = _needs_by_slot(team_counts, starters)
    demand_ratio = {
        p: (sum(1 for s in between_slots if needs.get(s, {}).get(p, 0) > 0) / max(1, len(between_slots)))
        for p in ["QB","RB","WR","TE","K","DEF"]
    }

    # -------------------- Strategy selection (lock on your first pick) --------------------
    if "strategy_name" not in st.session_state:
        st.session_state.strategy_name = None
    if you_on_clock and st.session_state.strategy_name is None:
        # Choose and lock
        strat, why, plan = _choose_strategy(avail_df, rnd, my_slot, teams, picks_until_next)
        st.session_state.strategy_name = strat
        st.session_state.strategy_why = why
        st.session_state.strategy_plan = plan
        st.session_state.strategy_targets = _STRAT_TARGETS.get(strat, _STRAT_TARGETS["Balanced"])
        st.session_state.strategy_locked_at_overall = next_overall

    # if not yet locked, show waiting message
    if st.session_state.strategy_name is None:
        st.info("Waiting for your first pick to choose the optimal draft strategy…")
    else:
        with st.container(border=True):
            st.markdown(f"**Current Strategy:** {st.session_state.strategy_name}")
            st.caption(st.session_state.strategy_why)
            if st.session_state.strategy_plan:
                dfp = pd.DataFrame(
                    [{"Round": r, "Ideal pick": st.session_state.strategy_plan[r]}
                     for r in sorted(st.session_state.strategy_plan.keys())]
                )
                st.table(dfp)
            tgt = st.session_state.strategy_targets or {}
            if tgt:
                st.caption(f"Roster targets — RB {tgt.get('RB','?')}, WR {tgt.get('WR','?')}, TE {tgt.get('TE','?')}, QB {tgt.get('QB','?')}.")

    # -------------------- Build needs baseline from targets --------------------
    # Use strategy targets → convert to "need right now" (starters+bench targets minus owned)
    targets = st.session_state.get("strategy_targets") or _STRAT_TARGETS["Balanced"]
    base_need = {"QB":0,"RB":0,"WR":0,"TE":0}
    for pos in base_need:
        want = max(targets.get(pos, 0), starters.get(pos, 0))
        base_need[pos] = max(0, want - my_counts.get(pos, 0))

    # -------------------- Suggested picks --------------------
    sugg = suggest(avail_df, base_need, weights, topk=8)

    # QB cap
    qb_have = int(my_counts.get("QB", 0))
    sugg = _apply_qb_cap(sugg, qb_have, QB_ROSTER_CAP)

    # Late-round contingent / spike preference (Round >= rounds_total-5): boost RB/WR with higher volatility/usage
    if rnd >= max(1, rounds_total - 5) and not sugg.empty:
        # crude bump: sort by VBD then prefer RB/WR rows with higher usage proxies if present
        order_cols = ["VBD","EVAL_PTS"]
        sugg = sugg.sort_values(order_cols, ascending=False)

    # Ensure K/DEF presence appropriately
    sugg = _ensure_k_def_in_suggestions(sugg, avail_df, rnd, rounds_total, include_k_def_anytime)

    # -------------------- Pivot check for this pick --------------------
    # derive current round target labels from plan/strategy
    target_labels = []
    if st.session_state.get("strategy_plan"):
        label = st.session_state.strategy_plan.get(rnd)
        if label:
            if "WR/TE" in label:
                target_labels = ["WR","TE"]
            elif "TE/RB" in label:
                target_labels = ["TE","RB"]
            elif "RB upside" in label:
                target_labels = ["RB"]
            elif "Best Value" in label:
                target_labels = ["RB","WR","TE"]
            elif "QB/K/DEF" in label:
                target_labels = ["QB","K","DEF"]
            else:
                target_labels = [label] if label in ("RB","WR","TE","QB","K","DEF") else ["RB","WR","TE"]
    else:
        target_labels = ["RB","WR","TE"]

    pivot_now, pivot_msg = _pivot_check(avail_df, target_labels)
    if pivot_now:
        st.warning(f"Pivoting this pick: {pivot_msg}.")
        # Re-sort suggestions purely by VBD/EVAL_PTS (don’t chase the run)
        sugg = sugg.sort_values(["VBD","EVAL_PTS"], ascending=False).head(8)

    # -------------------- Display with make-it-back + reasons --------------------
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

    col1, col2 = st.columns([1.25, 1])
    with col1:
        st.markdown("### Suggested Picks (Top 8)")
        if disp_df.empty:
            st.info("No candidates available.")
        else:
            st.dataframe(disp_df, use_container_width=True, height=420)
    with col2:
        st.markdown("### Player Board (Available)")
        show_cols = ["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD","INJURY_RISK","SOS_SEASON"]
        st.dataframe(
            avail_df[show_cols].sort_values(["VBD","EVAL_PTS"], ascending=False),
            use_container_width=True, height=420
        )

    with st.expander("Debug (Live)"):
        st.caption(f"Picks fetched: {total_picks}")
        st.caption(f"My slot (auto/fallback): {my_slot}")
        st.caption(f"Picks until next: {picks_until_next}; between slots: {len(between_slots)}")
        st.caption(f"Strategy: {st.session_state.get('strategy_name')} @ overall {st.session_state.get('strategy_locked_at_overall')}")
        if picks:
            st.json(picks[0])

# =========================
# MOCK TAB
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

    # Practice-only: your QB count to simulate cap behavior
    qb_have_practice = st.number_input("QBs drafted (practice)", min_value=0, max_value=QB_ROSTER_CAP, value=0, step=1)

    if clear_btn:
        st.session_state.pop("mock_state", None)
        for k in ["strategy_name","strategy_why","strategy_plan","strategy_targets","strategy_locked_at_overall"]:
            st.session_state.pop(k, None)
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

    # Neutral window ~ teams
    players_map = sleeper_players_cache()
    picked_names = sleeper.picked_player_names(picks, players_map)
    taken_keys = [norm_name(n) for n in picked_names]
    roster_positions = ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF"]
    avail_df, _ = evaluate_players(
        csv_df, SCORING_DEFAULT, teams, roster_positions, weights, current_picks=taken_keys, next_pick_window=teams
    )
    S["available"] = avail_df.reset_index(drop=True)
    st.session_state.mock_state = S

    # Strategy lock: pretend this is your first pick when you're on slot_on_clock and no lock set
    if st.session_state.get("strategy_name") is None:
        strat, why, plan = _choose_strategy(avail_df, rnd, slot_on_clock, teams, teams)
        st.session_state.strategy_name = strat
        st.session_state.strategy_why = why
        st.session_state.strategy_plan = plan
        st.session_state.strategy_targets = _STRAT_TARGETS.get(strat, _STRAT_TARGETS["Balanced"])
        st.session_state.strategy_locked_at_overall = next_overall

    with st.container(border=True):
        st.markdown(f"**Current Strategy:** {st.session_state.get('strategy_name')}")
        st.caption(st.session_state.get("strategy_why",""))
        plan = st.session_state.get("strategy_plan") or {}
        if plan:
            dfp = pd.DataFrame([{"Round": r, "Ideal pick": plan[r]} for r in sorted(plan.keys())])
            st.table(dfp)
        tgt = st.session_state.get("strategy_targets") or {}
        if tgt:
            st.caption(f"Roster targets — RB {tgt.get('RB','?')}, WR {tgt.get('WR','?')}, TE {tgt.get('TE','?')}, QB {tgt.get('QB','?')}.")

    # Build needs baseline from targets (mock has no personal roster; simulate “needs”)
    targets = st.session_state.get("strategy_targets") or _STRAT_TARGETS["Balanced"]
    base_need = {"QB":1,"RB":2,"WR":2,"TE":1}
    for pos in base_need:
        base_need[pos] = max(1, targets.get(pos, base_need[pos]))

    sugg = suggest(S["available"], base_need, weights, topk=8)
    # Apply QB cap
    sugg = _apply_qb_cap(sugg, qb_have_practice, QB_ROSTER_CAP)
    # Ensure K/DEF presence
    sugg = _ensure_k_def_in_suggestions(sugg, S["available"], rnd, rounds, include_k_def_anytime)

    # Simple demand ratios in mock (no team rosters here)
    demand_ratio = {p: 0.5 for p in ["QB","RB","WR","TE","K","DEF"]}

    disp_rows = []
    for _, row in sugg.iterrows():
        pos = row["POS"]
        prob_back = _make_it_back_probability(row, picks_until_next=teams, demand_ratio=demand_ratio.get(pos,0.5), current_overall=next_overall)
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
    disp_df = pd.DataFrame(disp_rows)

    col1, col2 = st.columns([1.25, 1])
    with col1:
        st.markdown("### Suggested Picks (Top 8)")
        if disp_df.empty:
            st.info("No candidates available.")
        else:
            st.dataframe(disp_df, use_container_width=True, height=420)
    with col2:
        st.markdown("### Player Board (Available)")
        show_cols = ["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD","INJURY_RISK","SOS_SEASON"]
        st.dataframe(S["available"][show_cols].sort_values(["VBD","EVAL_PTS"], ascending=False), use_container_width=True)

# =========================
# BOARD TAB
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
    st.title("Fantasy Football Draft Assistant — VBD + Smart Strategy & Pivots")
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

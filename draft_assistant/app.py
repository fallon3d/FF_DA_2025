# draft_assistant/app.py
from __future__ import annotations

import os
import sys
from typing import Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st

# Make package importable when running directly
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from draft_assistant.core import sleeper
from draft_assistant.core.evaluation import evaluate_players
from draft_assistant.core.suggestions import suggest
from draft_assistant.core.utils import (
    read_player_table,
    norm_name,
    snake_position,
    starters_from_roster_positions,
    slot_to_display_name,
)

st.set_page_config(page_title="Fantasy Football Draft Assistant — Stable VBD", layout="wide")

# --------------------------
# Tunables
# --------------------------
QB_ROSTER_CAP = 2                        # max QBs to consider for your roster
K_DEF_LAST_ROUNDS_ONLY = True            # show K/DEF only in final rounds
DEFAULT_ROSTER_POSITIONS = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]

# --------------------------
# Sidebar
# --------------------------
def sidebar_controls():
    st.sidebar.header("Data")
    up = st.sidebar.file_uploader("Upload Players (CSV or Excel)", type=["csv", "xlsx", "xls"])
    csv_df = read_player_table(up) if up is not None else None

    st.sidebar.header("Sleeper (Live)")
    league_id = st.sidebar.text_input("League ID", value="")
    username = st.sidebar.text_input("Your Sleeper username (or display name)", value="Fallon3D")
    seat_override = st.sidebar.number_input("Your draft slot (1–Teams; 0 = auto)", min_value=0, max_value=20, value=0)

    teams = st.sidebar.number_input("Teams (default if API missing)", min_value=8, max_value=16, value=12, step=1)
    rounds = st.sidebar.number_input("Rounds (default if API missing)", min_value=12, max_value=25, value=15, step=1)
    poll_secs = st.sidebar.slider("Auto-refresh seconds", 3, 30, 5, 1)
    auto_live = st.sidebar.toggle("Auto-refresh (Live tab)", value=False)

    return csv_df, league_id, username, int(seat_override), int(teams), int(rounds), poll_secs, auto_live

# --------------------------
# Helpers
# --------------------------
def _pos_from_meta(p: dict) -> str:
    return str((p.get("metadata") or {}).get("position") or "").upper().replace("DST", "DEF")

def _team_pos_counts_from_log(pick_log: List[dict], teams: int) -> Dict[int, Dict[str, int]]:
    counts = {slot: {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DEF": 0} for slot in range(1, teams + 1)}
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

def _resolve_user_id(users: List[dict], username_or_display: str) -> Optional[str]:
    target = (username_or_display or "").strip().lower()
    for u in users or []:
        if str(u.get("username", "")).lower() == target:
            return u.get("user_id")
        if str(u.get("display_name", "")).lower() == target:
            return u.get("user_id")
    return None

def _detect_my_slot(users: List[dict], draft_meta: dict, pick_log: List[dict], seat_override: int, username: str) -> int:
    if int(seat_override) > 0:
        return int(seat_override)
    my_uid = _resolve_user_id(users, username)
    draft_order = draft_meta.get("draft_order") if isinstance(draft_meta, dict) else None
    if my_uid and isinstance(draft_order, dict):
        slot = draft_order.get(my_uid)
        if slot:
            try:
                return int(slot)
            except Exception:
                pass
    # Fallback to first slot
    return 1

def compute_next_pick_window(teams: int, seat: int, current_overall_pick: int) -> int:
    """How many selections until your NEXT turn (not counting your current pick)."""
    if not (1 <= seat <= teams):
        return teams
    rnd = (current_overall_pick - 1) // teams + 1
    pos = (current_overall_pick - 1) % teams + 1
    my_pos_this = seat if rnd % 2 == 1 else (teams - seat + 1)
    if my_pos_this > pos:
        return my_pos_this - pos
    if my_pos_this == pos:
        my_pos_next = (teams - seat + 1) if rnd % 2 == 1 else seat
        return my_pos_next
    my_pos_next = (teams - seat + 1) if rnd % 2 == 1 else seat
    return (teams - pos) + my_pos_next

# --------------------------
# Strategy (simple lock on first pick)
# --------------------------
STRATS = [
    "Zero RB", "Modified Zero RB", "Hero RB", "Robust RB",
    "Hyper-Fragile RB", "WR-Heavy", "Pocket QB", "Bimodal RB", "Balanced"
]
STRAT_TARGETS = {
    "Zero RB":         {"RB": 5, "WR": 7, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
    "Modified Zero RB":{"RB": 4, "WR": 7, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
    "Hero RB":         {"RB": 5, "WR": 6, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
    "Robust RB":       {"RB": 6, "WR": 5, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
    "Hyper-Fragile RB":{"RB": 4, "WR": 7, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
    "WR-Heavy":        {"RB": 4, "WR": 8, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
    "Pocket QB":       {"RB": 5, "WR": 6, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
    "Bimodal RB":      {"RB": 5, "WR": 6, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
    "Balanced":        {"RB": 5, "WR": 6, "TE": 1, "QB": 1, "K": 1, "DEF": 1},
}

if "selected_strategy" not in st.session_state:
    st.session_state.selected_strategy = None

def _make_base_plan(name: str, rnd: int, total_rounds: int) -> Dict[int, str]:
    plan: Dict[int, str] = {}
    end_round = min(total_rounds, rnd + 5)
    for i in range(rnd, end_round + 1):
        if name == "Zero RB":
            plan[i] = "WR/TE" if i <= rnd + 3 else "RB upside"
        elif name == "Modified Zero RB":
            plan[i] = "WR/TE" if i <= rnd + 2 else "RB pocket"
        elif name == "Hero RB":
            plan[i] = "RB" if i == rnd else ("WR/TE" if i <= rnd + 3 else "QB")
        elif name == "Robust RB":
            plan[i] = "RB" if i in (rnd, rnd + 1) else "WR/TE"
        elif name == "Hyper-Fragile RB":
            plan[i] = "RB" if i <= rnd + 1 else "WR/TE"
        elif name == "WR-Heavy":
            plan[i] = "WR" if i <= rnd + 2 else "TE/RB"
        elif name == "Pocket QB":
            plan[i] = "WR/TE/RB" if i <= rnd + 5 else "QB"
        elif name == "Bimodal RB":
            plan[i] = "WR/TE" if i <= rnd + 2 else ("RB" if i in (rnd + 3, rnd + 4) else "WR/TE")
        else:
            plan[i] = "Best Value"
    return plan

# --------------------------
# Live tab
# --------------------------
def live_tab(csv_df, league_id, username, seat_override, teams_default, rounds_default, poll_secs, auto_live):
    st.subheader("Live Draft (Sleeper)")
    c1, c2 = st.columns([1, 1])
    if c1.button("Refresh now"):
        st.rerun()
    if auto_live:
        st.markdown(f"<meta http-equiv='refresh' content='{int(poll_secs)}'>", unsafe_allow_html=True)
        c2.caption(f"Auto-refresh every {poll_secs}s.")

    if not league_id:
        st.info("Enter your Sleeper League ID in the sidebar.")
        return
    if csv_df is None or csv_df.empty:
        st.info("Upload your player file in the sidebar.")
        return

    # League & draft meta
    league = sleeper.get_league_info(league_id) or {}
    users = sleeper.get_users(league_id) or []
    drafts = sleeper.get_drafts_for_league(league_id) or []

    teams = int(league.get("total_rosters", teams_default) or teams_default)
    rounds_total = int(league.get("settings", {}).get("rounds", rounds_default) or rounds_default)
    roster_positions = league.get("roster_positions") or DEFAULT_ROSTER_POSITIONS
    starters = starters_from_roster_positions(roster_positions)

    draft_id = ""
    for d in drafts:
        if (d.get("status") or "") in ("pre_draft", "in_progress"):
            draft_id = d.get("draft_id", "")
            break
    if not draft_id and drafts:
        draft_id = drafts[0].get("draft_id", "")

    if not draft_id:
        st.warning("No active draft found for this league.")
        return

    league_name = league.get("name") or league_id
    st.write(f"**League:** {league_name}  |  **Draft ID:** `{draft_id}`")

    # Picks / players
    raw_picks = sleeper.get_picks(draft_id) or []
    players_map = sleeper.get_players_nfl() or {}
    pick_log = sleeper.picks_to_internal_log(raw_picks, players_map, teams) or []

    total_picks = len(pick_log)
    next_overall = total_picks + 1
    rnd, pick_in_rnd, slot_on_clock = snake_position(next_overall, teams)

    my_slot = _detect_my_slot(users, sleeper.get_draft(draft_id) or {}, pick_log, seat_override, username)
    you_on_clock = (slot_on_clock == my_slot)
    st.markdown(
        f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — "
        f"**{slot_to_display_name(slot_on_clock, users)}** on the clock."
        + (" 🎯 _(That’s you)_" if you_on_clock else "")
    )
    st.caption(f"Using draft slot: {my_slot} ({'manual' if int(seat_override) > 0 else 'auto-detected'})")

    # Remove already-picked from your pool
    picked_names = sleeper.picked_player_names(raw_picks, players_map)
    taken_keys = {norm_name(n) for n in picked_names}

    # Evaluate availability (simple VBD)
    picks_until_next = compute_next_pick_window(teams, my_slot, next_overall)
    avail_df = evaluate_players(
        csv_df=csv_df,
        teams=teams,
        roster_positions=roster_positions,
        current_picks=list(taken_keys),            # remove drafted
        next_pick_window=picks_until_next          # for "make it back" calc
    )

    # Build your roster counts
    team_counts = _team_pos_counts_from_log(pick_log, teams)
    my_counts = team_counts.get(my_slot, {"QB": 0, "RB": 0, "WR": 0, "TE": 0, "K": 0, "DEF": 0})

    # FIRST PICK ONLY: lock strategy while on the clock
    if st.session_state.selected_strategy is None and you_on_clock and rnd == 1:
        st.markdown("### Choose Your Draft Strategy")
        default_choices = ["Zero RB", "Hero RB", "WR-Heavy", "Balanced"]
        choice = st.selectbox("Lock a strategy for the rest of the draft", options=STRATS, index=STRATS.index(default_choices[0]))
        if st.button("Lock Strategy"):
            st.session_state.selected_strategy = choice
            st.success(f"Locked: {choice}")

    if st.session_state.selected_strategy:
        st.info(f"**Locked strategy:** {st.session_state.selected_strategy}")
        plan = _make_base_plan(st.session_state.selected_strategy, rnd, rounds_total)
        st.caption("Next rounds plan: " + ", ".join([f"R{r}:{p}" for r, p in list(plan.items())[:4]]))

    # Need model based on target roster for locked strategy (or Balanced)
    targets = STRAT_TARGETS.get(st.session_state.selected_strategy or "Balanced", STRAT_TARGETS["Balanced"])
    base_need = {p: max(0, targets.get(p, 0) - my_counts.get(p, 0)) for p in ["QB", "RB", "WR", "TE"]}

    # Rank suggestions
    sugg = suggest(
        avail_df=avail_df,
        base_need=base_need,
        round_number=rnd,
        total_rounds=rounds_total,
        strategy_name=st.session_state.selected_strategy or "Balanced",
        qb_cap=QB_ROSTER_CAP,
        k_def_last_rounds_only=K_DEF_LAST_ROUNDS_ONLY
    )

    # Display suggestions
    st.markdown("### Suggested Picks (Top 8)")
    show_cols = ["PLAYER", "POS", "TEAM", "TIER", "ADP", "EVAL_PTS", "VBD", "WHY"]
    st.dataframe(sugg[show_cols].head(8), use_container_width=True, height=420)

    # Debug
    with st.expander("Debug"):
        st.caption(f"Picks normalized: {len(pick_log)} | Next overall: {next_overall}")
        st.caption(f"My slot: {my_slot} | On clock slot: {slot_on_clock}")
        st.caption(f"My counts: {my_counts}")
        st.caption(f"Targets: {targets}")
        st.caption(f"Removed {len(taken_keys)} drafted players from pool.")

# --------------------------
# Mock tab (simple mirror)
# --------------------------
def mock_tab(csv_df):
    st.subheader("Mock Draft (Sleeper)")
    url_or_id = st.text_input("Sleeper Mock URL or draft_id", value="")
    if st.button("Load / Re-sync Mock"):
        if not csv_df is None and not csv_df.empty and url_or_id.strip():
            draft_id = sleeper.parse_draft_id_from_url(url_or_id.strip())
            if not draft_id:
                st.error("Could not parse draft_id from input.")
                return
            dmeta = sleeper.get_draft(draft_id) or {}
            teams = int(dmeta.get("settings", {}).get("teams", dmeta.get("teams", 12)) or 12)
            rounds_total = int(dmeta.get("settings", {}).get("rounds", 15) or 15)
            roster_positions = DEFAULT_ROSTER_POSITIONS
            picks = sleeper.get_picks(draft_id) or []
            players_map = sleeper.get_players_nfl() or {}
            picked_names = sleeper.picked_player_names(picks, players_map)
            taken_keys = {norm_name(n) for n in picked_names}
            avail_df = evaluate_players(csv_df, teams, roster_positions, list(taken_keys))
            rnd, pick_in_rnd, slot = snake_position(len(picks) + 1, teams)
            st.success(f"Mock {draft_id} loaded — {len(picks)} picks synced, teams={teams}, rounds={rounds_total}.")
            st.markdown(f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd}, Slot {slot}")
            # Just show top suggestions in mock (no practice drafting in this stable build)
            targets = STRAT_TARGETS["Balanced"]
            base_need = {"QB":1,"RB":2,"WR":2,"TE":1}
            sugg = suggest(avail_df, base_need, rnd, rounds_total, "Balanced", qb_cap=QB_ROSTER_CAP, k_def_last_rounds_only=K_DEF_LAST_ROUNDS_ONLY)
            st.markdown("### Suggested Picks (Top 8)")
            st.dataframe(sugg[["PLAYER","POS","TEAM","TIER","ADP","EVAL_PTS","VBD","WHY"]].head(8), use_container_width=True)
        else:
            st.info("Upload players file and paste a mock URL/ID.")

# --------------------------
# Main
# --------------------------
def main():
    st.title("Fantasy Football Draft Assistant — Stable VBD")
    csv_df, league_id, username, seat_override, teams_default, rounds_default, poll_secs, auto_live = sidebar_controls()

    tabs = st.tabs(["Live Draft", "Mock Draft"])
    with tabs[0]:
        live_tab(csv_df, league_id, username, seat_override, teams_default, rounds_default, poll_secs, auto_live)
    with tabs[1]:
        mock_tab(csv_df)

if __name__ == "__main__":
    main()

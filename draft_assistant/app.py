import os
import sys
import time
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
    user_roster_id,          # uses users + rosters + username
    slot_to_display_name,
    remove_players_by_name,
)

st.set_page_config(page_title="FF Draft Assistant (VBD)", layout="wide")

DEFAULT_CSV_PATH = os.environ.get("FFDA_CSV_PATH", "")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
os.makedirs(DATA_DIR, exist_ok=True)

# === QB roster cap (env overrideable) ===
QB_ROSTER_CAP = int(os.environ.get("FFDA_QB_CAP", "2"))

# =========================
# Caching
# =========================

@st.cache_resource(show_spinner=False)
def sleeper_players_cache():
    # Big JSON; cache for session
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

    return csv_df, weights, league_id, username, int(seat), poll_secs, auto_live

# =========================
# Helper: compute VONA window
# =========================

def compute_next_pick_window(teams: int, seat: int, current_overall_pick: int) -> int:
    """
    Snake-draft distance (in selections) from the current overall pick to your next pick.
    Accepts seat=0 (auto): returns teams as a safe default; live/mocks compute a real seat when available.
    """
    if not (1 <= seat <= teams):
        return teams
    rnd = (current_overall_pick - 1) // teams + 1
    pos = (current_overall_pick - 1) % teams + 1
    my_pos_this = seat if rnd % 2 == 1 else (teams - seat + 1)
    if my_pos_this >= pos:
        return my_pos_this - pos
    my_pos_next = (teams - seat + 1) if rnd % 2 == 1 else seat
    return (teams - pos) + my_pos_next

def _apply_qb_cap_to_suggestions(sugg_df: pd.DataFrame, qb_have: int, qb_cap: int) -> pd.DataFrame:
    """
    Enforce a hard roster cap on QB suggestions.
    - If qb_have >= qb_cap: hide all QBs.
    - Else show at most (qb_cap - qb_have) QBs within the suggestions list.
    """
    if sugg_df is None or sugg_df.empty:
        return sugg_df
    remaining = max(0, qb_cap - max(0, qb_have))
    qbs = sugg_df[sugg_df["POS"] == "QB"]
    non_qbs = sugg_df[sugg_df["POS"] != "QB"]
    if remaining <= 0:
        return non_qbs.head(len(sugg_df)).reset_index(drop=True)
    # keep the list order as given by `suggest(...)`
    capped_qbs = qbs.head(remaining)
    combined = pd.concat([non_qbs, capped_qbs], ignore_index=True)
    # keep topk size the same as input
    return combined.head(len(sugg_df)).reset_index(drop=True)

# =========================
# LIVE TAB
# =========================

def live_tab(csv_df, weights, league_id, username, seat_override, poll_secs, auto_live):
    st.subheader("Live Draft (Sleeper)")
    c1, c2 = st.columns([1, 1])
    with c1:
        if st.button("🔄 Pull latest from Sleeper (Live)"):
            st.rerun()
    with c2:
        if auto_live:
            # Use browser meta refresh (reliable on Streamlit Cloud)
            st.markdown(f"<meta http-equiv='refresh' content='{int(poll_secs)}'>", unsafe_allow_html=True)
            st.caption(f"Auto-refresh every {poll_secs}s.")

    if not league_id:
        st.info("Enter your Sleeper League ID in the sidebar.")
        return

    # League & draft info
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

    # Choose a draft: prefer active/pre_draft
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

    # Picks & players
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

    # Identify your roster slot (auto via rosters; fallback to sidebar seat)
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

    # Availability via names
    picked_names = sleeper.picked_player_names(picks, players_map)
    taken_keys = [norm_name(n) for n in picked_names]
    next_window = compute_next_pick_window(teams, my_slot, next_overall)

    avail_df, starters = evaluate_players(
        csv_df, SCORING_DEFAULT, teams, roster_positions, weights, current_picks=taken_keys, next_pick_window=next_window
    )

    # Lightweight needs by position based on *your* picks so far
    my_picks = [p for p in picks if str(p.get("roster_id")) == str(my_slot)]
    my_pos_counts = {"QB":0,"RB":0,"WR":0,"TE":0}
    for p in my_picks:
        pos = str((p.get("metadata") or {}).get("position") or "").upper().replace("DST","DEF")
        if pos in my_pos_counts:
            my_pos_counts[pos] += 1
    # Guardrails: starters +1 target depth
    need = {}
    for pos in ["QB","RB","WR","TE"]:
        want = int(sum(1 for rp in roster_positions if rp == pos))
        want = max(1, want)  # at least 1
        need[pos] = max(0, (want + 1) - my_pos_counts.get(pos,0))

    # Rank suggestions (top 8)
    sugg = suggest(avail_df, need, weights, topk=8)

    # === Enforce QB cap in suggestions ===
    qb_have = int(my_pos_counts.get("QB", 0))
    sugg = _apply_qb_cap_to_suggestions(sugg, qb_have=qb_have, qb_cap=QB_ROSTER_CAP)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Suggested Picks (Top 8)")
        st.dataframe(sugg.drop(columns=["PLAYER_KEY"]), use_container_width=True, height=360)
    with col2:
        st.markdown("### Player Board (Available)")
        show_cols = ["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD","VONA","INJURY_RISK","SOS_SEASON"]
        st.dataframe(
            avail_df[show_cols].sort_values(["VBD","EVAL_PTS"], ascending=False),
            use_container_width=True, height=360
        )

    with st.expander("Debug (Live)"):
        st.caption(f"Total picks fetched: {total_picks}")
        st.caption(f"My slot (auto/fallback): {my_slot}")
        st.caption(f"QB have / cap: {qb_have} / {QB_ROSTER_CAP}")
        if picks:
            st.json(picks[0])

# =========================
# MOCK TAB (stateful practice)
# =========================

def mock_tab(csv_df, weights):
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

    # Practice-only: let you indicate how many QBs you've already taken
    qb_have_practice = st.number_input("QBs drafted (practice)", min_value=0, max_value=QB_ROSTER_CAP, value=0, step=1)

    if clear_btn:
        st.session_state.pop("mock_state", None)
        st.success("Practice state cleared.")

    if csv_df is None or csv_df.empty:
        st.info("Upload/load your player file in the sidebar.")
        return

    # Initial load / re-sync
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

                # Build initial available board using actual teams/rounds context
                roster_positions = ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF"]
                avail_df, _ = evaluate_players(
                    csv_df, SCORING_DEFAULT, teams, roster_positions, weights, current_picks=taken_keys
                )

                st.session_state.mock_state = {
                    "draft_id": draft_id,
                    "teams": teams,
                    "rounds": rounds,
                    "picks": picks,
                    "available": avail_df.reset_index(drop=True),
                }
                st.success(f"Mock {draft_id} loaded — {len(picks)} picks synced, teams={teams}, rounds={rounds}.")

            except Exception as e:
                st.error(f"Mock load failed: {e}")

    if "mock_state" not in st.session_state:
        st.info("Load a mock to begin.")
        return

    S = st.session_state.mock_state
    teams = int(S["teams"])
    rounds = int(S["rounds"])
    picks = S["picks"]
    players_map = sleeper_players_cache()

    next_overall = len(picks) + 1
    rnd, pick_in_rnd, slot_on_clock = snake_position(next_overall, teams)
    st.write(f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — Slot {slot_on_clock}")

    if force_btn:
        try:
            new_picks = sleeper.get_picks(S["draft_id"]) or []
            if len(new_picks) != len(picks):
                S["picks"] = new_picks
                picks = new_picks
                # Recompute availability after new picks
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

    # Suggestions at the current moment
    picked_names = sleeper.picked_player_names(picks, players_map)
    taken_keys = [norm_name(n) for n in picked_names]
    roster_positions = ["QB","RB","RB","WR","WR","TE","FLEX","K","DEF"]
    next_window = compute_next_pick_window(teams, seat=teams, current_overall_pick=next_overall)  # neutral window

    avail_df, _ = evaluate_players(
        csv_df, SCORING_DEFAULT, teams, roster_positions, weights, current_picks=taken_keys, next_pick_window=next_window
    )
    S["available"] = avail_df.reset_index(drop=True)
    st.session_state.mock_state = S

    st.markdown("### Suggested Picks (Top 8)")
    sugg = suggest(S["available"], {"QB":1,"RB":2,"WR":2,"TE":1}, weights, topk=8)
    # === Enforce QB cap in suggestions (practice) ===
    sugg = _apply_qb_cap_to_suggestions(sugg, qb_have=qb_have_practice, qb_cap=QB_ROSTER_CAP)
    st.dataframe(sugg.drop(columns=["PLAYER_KEY"]), use_container_width=True)

    st.markdown("### Player Board (Available)")
    show_cols = ["PLAYER","TEAM","POS","TIER","ADP","EVAL_PTS","VBD","VONA","INJURY_RISK","SOS_SEASON"]
    st.dataframe(S["available"][show_cols].sort_values(["VBD","EVAL_PTS"], ascending=False), use_container_width=True)

    with st.expander("Debug (Mock)"):
        st.caption(f"Fetched picks: {len(picks)}")
        st.caption(f"QB have (practice) / cap: {qb_have_practice} / {QB_ROSTER_CAP}")
        if picks:
            st.json(picks[0])

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

    pos = st.multiselect("Position", ["QB","RB","WR","TE","K","DEF"], default=["RB","WR","TE","QB"])
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
    st.title("Fantasy Football Draft Assistant — VBD Overlay (Enhanced)")
    csv_df, weights, league_id, username, seat, poll_secs, auto_live = sidebar_controls()

    tabs = st.tabs(["Live Draft", "Mock Draft", "Player Board"])
    with tabs[0]:
        live_tab(csv_df, weights, league_id, username, seat, poll_secs, auto_live)
    with tabs[1]:
        mock_tab(csv_df, weights)
    with tabs[2]:
        board_tab(csv_df, weights)

if __name__ == "__main__":
    main()

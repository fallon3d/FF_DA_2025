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
from draft_assistant.core.utils import norm_name
from draft_assistant.core.run_detection import detect_runs

st.set_page_config(page_title="FF Draft Assistant (VBD)", layout="wide")
DEFAULT_CSV_PATH = os.environ.get("FFDA_CSV_PATH", "")

# =========================
# Caching
# =========================

@st.cache_data(show_spinner=False)
def load_csv(path_or_buffer):
    return pd.read_csv(path_or_buffer)

@st.cache_resource(show_spinner=False)
def sleeper_players_cache():
    try:
        return sleeper.get_players()
    except Exception:
        return {}

# =========================
# Sidebar
# =========================

def sidebar_controls():
    st.sidebar.header("Data")
    csv_choice = st.sidebar.radio("Player data source", ["Upload CSV", "Path"])
    csv_df = None
    if csv_choice == "Upload CSV":
        file = st.sidebar.file_uploader("Upload your player CSV", type=["csv"])
        if file:
            csv_df = load_csv(file)
    else:
        path = st.sidebar.text_input("CSV path", value=DEFAULT_CSV_PATH)
        if st.sidebar.button("Load path") and path:
            csv_df = load_csv(path)

    st.sidebar.header("Weights")
    inj_w = st.sidebar.slider("Injury penalty weight", 0.0, 1.0, 0.5, 0.05)
    sos_w = st.sidebar.slider("Schedule strength weight", 0.0, 0.3, 0.05, 0.01)
    usage_w = st.sidebar.slider("Usage/Upside weight", 0.0, 0.5, 0.05, 0.01)
    weights = {"inj_w": inj_w, "sos_w": sos_w, "usage_w": usage_w}

    st.sidebar.header("League")
    league_id = st.sidebar.text_input("Sleeper League ID", value="")
    username = st.sidebar.text_input("Your Sleeper username (optional)", value="Fallon3D")
    seat = st.sidebar.number_input("Your draft slot (1–12, optional)", 1, 12, 1)

    poll_secs = st.sidebar.slider("Auto-refresh seconds", 2, 30, 5, 1)
    auto = st.sidebar.toggle("Auto-refresh", value=False)

    return csv_df, weights, league_id, poll_secs, auto, username, int(seat)

# =========================
# Helpers: picks parsing & snake window
# =========================

def normalize_picks(picks):
    """Normalize Sleeper picks to list[dict] whether response is a list or a dict wrapper."""
    if isinstance(picks, dict):
        for key in ("picks", "draft_picks", "data"):
            val = picks.get(key)
            if isinstance(val, list):
                return val
        return []
    return picks if isinstance(picks, list) else []

def map_taken_from_picks(picks, players_index):
    """Return (taken_keys, pos_history) robustly from Sleeper picks."""
    taken_keys, pos_history = [], []
    for p in normalize_picks(picks):
        if not isinstance(p, dict):
            continue
        meta = p.get("metadata") if isinstance(p.get("metadata"), dict) else {}
        pid = str(p.get("player_id") or meta.get("player_id") or "")
        pos = meta.get("position") or meta.get("position_group") or ""
        full_name = None
        if pid and pid in players_index:
            info = players_index.get(pid) or {}
            full_name = (
                info.get("full_name")
                or f"{info.get('first_name','')} {info.get('last_name','')}".strip()
            )
            pos = info.get("position") or pos
        if not full_name:
            full_name = (
                meta.get("full_name")
                or meta.get("player")
                or f"{meta.get('first_name','')} {meta.get('last_name','')}".strip()
            ).strip()
        if full_name:
            taken_keys.append(norm_name(full_name))
        if pos:
            pos_history.append(str(pos).upper().replace("DST", "DEF"))
    return taken_keys, pos_history

def roster_need_state(my_roster_counts, starters_counts):
    """Soft roster guardrails: aim for starters + 1 depth by mid draft."""
    need = {}
    for pos in ["QB", "RB", "WR", "TE"]:
        have = int(my_roster_counts.get(pos, 0))
        want = int(starters_counts.get(pos, 0))
        target = max(want, 1) + 1
        need[pos] = max(0, target - have)
    return need

def fetch_live_state(league_id):
    """Fetch league, roster positions, user map, active draft_id, and picks (normalized)."""
    league = sleeper.get_league(league_id)
    teams = int(league.get("total_rosters", 12) or 12)
    roster_positions = league.get("roster_positions") or []
    users = sleeper.get_users_in_league(league_id)
    user_map = {u.get("display_name", "?"): u.get("user_id") for u in users}
    drafts = sleeper.get_drafts_for_league(league_id)
    draft_id = ""
    for d in drafts:
        status = (d.get("status") or "")
        if status in ("pre_draft", "in_progress"):
            draft_id = d.get("draft_id", "")
            break
    if not draft_id and drafts:
        draft_id = drafts[0].get("draft_id", "")
    picks = sleeper.get_draft_picks(draft_id) if draft_id else []
    picks = normalize_picks(picks)
    return league, teams, roster_positions, user_map, draft_id, picks

def compute_next_pick_window(teams: int, seat: int, current_overall_pick: int) -> int:
    """Snake-draft distance (in selections) from current overall pick to your next pick."""
    if not (1 <= seat <= teams):
        return teams
    rnd = (current_overall_pick - 1) // teams + 1
    pos = (current_overall_pick - 1) % teams + 1
    my_pos_this = seat if rnd % 2 == 1 else (teams - seat + 1)
    if my_pos_this >= pos:
        return my_pos_this - pos
    my_pos_next = (teams - seat + 1) if rnd % 2 == 1 else seat
    return (teams - pos) + my_pos_next

# =========================
# Tabs
# =========================

def live_tab(csv_df, weights, league_id, poll_secs, auto_refresh, username, seat):
    st.subheader("Live Draft (Sleeper)")
    if not league_id:
        st.info("Enter your Sleeper League ID in the sidebar to begin.")
        return

    with st.spinner("Loading league & draft..."):
        try:
            league, teams, roster_positions, user_map, draft_id, picks = fetch_live_state(league_id)
        except Exception as e:
            st.error(f"Failed to load league/draft: {e}")
            return

    st.caption(f"Detected **{teams} teams**; roster positions: {roster_positions}")
    if not draft_id:
        st.warning("No draft found for this league.")
        return
    st.write(f"Draft ID: `{draft_id}`")

    # Identify your user by username first; else allow manual selection
    my_user_id = None
    matched_key = None
    if user_map and username:
        for disp, uid in user_map.items():
            if str(disp).strip().lower() == str(username).strip().lower():
                my_user_id = uid
                matched_key = disp
                break
    if my_user_id:
        st.success(f"Matched Sleeper user: **{matched_key}**")
    else:
        if user_map:
            my_display = st.selectbox("Select your user/team (for roster tracking)", list(user_map.keys()))
            my_user_id = user_map.get(my_display)

    players_index = sleeper_players_cache()
    taken_keys, pos_history = map_taken_from_picks(picks, players_index)

    # My roster counts from picks already made
    my_roster_counts = {"QB": 0, "RB": 0, "WR": 0, "TE": 0}
    if my_user_id:
        mine = [p for p in picks if isinstance(p, dict) and str(p.get("picked_by")) == str(my_user_id)]
        my_pos = []
        for p in mine:
            pid = str(p.get("player_id") or (p.get("metadata") or {}).get("player_id") or "")
            pos = players_index.get(pid, {}).get("position", "") if pid in players_index else ""
            if pos:
                my_pos.append(pos.upper().replace("DST", "DEF"))
        my_roster_counts = {k: my_pos.count(k) for k in ["QB", "RB", "WR", "TE"]}

    if csv_df is None or csv_df.empty:
        st.warning("Upload or load your player CSV in the sidebar.")
        return

    # Evaluate & suggest (snake-aware VONA)
    current_overall_pick = len(picks) + 1
    next_window = compute_next_pick_window(teams, seat, current_overall_pick)
    try:
        avail_df, starters = evaluate_players(
            csv_df, SCORING_DEFAULT, teams, roster_positions, weights, taken_keys, next_pick_window=next_window
        )
    except Exception as e:
        st.error(f"CSV evaluation failed: {e}")
        return

    needs = roster_need_state(my_roster_counts, starters)
    sugg = suggest(avail_df, needs, weights, topk=8)

    col1, col2 = st.columns([1, 1])
    with col1:
        st.markdown("### Suggested Picks (Top 8)")
        st.dataframe(sugg.drop(columns=["PLAYER_KEY"]), use_container_width=True, height=340)
    with col2:
        st.markdown("### Player Board (Available)")
        show_cols = ["PLAYER", "TEAM", "POS", "TIER", "ADP", "EVAL_PTS", "VBD", "VONA", "INJURY_RISK", "SOS_SEASON"]
        st.dataframe(
            avail_df[show_cols].sort_values(["VBD", "EVAL_PTS"], ascending=False),
            use_container_width=True,
            height=340
        )

    runs = detect_runs(pos_history, lookback=10, threshold=3)
    if runs:
        st.warning(" | ".join(runs))
    st.caption(f"Taken so far: {len(taken_keys)} | Recent positions: {' → '.join(pos_history[-12:])}")

    # Manual/auto refresh
    colr1, colr2 = st.columns([1, 1])
    with colr1:
        if st.button("🔄 Pull latest from Sleeper"):
            st.rerun()
    with colr2:
        if auto_refresh:
            time.sleep(poll_secs)
            st.rerun()

def mock_tab(csv_df, weights, seat):
    st.subheader("Mock Draft (Sleeper URL)")
    url = st.text_input("Paste Sleeper mock draft URL")
    colbtn1, _ = st.columns([1, 3])
    with colbtn1:
        force = st.button("🔄 Pull latest from Sleeper (Mock)")

    if not url:
        st.info("Paste a Sleeper mock URL to mirror the room.")
        return

    draft_id = sleeper.parse_mock_draft_id_from_url(url)
    if not draft_id:
        st.error("Could not parse draft id from URL.")
        return
    st.write(f"Draft ID: `{draft_id}`")

    players_index = sleeper_players_cache()
    try:
        picks_raw = sleeper.get_draft_picks(draft_id)
    except Exception as e:
        st.error(f"Failed to load mock picks: {e}")
        return

    picks = normalize_picks(picks_raw)
    taken_keys, pos_history = map_taken_from_picks(picks, players_index)

    # Debug panel
    with st.expander("Debug (mock)"):
        st.caption(f"Fetched picks: {len(picks)}")
        if len(picks) > 0:
            st.json(picks[0])

    teams = 12  # generic for public mocks
    roster_positions = ["QB", "RB", "RB", "WR", "WR", "TE", "FLEX", "K", "DEF"]

    if csv_df is None or csv_df.empty:
        st.warning("Upload or load your player CSV in the sidebar.")
        return

    current_overall_pick = len(picks) + 1
    next_window = compute_next_pick_window(teams, seat, current_overall_pick)
    try:
        avail_df, starters = evaluate_players(
            csv_df, SCORING_DEFAULT, teams, roster_positions, weights, taken_keys, next_pick_window=next_window
        )
    except Exception as e:
        st.error(f"CSV evaluation failed: {e}")
        return

    needs = {"QB": 1, "RB": 2, "WR": 2, "TE": 1}
    sugg = suggest(avail_df, needs, weights, topk=8)

    st.markdown("### Suggested Picks (Top 8)")
    st.dataframe(sugg.drop(columns=["PLAYER_KEY"]), use_container_width=True)

    st.markdown("### Player Board (Available)")
    show_cols = ["PLAYER", "TEAM", "POS", "TIER", "ADP", "EVAL_PTS", "VBD", "VONA", "INJURY_RISK", "SOS_SEASON"]
    st.dataframe(avail_df[show_cols].sort_values(["VBD", "EVAL_PTS"], ascending=False), use_container_width=True)

    if force:
        st.rerun()

def board_tab(csv_df, weights):
    st.subheader("Player Board & Filters")
    if csv_df is None or csv_df.empty:
        st.info("Upload or load your player CSV in the sidebar.")
        return

    teams = st.number_input("Teams", 8, 16, 12, 1)
    roster_positions = st.text_input("Roster positions (comma)", value="QB,RB,RB,WR,WR,TE,FLEX,K,DEF")
    roster_positions = [p.strip().upper() for p in roster_positions.split(",") if p.strip()]

    try:
        avail_df, starters = evaluate_players(
            csv_df, SCORING_DEFAULT, int(teams), roster_positions, weights, current_picks=[]
        )
    except Exception as e:
        st.error(f"CSV evaluation failed: {e}")
        return

    pos = st.multiselect("Position", ["QB", "RB", "WR", "TE", "K", "DEF"], default=["RB", "WR", "TE", "QB"])
    team = st.text_input("Team filter (e.g., KC, SF)")
    tier_max = st.number_input("Max Tier (optional)", 0, 20, 20, 1)

    filt = avail_df[avail_df["POS"].isin(pos)].copy()
    if team:
        filt = filt[filt["TEAM"].fillna("").str.upper().str.contains(team.strip().upper(), na=False)]
    if tier_max < 20:
        filt = filt[(filt["TIER"].isna()) | (filt["TIER"] <= tier_max)]

    st.dataframe(filt.sort_values(["VBD", "EVAL_PTS"], ascending=False), use_container_width=True)
    csv = filt.to_csv(index=False).encode()
    st.download_button("Download filtered board CSV", csv, "filtered_board.csv", "text/csv")

# =========================
# Main
# =========================

def main():
    st.title("Fantasy Football Draft Assistant — VBD Overlay")
    csv_df, weights, league_id, poll_secs, auto_refresh, username, seat = sidebar_controls()

    tabs = st.tabs(["Live Draft", "Mock Draft", "Player Board"])
    with tabs[0]:
        live_tab(csv_df, weights, league_id, poll_secs, auto_refresh, username, seat)
    with tabs[1]:
        mock_tab(csv_df, weights, seat)
    with tabs[2]:
        board_tab(csv_df, weights)

if __name__ == "__main__":
    main()

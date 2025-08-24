import os
import sys
import math
from typing import Dict, List, Tuple, Optional
import pandas as pd
import streamlit as st

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

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
ADP_GUARD_TOL = int(os.environ.get("FFDA_ADP_GUARD_TOL", "6"))  # <-- stronger default

@st.cache_resource(show_spinner=False)
def sleeper_players_cache():
    try:
        return sleeper.get_players_nfl()
    except Exception:
        return {}

@st.cache_data(show_spinner=False)
def load_local_csv(path: str):
    return read_player_table(path)

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
    usage_w = st.sidebar.slider("Usage/Upside weight (reserved)", 0.0, 0.5, 0.05, 0.01)
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

def compute_next_pick_window(teams: int, seat: int, current_overall_pick: int) -> int:
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

def _detect_my_slot(users: List[dict], draft_meta: dict, pick_log: List[dict], seat_override: int, username: str, rosters: List[dict]) -> int:
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
    if my_uid:
        for r in rosters or []:
            if str(r.get("owner_id","")) == str(my_uid):
                try:
                    rid = int(r.get("roster_id", 0))
                    if rid > 0:
                        return rid
                except Exception:
                    pass
    if my_uid:
        for p in pick_log or []:
            if str(p.get("team") or "") == str(my_uid):
                try:
                    s = int(p.get("slot", 0))
                    if s > 0:
                        return s
                except Exception:
                    pass
    return 1

def _count_owned_for_slot_raw(raw_picks: List[dict], players_map: dict, my_slot: int) -> Dict[str,int]:
    counts = {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0}
    if my_slot <= 0:
        return counts
    for p in raw_picks or []:
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
        if pos in ("DST","D/ST","DEFENSE","TEAM D","TEAM DEF"):
            pos = "DEF"
        if pos in counts:
            counts[pos] += 1
    return counts

def _ensure_k_def_in_suggestions(sugg_df: pd.DataFrame, avail_df: pd.DataFrame, rnd: int, total_rounds: int, include_anytime: bool, need_k: int, need_def: int) -> pd.DataFrame:
    if sugg_df is None or sugg_df.empty:
        return sugg_df
    have_k = (sugg_df["POS"]=="K").any()
    have_d = (sugg_df["POS"]=="DEF").any()
    force_window = rnd >= total_rounds - 1 or (rnd >= total_rounds - 2 and (need_k > 0 or need_def > 0))
    if (have_k and have_d) and not force_window:
        return sugg_df
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

def _apply_qb_cap(sugg_df: pd.DataFrame, qbs_owned: int, cap: int) -> pd.DataFrame:
    if sugg_df is None or sugg_df.empty: return sugg_df
    remaining = max(0, cap - max(0, qbs_owned))
    qbs = sugg_df[sugg_df["POS"]=="QB"]
    non = sugg_df[sugg_df["POS"]!="QB"]
    if remaining <= 0:
        return non.head(len(sugg_df)).reset_index(drop=True)
    return pd.concat([non, qbs.head(remaining)], ignore_index=True).head(len(sugg_df)).reset_index(drop=True)

# ---------- ADP guard helpers ----------
def _should_apply_adp_guard(avail_df: pd.DataFrame, next_overall: int, raw_picks_len: int, expected_picks: int, tol: int) -> bool:
    if raw_picks_len < expected_picks:
        return True
    if "ADP" in avail_df.columns:
        try:
            m = pd.to_numeric(avail_df["ADP"], errors="coerce").min()
            if pd.notna(m):
                return m < (next_overall - tol)
        except Exception:
            pass
    return False

def _apply_adp_guard(df: pd.DataFrame, next_overall: int, tol: int) -> pd.DataFrame:
    if df is None or df.empty or "ADP" not in df.columns:
        return df
    floor_adp = max(1, next_overall - tol)
    keep = df["ADP"].isna() | (pd.to_numeric(df["ADP"], errors="coerce") >= floor_adp)
    return df[keep].reset_index(drop=True)

# -------------------- LIVE TAB --------------------

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

    my_slot = _detect_my_slot(users, draft_meta, pick_log, seat_override, username, rosters)
    you_on_clock = (slot_on_clock == my_slot)

    st.markdown(
        f"**Current Pick:** Round {rnd}, Pick {pick_in_rnd} — **{team_display}** on the clock."
        + (" 🎯 _(That’s you)_" if you_on_clock else "")
    )
    manual_used = (seat_override is not None and int(seat_override) > 0)
    st.caption(f"Using draft slot: {my_slot} ({'manual' if manual_used else 'auto-detected'})")

    if csv_df is None or csv_df.empty:
        st.warning("Upload/load your player file in the sidebar.")
        return

    picked_names = sleeper.picked_player_names(raw_picks, players_map)
    taken_keys = [norm_name(n) for n in picked_names]
    picks_until_next = compute_next_pick_window(teams, my_slot, next_overall)

    avail_df, _ = evaluate_players(
        csv_df, SCORING_DEFAULT, teams, roster_positions, weights,
        current_picks=taken_keys, next_pick_window=picks_until_next
    )

    # ADP guard: apply if picks are behind OR top ADP looks impossibly early
    expected_picks = max(0, next_overall - 1)
    if _should_apply_adp_guard(avail_df, next_overall, len(raw_picks), expected_picks, ADP_GUARD_TOL):
        avail_df = _apply_adp_guard(avail_df, next_overall, ADP_GUARD_TOL)

    team_counts = _team_pos_counts_from_log(pick_log, teams)
    my_counts = team_counts.get(my_slot, {"QB":0,"RB":0,"WR":0,"TE":0,"K":0,"DEF":0})
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

    strat_ranked = score_strategies(avail_df, my_counts, starters, rnd, teams, my_slot, picks_until_next, runs, rounds_total)
    current = strat_ranked[0]
    targets = STRAT_TARGETS.get(current["name"], STRAT_TARGETS["Balanced"])

    top3 = strat_ranked[:3]
    render_strategy_choices(top3, rounds_total)

    base_need = {"QB":0,"RB":0,"WR":0,"TE":0}
    for pos in base_need:
        want = max(1, targets.get(pos, 0))
        base_need[pos] = max(0, want - my_counts.get(pos, 0))

    sugg = suggest(
        avail_df,
        base_need,
        weights,
        topk=8,
        strategy_name=current["name"],
        round_number=rnd,
        total_rounds=rounds_total,
    )

    qb_have = int(my_counts.get("QB", 0))
    sugg = _apply_qb_cap(sugg, qb_have, QB_ROSTER_CAP)

    need_k = max(0, targets.get("K",1) - my_counts.get("K",0))
    need_def = max(0, targets.get("DEF",1) - my_counts.get("DEF",0))
    sugg = _ensure_k_def_in_suggestions(
        sugg, avail_df, rnd, rounds_total, include_k_def_anytime, need_k, need_def
    )

    if not sugg.empty:
        top_pos = sugg.iloc[0]["POS"]
        depth_top, edge_top = _tier_depth(avail_df, top_pos)
        alt_best = avail_df[avail_df["POS"] != top_pos].sort_values(["VBD","EVAL_PTS"], ascending=False).head(1)
        if not alt_best.empty:
            alt_edge = float(alt_best.iloc[0].get("VBD", 0.0))
            if depth_top <= 1 or (edge_top - alt_edge) <= 5:
                st.info("Run/tier cliff detected — pivoting to best overall VBD for this pick.")
                sugg = sugg.sort_values(["VBD","EVAL_PTS"], ascending=False).head(8)

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

    render_strategy_panel(current, targets, rounds_total)
    render_strategy_health(my_counts, targets, current.get("plan", {}), rounds_total)

    with st.expander("Debug (Live)"):
        st.caption(f"Picks fetched (normalized): {len(pick_log)}")
        st.caption(f"Resolved my_slot: {my_slot} | On clock slot: {slot_on_clock}")
        st.caption(f"Expected picks so far: {next_overall-1} | Raw picks returned: {len(raw_picks)}")
        st.caption(f"Picks until next: {picks_until_next} | Recent runs: {runs}")
        st.caption(f"Owned counts (my team): {my_counts}")

# -------------------- MOCK TAB / BOARD TAB / score_strategies etc. --------------------
# (Unchanged from your last working version; keep your existing implementations
# of score_strategies, render_strategy_choices/panel/health, mock_tab, board_tab, and main.)

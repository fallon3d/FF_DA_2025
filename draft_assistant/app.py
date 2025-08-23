# FILE: pages/2_Rotation_Builder.py
from __future__ import annotations
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st

from rotation_core.constants import positions_for, CATEGORY_MAP
from rotation_core.models import Config, Player
from rotation_core.assignment import assign_one_rotation
from rotation_core.suitability import preference_rank
from rotation_core.io import generate_cards

st.title("2) Rotation Builder — First Lineup & Game Mode")

# ---- segmented_control fallback (older Streamlit) ----
_HAS_SEG = hasattr(st, "segmented_control")
def seg(label: str, options: List[str], *, key=None, default=None, help=None, horizontal=True):
    if _HAS_SEG:
        return st.segmented_control(label, options, default=default, key=key, help=help)
    idx = options.index(default) if default in options else 0
    return st.radio(label, options, index=idx, key=key, help=help, horizontal=horizontal)

if "roster" not in st.session_state:
    st.warning("Please import a roster on the **1_Roster** page first.")
    st.stop()

# ---------- Helpers ----------
def _players_present(df: pd.DataFrame) -> List[Player]:
    return [Player(**row) for _, row in df.iterrows() if int(row.get("IsPresent", 1)) == 1]

def _eligible_candidates(df: pd.DataFrame, pos: str, side: str) -> List[str]:
    cands: List[str] = []
    for _, row in df.iterrows():
        if int(row.get("IsPresent", 1)) != 1:
            continue
        p = Player(**row)
        if preference_rank(p, pos, side) is not None:
            cands.append((p.Name or "").strip())
    return cands

def _build_config() -> Config:
    ss = st.session_state
    return Config(
        fairness=ss["fairness_slider"],
        formation=(ss["def_formation"] if ss["mode_side"] == "Defense" else None),
        num_rotations=1,
        max_consecutive=int(ss.get("max_consecutive", 0)),
    )

def _filter_positions_map(d: Dict[str, str], positions: List[str]) -> Dict[str, str]:
    """Keep only keys for current positions; normalize empty to ''."""
    allowed = set(positions)
    return {pos: (d.get(pos) or "") for pos in positions if pos in allowed}

# ---------- Session state ----------
ss = st.session_state
ss.setdefault("mode_side", "Offense")
ss.setdefault("def_formation", "5-3")
ss.setdefault("fairness_slider", "Balanced")   # Fairness-first | Balanced | Win-push
ss.setdefault("num_rotations_target", 4)
ss.setdefault("max_consecutive", 0)
ss.setdefault("first_lineup_locked", False)
ss.setdefault("appearances_all", {})           # Name -> count
ss.setdefault("consecutive_counts", {})        # Name -> consecutive series
ss.setdefault("series_history", [])            # [{turn, lineup}]
ss.setdefault("current_turn", 1)
ss.setdefault("pins", {})                      # pos -> Name (one-shot)
ss.setdefault("excludes", {})                  # Name -> bool (one-shot)
ss.setdefault("current_effective", {})         # pos -> Name
ss.setdefault("next_effective", {})            # pos -> Name
ss.setdefault("first_series_plan", {})         # pos -> Name
ss.setdefault("last_sig", None)                # (mode_side, def_formation)

# ---------- Sidebar controls ----------
with st.sidebar:
    st.header("Rotation Settings")
    ss["mode_side"] = st.radio("Segment", ["Offense", "Defense"], horizontal=True)
    if ss["mode_side"] == "Defense":
        ss["def_formation"] = seg("Defense Formation", ["5-3", "4-4"], default=ss["def_formation"])
    ss["fairness_slider"] = st.select_slider(
        "Fairness vs Strength",
        options=["Fairness-first", "Balanced", "Win-push"],
        value=ss["fairness_slider"],
    )
    ss["num_rotations_target"] = st.number_input("Target rotations (game plan)", 1, 12, ss["num_rotations_target"])
    ss["max_consecutive"] = st.number_input("Max consecutive series (0=off)", 0, 5, ss["max_consecutive"])
    st.divider()
    st.caption("Pins & excludes affect the **next** build only.")
    # Pins / excludes pickers
    pos_list_for_picker = positions_for("offense" if ss["mode_side"] == "Offense" else "defense", ss.get("def_formation"))
    pin_pos = st.selectbox("Pin position", ["(none)"] + pos_list_for_picker, index=0)
    # normalize names for options
    roster_df: pd.DataFrame = ss.roster.copy()
    if "Name" in roster_df.columns:
        roster_df["Name"] = roster_df["Name"].astype(str).str.strip()
    all_names = [r["Name"] for _, r in roster_df.iterrows() if int(r.get("IsPresent", 1)) == 1]
    pin_name = st.selectbox("Pin player (eligible preferred)", ["(none)"] + all_names, index=0)
    c1, c2 = st.columns(2)
    if c1.button("Set Pin"):
        if pin_pos != "(none)" and pin_name != "(none)":
            ss["pins"][pin_pos] = pin_name
            st.success(f"Pinned {pin_name} at {pin_pos} for next build.")
    if c2.button("Clear Pins"):
        ss["pins"].clear()
    st.divider()
    exc_name = st.selectbox("Exclude player (next build only)", ["(none)"] + all_names, index=0)
    e1, e2 = st.columns(2)
    if e1.button("Add Exclude"):
        if exc_name != "(none)":
            ss["excludes"][exc_name] = True
            st.info(f"Excluded {exc_name} for next build.")
    if e2.button("Clear Excludes"):
        ss["excludes"].clear()

# ---------- Normalize roster names (whitespace) ----------
roster_df: pd.DataFrame = ss.roster.copy()
if "Name" in roster_df.columns:
    roster_df["Name"] = roster_df["Name"].astype(str).str.strip()

players_present = _players_present(roster_df)
if not players_present:
    st.warning("No present players (IsPresent=1). Toggle attendance on the Roster page.")
    st.stop()

# ---------- Detect segment/formation change & sanitize lineups ----------
side = "offense" if ss["mode_side"] == "Offense" else "defense"
positions = positions_for(side, ss.get("def_formation"))
sig_now = (ss["mode_side"], ss.get("def_formation"))
if ss["last_sig"] != sig_now:
    ss["last_sig"] = sig_now
    ss["first_series_plan"] = _filter_positions_map(ss.get("first_series_plan", {}), positions)
    ss["current_effective"] = _filter_positions_map(ss.get("current_effective", {}), positions)
    ss["next_effective"] = _filter_positions_map(ss.get("next_effective", {}), positions)

# ---------- Step 1 — Quick rating ----------
st.subheader("Step 1 — Quick rating (Role & Energy)")
role_opts = ["Explorer", "Connector", "Driver"]
energy_opts = ["Low", "Medium", "High"]

hdr1, hdr2, hdr3 = st.columns(3)
with hdr1: st.caption("Player")
with hdr2: st.caption("Role Today")
with hdr3: st.caption("Energy Today")

for p in players_present:
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(p.Name)
    with c2:
        val = seg(" ", role_opts, key=f"role_{p.Name}", default=p.RoleToday or "Connector")
        roster_df.loc[roster_df["Name"] == p.Name, "RoleToday"] = val
    with c3:
        ev = seg("  ", energy_opts, key=f"energy_{p.Name}", default=p.EnergyToday or "Medium")
        roster_df.loc[roster_df["Name"] == p.Name, "EnergyToday"] = ev

st.divider()

# ---------- Step 2 — Build 1st Lineup ----------
st.subheader("Step 2 — Build 1st Lineup")
cfg = _build_config()

if not ss["first_series_plan"]:
    assignment, _ = assign_one_rotation(
        roster_df, side, ss.get("def_formation"), cfg,
        appearances=ss["appearances_all"], quotas=None,
        consecutive=ss["consecutive_counts"], pins=ss["pins"], excludes=ss["excludes"]
    )
    ss["first_series_plan"] = _filter_positions_map(assignment, positions)

# Editable table
edits: Dict[str, Optional[str]] = {}
hc1, hc2, hc3 = st.columns(3)
with hc1: st.caption("Position")
with hc2: st.caption("Player")
with hc3: st.caption("Category")

for pos in positions:
    options = _eligible_candidates(roster_df, pos, side)
    current = (ss["first_series_plan"].get(pos) or "").strip()
    if current and current not in options:
        current = ""
    c1, c2, c3 = st.columns(3)
    with c1:
        st.write(pos)
    with c2:
        pick = st.selectbox(
            f"{pos} pick",
            options=["(empty)"] + options,
            index=(["(empty)"] + options).index(current) if current in options else 0,
            key=f"pick_{pos}",
        )
        edits[pos] = None if pick == "(empty)" else pick
    with c3:
        st.write(CATEGORY_MAP.get(pos, "GEN"))

for pos, pick in edits.items():
    ss["first_series_plan"][pos] = (pick or "").strip()

c1, c2, c3 = st.columns(3)
with c1:
    if st.button("Auto-Fill Best (respect eligibility & fairness bias)"):
        assignment, _ = assign_one_rotation(
            roster_df, side, ss.get("def_formation"), cfg,
            appearances=ss["appearances_all"], quotas=None,
            consecutive=ss["consecutive_counts"], pins=ss["pins"], excludes=ss["excludes"]
        )
        ss["first_series_plan"] = _filter_positions_map(assignment, positions)
        st.success("Auto-filled best 1st lineup.")
with c2:
    if not ss["first_lineup_locked"] and st.button("Lock 1st Lineup"):
        # ✅ Duplicate check ONLY across CURRENT positions
        used: set[str] = set()
        dupes: set[str] = set()
        for pos in positions:
            name = (ss["first_series_plan"].get(pos) or "").strip()
            if not name:
                continue
            if name in used:
                dupes.add(name)
            used.add(name)
        if dupes:
            st.error(f"Duplicate player(s) in lineup: {', '.join(sorted(dupes))}")
        else:
            ss["current_effective"] = _filter_positions_map(ss["first_series_plan"], positions)
            ss["first_lineup_locked"] = True
            st.success("1st lineup locked. Ready for Game.")
with c3:
    if st.button("Clear 1st Lineup"):
        ss["first_series_plan"] = {}

st.divider()

# ---------- Step 3 — Game Mode ----------
st.subheader("Step 3 — Game Mode")
if not ss["first_lineup_locked"]:
    st.info("Lock the 1st lineup to start the game loop.")
    st.stop()

def _commit_series(lineup: Dict[str, str]):
    used = set()
    for pos in positions:
        name = (lineup.get(pos) or "").strip()
        if not name:
            continue
        ss["appearances_all"][name] = ss["appearances_all"].get(name, 0) + 1
        used.add(name)
    all_names = [p.Name for p in _players_present(roster_df)]
    for nm in all_names:
        if nm in used:
            ss["consecutive_counts"][nm] = ss["consecutive_counts"].get(nm, 0) + 1
        else:
            ss["consecutive_counts"][nm] = 0

def _build_next_series():
    nxt, _ = assign_one_rotation(
        roster_df, side, ss.get("def_formation"), cfg,
        appearances=ss["appearances_all"], quotas=None,
        consecutive=ss["consecutive_counts"], pins=ss["pins"], excludes=ss["excludes"]
    )
    ss["next_effective"] = _filter_positions_map(nxt, positions)

prev_lineup = ss["series_history"][-1]["lineup"] if ss["series_history"] else {}
cA, cB, cC = st.columns(3)

with cA:
    st.caption("Previous")
    if prev_lineup:
        for pos in positions:
            st.write(f"**{pos}** — {prev_lineup.get(pos,'') or '—'}")
    else:
        st.write("—")

with cB:
    st.caption(f"Current — Series {ss['current_turn']}")
    cur_edits: Dict[str, Optional[str]] = {}
    for pos in positions:
        opts = _eligible_candidates(roster_df, pos, side)
        cur_val = (ss["current_effective"].get(pos) or "").strip()
        if cur_val and cur_val not in opts:
            cur_val = ""
        pick = st.selectbox(
            f"Cur {pos}", ["(empty)"] + opts,
            index=(["(empty)"] + opts).index(cur_val) if cur_val in opts else 0,
            key=f"cur_{pos}",
        )
        cur_edits[pos] = None if pick == "(empty)" else pick
    for pos, pick in cur_edits.items():
        ss["current_effective"][pos] = (pick or "").strip()

with cC:
    st.caption(f"Next — Series {ss['current_turn']+1}")
    if not ss["next_effective"]:
        _build_next_series()
    for pos in positions:
        st.write(f"**{pos}** — {ss['next_effective'].get(pos,'') or '—'}")

colg1, colg2, colg3, colg4 = st.columns(4)

with colg1:
    if st.button("End Series (Commit Current)"):
        ss["current_effective"] = _filter_positions_map(ss["current_effective"], positions)
        _commit_series(ss["current_effective"])
        ss["series_history"].append({"turn": ss["current_turn"], "lineup": ss["current_effective"].copy()})
        ss["current_turn"] += 1
        ss["pins"].clear(); ss["excludes"].clear()
        _build_next_series()
        ss["current_effective"] = ss["next_effective"].copy()
        st.success("Series committed.")

with colg2:
    if st.button("Rebuild Current (respect fairness bias)"):
        cur, _ = assign_one_rotation(
            roster_df, side, ss.get("def_formation"), cfg,
            appearances=ss["appearances_all"], quotas=None,
            consecutive=ss["consecutive_counts"], pins=ss["pins"], excludes=ss["excludes"]
        )
        ss["current_effective"] = _filter_positions_map(cur, positions)
        st.info("Rebuilt current series.")

with colg3:
    if st.button("Undo Last Series") and ss["series_history"]:
        last = ss["series_history"].pop()
        for pos, name in last["lineup"].items():
            if name:
                ss["appearances_all"][name] = max(0, ss["appearances_all"].get(name, 0) - 1)
        ss["current_turn"] = last["turn"]
        ss["current_effective"] = _filter_positions_map(last["lineup"], positions)
        _build_next_series()
        st.warning("Undid last series.")

with colg4:
    pdf_bytes = generate_cards([_filter_positions_map(ss["current_effective"], positions),
                                _filter_positions_map(ss["next_effective"], positions)])
    st.download_button("Download Current+Next Cards (PDF)", data=pdf_bytes, file_name="cards.pdf", mime="application/pdf")

st.divider()
st.subheader("Fairness Meter & Who Hasn't Played")
appear = ss["appearances_all"]
if appear:
    vals = list(appear.values())
    delta = (max(vals) - min(vals)) if vals else 0
    st.metric("Evenness Δ (≤ 1 ideal)", delta)
    zeroes = [p.Name for p in _players_present(roster_df) if appear.get(p.Name, 0) == 0]
    st.write("Haven't played yet:", ", ".join(zeroes) if zeroes else "—")
else:
    st.write("No series committed yet.")

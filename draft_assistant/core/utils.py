from __future__ import annotations

from typing import List, Dict, Optional, Tuple
import pandas as pd

POS_NORMALIZE = {"D/ST": "DEF", "DST": "DEF", "TEAM D": "DEF", "TEAM DEF": "DEF", "DEFENSE": "DEF"}

def norm_name(s: str) -> str:
    return "".join(ch for ch in str(s).lower() if ch.isalnum())

def read_player_table(file_or_path) -> pd.DataFrame:
    if file_or_path is None:
        return None
    try:
        if hasattr(file_or_path, "read"):  # UploadedFile
            name = getattr(file_or_path, "name", "")
            if str(name).lower().endswith((".xlsx",".xls")):
                df = pd.read_excel(file_or_path)
            else:
                df = pd.read_csv(file_or_path)
        else:
            path = str(file_or_path)
            if path.lower().endswith((".xlsx",".xls")):
                df = pd.read_excel(path)
            else:
                df = pd.read_csv(path)
    except Exception:
        return None

    ren = {}
    cl = {c.lower(): c for c in df.columns}
    if "player" in cl and "PLAYER" not in df.columns: ren[cl["player"]] = "PLAYER"
    if "pos" in cl and "POS" not in df.columns: ren[cl["pos"]] = "POS"
    if "team" in cl and "TEAM" not in df.columns: ren[cl["team"]] = "TEAM"
    if "adp" in cl and "ADP" not in df.columns: ren[cl["adp"]] = "ADP"
    if "tier" in cl and "TIER" not in df.columns: ren[cl["tier"]] = "TIER"
    if "proj_pts" in cl and "PROJ_PTS" not in df.columns: ren[cl["proj_pts"]] = "PROJ_PTS"
    if "bye" in cl and "BYE" not in df.columns: ren[cl["bye"]] = "BYE"
    if ren: df = df.rename(columns=ren)

    if "PLAYER" in df.columns: df["PLAYER"] = df["PLAYER"].astype(str).str.strip()
    if "POS" in df.columns: df["POS"] = df["POS"].astype(str).str.upper().map(POS_NORMALIZE).fillna(df["POS"])
    if "TEAM" in df.columns: df["TEAM"] = df["TEAM"].astype(str).str.upper()
    return df

def snake_position(overall_pick: int, teams: int) -> Tuple[int, int, int]:
    rnd = (overall_pick - 1) // teams + 1
    pick_in_round = (overall_pick - 1) % teams + 1
    slot = pick_in_round if rnd % 2 == 1 else teams - pick_in_round + 1
    return rnd, pick_in_round, slot

def starters_from_roster_positions(roster_positions: List[str]) -> Dict[str, int]:
    starters = {"QB":0, "RB":0, "WR":0, "TE":0, "K":0, "DEF":0}
    flex = 0
    for r in roster_positions:
        R = str(r or "").upper().strip()
        R = POS_NORMALIZE.get(R, R)
        if R in starters:
            starters[R] += 1
        elif R == "FLEX":
            flex += 1
    starters["RB"] += 0.5 * flex
    starters["WR"] += 0.5 * flex
    return starters

def slot_to_display_name(slot: int, users: List[dict], rosters: Optional[List[dict]] = None) -> str:
    return f"Slot {slot}"

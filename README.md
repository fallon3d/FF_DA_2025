# Fantasy Football Draft Assistant (2025) — Streamlit + VBD

Local, privacy-friendly draft coach that mirrors your Sleeper league (or a public mock) and suggests the best pick **in real time** using a **flexible Value-Based Drafting (VBD) overlay**. No fixed “profile” — it adapts to the room, detects runs/tier cliffs, and keeps your roster construction on track.

---

## ✨ Features

- **Live Draft (Sleeper)**: Enter a League ID to mirror your real draft (order, picks, rosters, settings).
- **Mock Draft (Sleeper URL)**: Practice against live public mocks; suggestions update as the room drafts.
- **Player Board**: Searchable, tiered board with filters (POS/team/bye/injury/ADP delta).
- **Suggestions (Top 5–8)**: Ranked by composite score: VBD, VONA (value vs. next available), roster needs, tier cliff proximity, ADP value, and injury risk.
- **Run & Tier Alerts**: Warns when a position run is happening or a tier is about to evaporate.
- **Local-first**: CSV stays on your machine. No subscriptions, no accounts.

---

## 🚀 Quick Start

```bash
# From repo root (FF_DA_2025/)
python -m venv .venv

# Windows PowerShell
. .venv/Scripts/Activate.ps1
# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
streamlit run draft_assistant/app.py

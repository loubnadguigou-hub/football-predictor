"""
merge_players.py
Merges players_database_full.csv (663 players, all 20 teams, no stats)
into players_database.csv (your original file, 6 teams, has real stats),
normalizing team names to match your schedule CSV, and filling
placeholder stats for new players using position-based medians
from your existing real data (not zeros — avoids unfairly tanking them).
"""

import pandas as pd

# ---- 1. Team name mapping: API name -> your schedule CSV's exact spelling ----
TEAM_NAME_MAP = {
    "Arsenal FC": "Arsenal",
    "Aston Villa FC": "Aston Villa",
    "AFC Bournemouth": "AFC Bournemouth",
    "Brentford FC": "Brentford",
    "Brighton & Hove Albion FC": "Brighton & Hove Albion",
    "Chelsea FC": "Chelsea",
    "Crystal Palace FC": "Crystal Palace",
    "Everton FC": "Everton",
    "Fulham FC": "Fulham",
    "Coventry City FC": "Coventry City",
    "Hull City AFC": "Hull City",
    "Liverpool FC": "Liverpool",
    "Manchester City FC": "Manchester City",
    "Manchester United FC": "Manchester United",
    "Newcastle United FC": "Newcastle United",
    "Nottingham Forest FC": "Nottingham Forest",
    "Tottenham Hotspur FC": "Tottenham Hotspur",
    "Leeds United FC": "Leeds United",
    "Sunderland AFC": "Sunderland",
    "Ipswich Town FC": "Ipswich Town",
}

STAT_COLS = ["xg90", "xsot90", "goals", "xg", "minutes"]

# ---- 2. Load both files ----
# Your file DOES have a header row: team,player,pos,xg90,xsot90,goals,xg,minutes
original = pd.read_csv("data/players_database.csv")
new_players = pd.read_csv("players_database_full.csv")  # header: team, player_name, position

# ---- 3. Normalize new players' team names + column names ----
new_players["team"] = new_players["team"].map(TEAM_NAME_MAP).fillna(new_players["team"])
new_players = new_players.rename(columns={"player_name": "player", "position": "pos"})

# The API returns full words (Goalkeeper/Defence/Midfield/Offence) but your
# original file uses short codes (GK/DF/MF/FW) — normalize so position
# filters elsewhere in your app (and the median lookup below) actually match.
POSITION_MAP = {
    "Goalkeeper": "GK",
    "Defence": "DF",
    "Midfield": "MF",
    "Offence": "FW",
}
new_players["pos"] = new_players["pos"].map(POSITION_MAP).fillna(new_players["pos"])

# Fix any short-name mismatches in the ORIGINAL file too (e.g. Tottenham -> Tottenham Hotspur)
SHORT_TO_FULL = {
    "Tottenham": "Tottenham Hotspur",
    "Man United": "Manchester United",
    "Man City": "Manchester City",
    "Spurs": "Tottenham Hotspur",
}
original["team"] = original["team"].replace(SHORT_TO_FULL)

# ---- 4. Build position-based placeholder stats from your REAL data ----
position_medians = original.groupby("pos")[STAT_COLS].median()
overall_median = original[STAT_COLS].median()

def get_placeholder(pos):
    if pos in position_medians.index:
        return position_medians.loc[pos]
    return overall_median  # fallback for unusual/missing position labels

placeholder_rows = new_players["pos"].apply(get_placeholder)
for col in STAT_COLS:
    new_players[col] = placeholder_rows[col].values

new_players["is_placeholder_stats"] = True
original["is_placeholder_stats"] = False

# ---- 5. Combine, drop exact duplicates (team+player), prefer REAL stats over placeholders ----
combined = pd.concat([original, new_players], ignore_index=True)
combined = combined.sort_values("is_placeholder_stats")  # real rows (False) first
combined = combined.drop_duplicates(subset=["team", "player"], keep="first")

# Reorder columns to match your original layout, with the flag column last
combined = combined[["team", "player", "pos"] + STAT_COLS + ["is_placeholder_stats"]]

# ---- 6. Save ----
combined.to_csv("data/players_database_merged.csv", index=False)

print(f"Merged file: {len(combined)} total players")
print(f"  Real stats: {(~combined['is_placeholder_stats']).sum()}")
print(f"  Placeholder stats: {combined['is_placeholder_stats'].sum()}")
print(f"  Teams covered: {combined['team'].nunique()}")
print("\nFull team list (cross-check against your schedule CSV):")
print(sorted(combined["team"].unique()))
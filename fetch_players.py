import requests
import pandas as pd

API_KEY = "a51efce2f267486eb7c3dd0ee5cbd6b3"
HEADERS = {"X-Auth-Token": API_KEY}

resp = requests.get(
    "https://api.football-data.org/v4/competitions/PL/teams",
    headers=HEADERS
)
resp.raise_for_status()  # will throw immediately if the key is invalid
data = resp.json()

rows = []
for team in data["teams"]:
    official_name = team["name"]
    for p in team.get("squad", []):
        rows.append({
            "team": official_name,
            "player_name": p["name"],
            "position": p.get("position", ""),
        })

df = pd.DataFrame(rows)
df.to_csv("players_database_full.csv", index=False)
print(f"Got {len(df)} players across {df['team'].nunique()} teams")
print(df["team"].unique())
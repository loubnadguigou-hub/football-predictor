import requests

from src.data_loader import load_schedule
from src.results_tracker import BASE_URL, SEASON_ID, ACCESS_LEVEL, LANG, _get_api_key

df = load_schedule()
df.columns = df.columns.str.lower().str.strip()
print("SCHEDULE COLUMNS:", list(df.columns))
for c in ("wk", "week", "matchweek"):
    if c in df.columns:
        print(c, "sample values:", sorted(df[c].unique())[:5], "| dtype:", df[c].dtype)

url = BASE_URL.format(access_level=ACCESS_LEVEL, lang=LANG, season_id=SEASON_ID)
print("\nENDPOINT:", url)

for rnd in (1, 2):
    try:
        r = requests.get(
            url,
            params={"api_key": _get_api_key(), "round": rnd, "limit": 10},
            timeout=15,
        )
        print(f"\n--- ROUND {rnd}: HTTP {r.status_code}")
        print(r.text[:800])
    except Exception as e:
        print(f"\n--- ROUND {rnd} ERROR:", type(e).__name__, e)
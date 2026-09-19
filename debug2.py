import os
import pickle
import traceback

from src.data_loader import load_schedule
from src.elo_dixon_coles import EloDixonColesModel
import src.results_tracker as rt

print("New parser present:", hasattr(rt, "_parse_form_standings"))

df = load_schedule()
df.columns = df.columns.str.lower().str.strip()
df = df.rename(columns={"wk": "matchweek", "week": "matchweek",
                        "home": "home_team", "away": "away_team"})

s1 = rt.fetch_form_standings(1)
print("\nROUND 1 rows read from API:", len(s1))
res1 = rt.deduce_round_results(1)
print("ROUND 1 results deduced:", len(res1), dict(list(res1.items())[:4]))

api = {rt._norm(t) for t in res1}
sched1 = df[df["matchweek"] == 1]
raw_names = set(sched1["home_team"]) | set(sched1["away_team"])
names = {rt._norm(x) for x in raw_names}
print("Schedule names NOT found in API:", sorted(names - api))
print("API names NOT found in schedule:", sorted(api - names))

path = os.path.join("data", "trained_model.pkl")
model = pickle.load(open(path, "rb")) if os.path.exists(path) else EloDixonColesModel()
print("Round-1 teams unknown to the model:", sorted(set(map(str, raw_names)) - set(model.get_teams())))

row = sched1.iloc[0]
try:
    r = model.predict(str(row["home_team"]), str(row["away_team"]))
    print("\nPredict OK:", row["home_team"], "vs", row["away_team"],
          {k: round(float(r[k]), 3) for k in ("home_win_p", "draw_p", "away_win_p")})
except Exception:
    print("\nPredict FAILED:")
    traceback.print_exc()

print("\nBACKTEST:")
try:
    out = rt.backtest_accuracy(df, model)
    print({k: v for k, v in out.items() if k != "detail"})
except Exception:
    traceback.print_exc()
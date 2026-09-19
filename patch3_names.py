TEAM_NAMES = '''"""
src/team_names.py

Maps the team names used by the schedule / results API
(e.g. "Manchester City") to the names used by the trained model
(e.g. "Man City"). Without this, the model does not recognise the team.
"""

MODEL_NAME_MAP = {
    "AFC Bournemouth": "Bournemouth",
    "Brighton & Hove Albion": "Brighton",
    "Brighton and Hove Albion": "Brighton",
    "Coventry City": "Coventry",
    "Hull City": "Hull",
    "Ipswich Town": "Ipswich",
    "Leeds United": "Leeds",
    "Leicester City": "Leicester",
    "Manchester City": "Man City",
    "Manchester United": "Man United",
    "Newcastle United": "Newcastle",
    "Nottingham Forest": "Nott'm Forest",
    "Tottenham Hotspur": "Tottenham",
    "West Ham United": "West Ham",
    "Wolverhampton Wanderers": "Wolves",
}


def to_model_team(name, model_teams):
    """Returns the name of the team as the trained model knows it."""
    teams = list(model_teams)
    if name in teams:
        return name
    mapped = MODEL_NAME_MAP.get(name)
    if mapped in teams:
        return mapped
    low = str(name).lower()
    hits = [t for t in teams if t.lower() in low]
    if len(hits) == 1:
        return hits[0]
    return name
'''

with open("src/team_names.py", "w", encoding="utf-8") as f:
    f.write(TEAM_NAMES)
print("src/team_names.py written")


def patch(path, edits):
    s = open(path, encoding="utf-8").read()
    for old, new, marker in edits:
        if marker in s:
            continue   # already patched
        if old not in s:
            raise SystemExit("Cannot patch " + path + ": text not found -> " + old)
        s = s.replace(old, new, 1)
    open(path, "w", encoding="utf-8").write(s)
    print("patched", path)


patch("app.py", [
    ("from src.results_tracker import backtest_accuracy\n",
     "from src.results_tracker import backtest_accuracy\nfrom src.team_names import to_model_team\n",
     "from src.team_names import to_model_team"),
    ("res = dixon_coles_engine.predict(home_team, away_team)",
     "res = dixon_coles_engine.predict(to_model_team(home_team, teams), to_model_team(away_team, teams))",
     "dixon_coles_engine.predict(to_model_team("),
])

patch("src/results_tracker.py", [
    ("from src.prediction_store import outcome_from_probs\n",
     "from src.prediction_store import outcome_from_probs\nfrom src.team_names import to_model_team\n",
     "from src.team_names import to_model_team"),
    ("res = _model.predict(home, away)",
     "res = _model.predict(to_model_team(home, _model.get_teams()), to_model_team(away, _model.get_teams()))",
     "_model.predict(to_model_team("),
])
print("PATCH 3 OK")
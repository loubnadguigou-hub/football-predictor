"""
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

"""
src/results_tracker.py

Tracks the running win-prediction accuracy % of the EPL Predictive Analytics Hub.

How it works:
1. Every time a user views a match prediction, we save it (matchweek, home,
   away, predicted outcome: H/D/A) to a local CSV via prediction_store.py.
2. This module calls the Sportradar "Season Form Standings" endpoint twice —
   once for round N and once for round N-1 — and diffs the win/draw/loss
   counters per team to figure out what actually happened in round N for
   each team (since the endpoint gives cumulative counts, not per-match
   results tied to an opponent).
3. We match that deduced result back to our stored predictions (we already
   know home/away from our own schedule data) and compute a running
   accuracy percentage.

IMPORTANT: never hardcode the API key here. Put it in
.streamlit/secrets.toml locally, and in the Streamlit Cloud "Secrets" panel
in production:

    [sportradar]
    api_key = "YOUR_KEY_HERE"

Then read it with st.secrets["sportradar"]["api_key"].
"""

import requests
import pandas as pd
import streamlit as st

BASE_URL = "https://api.sportradar.com/soccer/{access_level}/v4/{lang}/seasons/{season_id}/form_standings.json"

SEASON_ID = "sr:season:140756"   # Premier League 26/27
ACCESS_LEVEL = "trial"           # switch to "production" once you upgrade the key
LANG = "en"


def _get_api_key() -> str:
    """Reads the API key from Streamlit secrets. Never hardcode it in this file."""
    try:
        return st.secrets["sportradar"]["api_key"]
    except Exception:
        st.error("⚠️ Sportradar API key missing. Add it to .streamlit/secrets.toml or Streamlit Cloud Secrets.")
        st.stop()


def fetch_form_standings(round_number: int) -> pd.DataFrame:
    """
    Calls the Season Form Standings endpoint for a given round and returns
    a DataFrame with one row per team: played, win, draw, loss, points, etc.
    """
    api_key = _get_api_key()
    url = BASE_URL.format(access_level=ACCESS_LEVEL, lang=LANG, season_id=SEASON_ID)

    params = {
        "api_key": api_key,
        "round": round_number,
        "limit": 10,
    }

    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for group in data.get("season_form_standing", {}).get("groups", []):
        for fs in group.get("form_standings", []):
            if fs.get("type") != "total":
                continue
            for standing in fs.get("form_standing", []):
                comp = standing.get("competitor", {})
                rows.append({
                    "team": comp.get("name"),
                    "played": standing.get("played", 0),
                    "win": standing.get("win", 0),
                    "draw": standing.get("draw", 0),
                    "loss": standing.get("loss", 0),
                    "form": standing.get("form", ""),
                })
    return pd.DataFrame(rows)


def deduce_round_results(round_number: int) -> dict:
    """
    Compares round N standings vs round N-1 to deduce each team's result
    (W/D/L) for round N specifically. Returns {team_name: "W" | "D" | "L"}.
    """
    if round_number <= 1:
        current = fetch_form_standings(round_number)
        prev = pd.DataFrame(columns=current.columns)
    else:
        current = fetch_form_standings(round_number)
        prev = fetch_form_standings(round_number - 1)

    prev_indexed = prev.set_index("team") if not prev.empty else None
    results = {}

    for _, row in current.iterrows():
        team = row["team"]
        if prev_indexed is not None and team in prev_indexed.index:
            prev_row = prev_indexed.loc[team]
            if row["win"] > prev_row["win"]:
                results[team] = "W"
            elif row["draw"] > prev_row["draw"]:
                results[team] = "D"
            elif row["loss"] > prev_row["loss"]:
                results[team] = "L"
        else:
            # Round 1: no previous data, use current totals directly
            if row["win"] == 1:
                results[team] = "W"
            elif row["draw"] == 1:
                results[team] = "D"
            elif row["loss"] == 1:
                results[team] = "L"

    return results


def compute_accuracy(predictions_df: pd.DataFrame) -> dict:
    """
    predictions_df must have columns: matchweek, home_team, away_team, predicted_outcome (H/D/A)

    Returns {"accuracy_pct": float, "correct": int, "total": int, "detail": DataFrame}
    """
    if predictions_df.empty:
        return {"accuracy_pct": 0.0, "correct": 0, "total": 0, "detail": pd.DataFrame()}

    detail_rows = []
    rounds_needed = sorted(predictions_df["matchweek"].unique())
    round_results_cache = {r: deduce_round_results(int(r)) for r in rounds_needed}

    for _, pred in predictions_df.iterrows():
        mw = int(pred["matchweek"])
        home, away = pred["home_team"], pred["away_team"]
        round_results = round_results_cache.get(mw, {})

        home_result = round_results.get(home)
        away_result = round_results.get(away)

        if home_result is None or away_result is None:
            continue  # match not played yet, skip

        # Deduce actual outcome from the two teams' individual results
        if home_result == "W":
            actual = "H"
        elif away_result == "W":
            actual = "A"
        elif home_result == "D" and away_result == "D":
            actual = "D"
        else:
            continue  # inconsistent data, skip defensively

        correct = (pred["predicted_outcome"] == actual)
        detail_rows.append({
            "matchweek": mw, "home_team": home, "away_team": away,
            "predicted": pred["predicted_outcome"], "actual": actual, "correct": correct
        })

    detail_df = pd.DataFrame(detail_rows)
    if detail_df.empty:
        return {"accuracy_pct": 0.0, "correct": 0, "total": 0, "detail": detail_df}

    correct = int(detail_df["correct"].sum())
    total = len(detail_df)
    return {
        "accuracy_pct": round(100 * correct / total, 1),
        "correct": correct,
        "total": total,
        "detail": detail_df,
    }
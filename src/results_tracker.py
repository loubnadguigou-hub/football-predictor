"""
src/results_tracker.py

Tracks the running win-prediction accuracy % of the EPL Predictive Analytics Hub.

How it works:
1. This module calls the Sportradar "Season Form Standings" endpoint for round N
   and round N-1 and diffs the win/draw/loss counters per team to deduce what
   actually happened in round N (the endpoint gives cumulative counts).
2. backtest_accuracy() goes through EVERY matchweek already played this season,
   asks the model for its prediction for each fixture, and compares it with the
   real result -> running accuracy of the model over the season.

IMPORTANT: never hardcode the API key here. Put it in
.streamlit/secrets.toml locally, and in the Streamlit Cloud "Secrets" panel
in production:

    [sportradar]
    api_key = "YOUR_KEY_HERE"
"""

import time

import requests
import pandas as pd
import streamlit as st

from src.prediction_store import outcome_from_probs
from src.team_names import to_model_team

BASE_URL = "https://api.sportradar.com/soccer/{access_level}/v4/{lang}/seasons/{season_id}/form_standings.json"

SEASON_ID = "sr:season:140756"   # Premier League 26/27
ACCESS_LEVEL = "trial"           # switch to "production" once you upgrade the key
LANG = "en"


class ResultsAPIError(RuntimeError):
    """Raised when the results API cannot be reached or is not configured."""


# Small alias table so schedule names and API names can be matched
_ALIASES = {
    "man utd": "manchester united", "man united": "manchester united",
    "man city": "manchester city", "spurs": "tottenham hotspur",
    "tottenham": "tottenham hotspur", "wolves": "wolverhampton wanderers",
    "newcastle": "newcastle united", "nott'm forest": "nottingham forest",
    "nottm forest": "nottingham forest", "brighton": "brighton and hove albion",
    "west ham": "west ham united", "leeds": "leeds united",
    "leicester": "leicester city",
}


def _norm(name) -> str:
    """Normalises a team name so 'Man United', 'Manchester United FC' etc. match."""
    n = str(name).lower().strip().replace("&", "and").replace(".", "")
    for token in (" fc", " afc"):
        if n.endswith(token):
            n = n[: -len(token)]
    if n.startswith("afc "):
        n = n[4:]
    n = " ".join(n.split())
    return _ALIASES.get(n, n)


def _get_api_key() -> str:
    """Reads the API key from Streamlit secrets. Never hardcode it in this file."""
    try:
        return st.secrets["sportradar"]["api_key"]
    except Exception as e:
        raise ResultsAPIError(
            "Sportradar API key missing (add it to .streamlit/secrets.toml or Streamlit Cloud Secrets)."
        ) from e


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_form_standings(round_number: int) -> pd.DataFrame:
    """
    Calls the Season Form Standings endpoint for a given round and returns
    a DataFrame with one row per team: played, win, draw, loss, form.
    Cached for 1 hour so Streamlit reruns do not hit the API again (the trial
    key is rate limited).
    """
    api_key = _get_api_key()
    url = BASE_URL.format(access_level=ACCESS_LEVEL, lang=LANG, season_id=SEASON_ID)

    params = {
        "api_key": api_key,
        "round": round_number,
        "limit": 10,
    }

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()   # HTTPError (401, 403, 404, 429...) propagates
    except (requests.ConnectionError, requests.Timeout) as e:
        raise ResultsAPIError(
            "Cannot reach the Sportradar API (check internet connection, VPN or firewall)."
        ) from e

    time.sleep(1.1)   # trial keys allow ~1 request per second
    return _parse_form_standings(resp.json())


def _parse_form_standings(data: dict) -> pd.DataFrame:
    """Parses the real Sportradar structure (season_form_standings / full_time_total)."""
    blocks = data.get("season_form_standings") or data.get("season_form_standing") or []
    if isinstance(blocks, dict):
        blocks = [blocks]
    total_blocks = [b for b in blocks if b.get("type") == "full_time_total"]
    if not total_blocks:
        total_blocks = [b for b in blocks if "total" in str(b.get("type", ""))][:1] or blocks[:1]

    rows = []
    for block in total_blocks:
        for group in block.get("groups", []):
            for standing in group.get("form_standings", []):
                comp = standing.get("competitor", {})
                rows.append({
                    "team": comp.get("name"),
                    "played": standing.get("played", 0),
                    "win": standing.get("win", 0),
                    "draw": standing.get("draw", 0),
                    "loss": standing.get("loss", 0),
                    "goals_for": standing.get("goals_for", 0),
                    "goals_against": standing.get("goals_against", 0),
                    "form": standing.get("form", ""),
                })
    df = pd.DataFrame(rows)
    if not df.empty:
        df = df.drop_duplicates(subset="team", keep="first").reset_index(drop=True)
    return df


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


def _actual_outcome(home_result, away_result):
    """H/D/A from the two teams' individual results, or None if inconsistent."""
    if home_result == "W":
        return "H"
    if away_result == "W":
        return "A"
    if home_result == "D" and away_result == "D":
        return "D"
    return None


@st.cache_data(ttl=3600, show_spinner="Computing season accuracy…")
def backtest_accuracy(schedule_df: pd.DataFrame, _model) -> dict:
    """
    Season-to-date accuracy of the model.

    For every matchweek already played, predicts each fixture with the model
    and compares the predicted outcome (H/D/A) with the real one.

    schedule_df needs columns: matchweek, home_team, away_team.
    Returns {"accuracy_pct", "correct", "total", "detail": DataFrame,
             "possible_name_mismatch": [team names]}
    Network errors are NOT caught here (so they are not cached); the caller
    should wrap the call in try/except.
    """
    empty = {"accuracy_pct": 0.0, "correct": 0, "total": 0,
             "detail": pd.DataFrame(), "possible_name_mismatch": []}
    if schedule_df is None or schedule_df.empty or "matchweek" not in schedule_df.columns:
        return empty

    mw_num = pd.to_numeric(schedule_df["matchweek"], errors="coerce")
    rounds = sorted({int(x) for x in mw_num.dropna().unique()})

    detail_rows = []
    api_names = set()
    schedule_names = set()

    for rnd in rounds:
        try:
            results = deduce_round_results(rnd)
        except requests.HTTPError as e:
            # Round not available yet on the API -> stop; rate limit etc. -> re-raise
            if e.response is not None and e.response.status_code in (400, 404):
                break
            raise
        if not results:
            break   # this round has not been played yet

        results_norm = {_norm(t): r for t, r in results.items()}
        api_names.update(results_norm.keys())

        fixtures = schedule_df[mw_num == rnd]
        for _, fx in fixtures.iterrows():
            home, away = str(fx["home_team"]), str(fx["away_team"])
            schedule_names.update([_norm(home), _norm(away)])

            actual = _actual_outcome(results_norm.get(_norm(home)), results_norm.get(_norm(away)))
            if actual is None:
                continue   # not played yet (or inconsistent data)

            try:
                res = _model.predict(to_model_team(home, _model.get_teams()), to_model_team(away, _model.get_teams()))
            except Exception:
                continue   # team unknown to the model

            predicted = outcome_from_probs(res["home_win_p"], res["draw_p"], res["away_win_p"])
            detail_rows.append({
                "matchweek": rnd, "home_team": home, "away_team": away,
                "predicted": predicted, "actual": actual,
                "correct": predicted == actual,
            })

    detail_df = pd.DataFrame(detail_rows)
    mismatch = sorted(schedule_names - api_names) if api_names else []
    if detail_df.empty:
        return {**empty, "possible_name_mismatch": mismatch}

    correct = int(detail_df["correct"].sum())
    total = len(detail_df)
    return {
        "accuracy_pct": round(100 * correct / total, 1),
        "correct": correct,
        "total": total,
        "detail": detail_df,
        "possible_name_mismatch": mismatch,
    }


def compute_accuracy(predictions_df: pd.DataFrame) -> dict:
    """
    Accuracy of the predictions that were saved by save_prediction()
    (only fixtures a user actually opened). Kept for compatibility.
    predictions_df columns: matchweek, home_team, away_team, predicted_outcome (H/D/A)
    """
    if predictions_df.empty:
        return {"accuracy_pct": 0.0, "correct": 0, "total": 0, "detail": pd.DataFrame()}

    detail_rows = []
    rounds_needed = sorted(predictions_df["matchweek"].unique())
    round_results_cache = {r: deduce_round_results(int(r)) for r in rounds_needed}

    for _, pred in predictions_df.iterrows():
        mw = int(pred["matchweek"])
        home, away = pred["home_team"], pred["away_team"]
        round_results = {_norm(t): r for t, r in round_results_cache.get(mw, {}).items()}

        actual = _actual_outcome(round_results.get(_norm(home)), round_results.get(_norm(away)))
        if actual is None:
            continue

        detail_rows.append({
            "matchweek": mw, "home_team": home, "away_team": away,
            "predicted": pred["predicted_outcome"], "actual": actual,
            "correct": pred["predicted_outcome"] == actual,
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
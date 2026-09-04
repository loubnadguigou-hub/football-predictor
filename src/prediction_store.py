"""
src/prediction_store.py

Saves every prediction the model makes to a local CSV so results_tracker.py
can later compare them against real results and compute accuracy %.
"""

import os
import pandas as pd

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "predictions_log.csv")

COLUMNS = ["matchweek", "home_team", "away_team", "predicted_outcome", "timestamp"]


def _outcome_from_probs(home_win_p: float, draw_p: float, away_win_p: float) -> str:
    """Turns the model's three probabilities into a single H/D/A predicted outcome."""
    best = max(home_win_p, draw_p, away_win_p)
    if best == home_win_p:
        return "H"
    elif best == draw_p:
        return "D"
    return "A"


def save_prediction(matchweek: int, home_team: str, away_team: str,
                     home_win_p: float, draw_p: float, away_win_p: float) -> None:
    """
    Appends a prediction row if this exact fixture hasn't already been logged
    for this matchweek (avoids duplicate rows every time the user re-opens
    the same match).
    """
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)

    predicted_outcome = _outcome_from_probs(home_win_p, draw_p, away_win_p)
    new_row = {
        "matchweek": matchweek,
        "home_team": home_team,
        "away_team": away_team,
        "predicted_outcome": predicted_outcome,
        "timestamp": pd.Timestamp.now().isoformat(),
    }

    if os.path.exists(STORE_PATH):
        df = pd.read_csv(STORE_PATH)
        exists = (
            (df["matchweek"] == matchweek)
            & (df["home_team"] == home_team)
            & (df["away_team"] == away_team)
        ).any()
        if exists:
            return
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    else:
        df = pd.DataFrame([new_row])

    df.to_csv(STORE_PATH, index=False)


def load_predictions() -> pd.DataFrame:
    if not os.path.exists(STORE_PATH):
        return pd.DataFrame(columns=COLUMNS)
    return pd.read_csv(STORE_PATH)

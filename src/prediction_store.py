"""
src/prediction_store.py

Saves every prediction the model makes to a local CSV so results_tracker.py
can later compare them against real results and compute accuracy %.

It also holds outcome_from_probs(), the ONE rule used everywhere (app display,
saved predictions, season backtest) to turn the model's three probabilities
into a single H/D/A prediction.
"""

import os
import pandas as pd

STORE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "predictions_log.csv")

COLUMNS = ["matchweek", "home_team", "away_team", "predicted_outcome", "timestamp"]

# A draw is predicted only when home and away win probabilities are this close.
DRAW_MARGIN = 0.05


def outcome_from_probs(home_win_p: float, draw_p: float, away_win_p: float) -> str:
    """
    Turns the model's three probabilities into a single H/D/A predicted outcome.
    Draw only when the match is really balanced; otherwise the stronger side.
    """
    if abs(home_win_p - away_win_p) < DRAW_MARGIN:
        return "D"
    return "H" if home_win_p > away_win_p else "A"


def save_prediction(matchweek: int, home_team: str, away_team: str,
                    home_win_p: float, draw_p: float, away_win_p: float) -> None:
    """
    Appends a prediction row if this exact fixture hasn't already been logged
    for this matchweek (avoids duplicate rows every time the user re-opens
    the same match).
    """
    os.makedirs(os.path.dirname(STORE_PATH), exist_ok=True)

    predicted_outcome = outcome_from_probs(home_win_p, draw_p, away_win_p)
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
def outcome_from_probs(home_win_p, draw_p, away_win_p):
    if abs(home_win_p - away_win_p) < 0.05:
        return "D"
    return "H" if home_win_p > away_win_p else "A"

_outcome_from_probs = outcome_from_probs

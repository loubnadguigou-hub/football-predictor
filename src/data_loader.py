import os
import pandas as pd
import requests


# ─────────────────────────────────────────────
# Match data loading
# ─────────────────────────────────────────────

def load_and_preprocess_matches(filepath_or_url: str) -> pd.DataFrame:
    """
    Loads fixture data, standardizes team names and headers,
    and sorts chronologically.
    """
    df = pd.read_csv(filepath_or_url)

    # Standardize column naming conventions
    rename_dict = {
        'HomeTeam': 'home_team', 'AwayTeam': 'away_team',
        'FTHG': 'home_goals', 'FTAG': 'away_goals',
        'Date': 'date'
    }
    df = df.rename(columns=rename_dict)

    # Ensure required columns are present
    required_cols = ['home_team', 'away_team', 'home_goals', 'away_goals']
    for col in required_cols:
        if col not in df.columns:
            raise KeyError(f"Missing required column: '{col}' in match dataset.")

    # Parse dates and sort chronologically
    if 'date' in df.columns:
        df['date'] = pd.to_datetime(df['date'], dayfirst=True, errors='coerce')
        df = df.sort_values('date').reset_index(drop=True)

    # Drop records with missing scores
    df = df.dropna(subset=['home_goals', 'away_goals'])
    df['home_goals'] = df['home_goals'].astype(int)
    df['away_goals'] = df['away_goals'].astype(int)

    return df


def generate_sample_dataset() -> pd.DataFrame:
    """
    Generates a fallback sample dataset for quick local testing.
    """
    sample_fixtures = [
        {"date": "2024-08-16", "home_team": "Arsenal", "away_team": "Wolves", "home_goals": 2, "away_goals": 0},
        {"date": "2024-08-17", "home_team": "Everton", "away_team": "Brighton", "home_goals": 0, "away_goals": 3},
        {"date": "2024-08-17", "home_team": "Chelsea", "away_team": "Man City", "home_goals": 0, "away_goals": 2},
        {"date": "2024-08-24", "home_team": "Brighton", "away_team": "Man United", "home_goals": 2, "away_goals": 1},
        {"date": "2024-08-24", "home_team": "Man City", "away_team": "Ipswich", "home_goals": 4, "away_goals": 1},
        {"date": "2024-08-25", "home_team": "Liverpool", "away_team": "Brentford", "home_goals": 2, "away_goals": 0},
        {"date": "2024-08-31", "home_team": "Arsenal", "away_team": "Brighton", "home_goals": 1, "away_goals": 1},
        {"date": "2024-09-01", "home_team": "Man United", "away_team": "Liverpool", "home_goals": 0, "away_goals": 3},
        {"date": "2024-09-15", "home_team": "Tottenham", "away_team": "Arsenal", "home_goals": 0, "away_goals": 1},
        {"date": "2024-09-22", "home_team": "Man City", "away_team": "Arsenal", "home_goals": 2, "away_goals": 2},
    ]
    df = pd.DataFrame(sample_fixtures)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_data(filepath_or_url: str) -> pd.DataFrame:
    """
    Backward-compatible alias kept so existing imports (e.g. in app.py)
    keep working without changes.
    """
    return load_and_preprocess_matches(filepath_or_url)


# ─────────────────────────────────────────────
# Schedule loading
# ─────────────────────────────────────────────

def load_schedule(filepath_or_url: str = "data/epl_2026_27_schedule.csv") -> pd.DataFrame:
    """
    Loads the season fixture schedule (Matchweek, Day, Date, Time, Home, Away).
    Falls back to an empty DataFrame if the source can't be loaded, so the
    app degrades gracefully to the manual team-selector instead of crashing.

    NOTE: Update `filepath_or_url` (or pass one in) to point at your real
    schedule source — a local CSV in your repo, or a hosted URL.
    """
    try:
        df = pd.read_csv(filepath_or_url)
        return df
    except Exception as e:
        print(f"[load_schedule] Could not load schedule from '{filepath_or_url}': {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────
# Sportradar player profile
# ─────────────────────────────────────────────

def fetch_sportradar_player_profile(player_id: str, api_key: str = None) -> dict:
    """
    Fetches a player profile from the Sportradar API.

    NOTE: Set your real API key as a Streamlit secret / environment
    variable named SPORTRADAR_API_KEY. Adjust the endpoint/response
    parsing below to match your actual Sportradar package (Soccer v4, etc.)
    if this generic version doesn't match your subscription's response shape.
    """
    api_key = api_key or os.environ.get("SPORTRADAR_API_KEY")

    if not api_key:
        print("[fetch_sportradar_player_profile] No API key configured (set SPORTRADAR_API_KEY).")
        return None

    url = f"https://api.sportradar.com/soccer/trial/v4/en/players/{player_id}/profile.json"

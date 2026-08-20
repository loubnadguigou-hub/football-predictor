import pandas as pd

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
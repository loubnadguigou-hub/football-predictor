import numpy as np
import pandas as pd

# 1. Fetch 2025-2026 Premier League results
url = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
try:
    df_matches = pd.read_csv(url)
    print(f"Dataset successfully fetched: {len(df_matches)} total matches loaded.\n")
except Exception as e:
    print(f"Error fetching data: {e}")
    exit()

# 2. Backtest Engine Setup
correct_predictions = 0
total_matches = 0
brier_scores = []

for idx, match in df_matches.iterrows():
    # Skip unplayed matches or missing scorelines
    if pd.isna(match["FTHG"]) or pd.isna(match["FTAG"]):
        continue

    home_goals = match["FTHG"]
    away_goals = match["FTAG"]

    # Match outcome: H (Home), D (Draw), A (Away)
    if home_goals > away_goals:
        actual_outcome = "H"
        actual_vector = [1, 0, 0]
    elif home_goals == away_goals:
        actual_outcome = "D"
        actual_vector = [0, 1, 0]
    else:
        actual_outcome = "A"
        actual_vector = [0, 0, 1]

    # Model probability baseline (Home Win, Draw, Away Win)
    p_home, p_draw, p_away = 0.46, 0.26, 0.28
    predicted_outcome = "H" if p_home > max(p_draw, p_away) else ("A" if p_away > p_draw else "D")

    if predicted_outcome == actual_outcome:
        correct_predictions += 1

    # Brier score calibration metric
    brier = np.sum((np.array([p_home, p_draw, p_away]) - np.array(actual_vector)) ** 2)
    brier_scores.append(brier)
    total_matches += 1

# 3. Output Metrics
if total_matches > 0:
    accuracy = (correct_predictions / total_matches) * 100
    avg_brier = np.mean(brier_scores)

    print("==========================================")
    print("      2025-2026 BACKTEST SUMMARY          ")
    print("==========================================")
    print(f"Matches Evaluated:     {total_matches}")
    print(f"1X2 Accuracy Rate:     {accuracy:.2f}%")
    print(f"Average Brier Score:   {avg_brier:.4f}")
    print("==========================================")
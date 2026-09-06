"""
scripts/train_model.py

Loads historical EPL results (2024/25, 2025/26, and the 2026/27 matches
played so far), fits the Elo ratings and Dixon-Coles MLE parameters, and
saves the trained model to disk so app.py can load it instantly on startup
instead of retraining on every page load.

Run this once locally whenever you want to refresh the model with newer
results:

    python scripts/train_model.py

It writes data/trained_model.pkl — commit that file to GitHub so the
deployed Streamlit app uses the trained model too.
"""

import sys
import os
import pickle
import pandas as pd

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.elo_dixon_coles import EloDixonColesModel

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")

# List every raw results CSV you have, oldest season first.
# The order matters: Elo ratings are built up sequentially match by match,
# so older seasons must be fit before newer ones.
SOURCE_FILES = [
    "epl_2024_25.csv",   # rename E0__3_.csv to this and place in data/
    "epl_2025_26.csv",   # rename E0__2_.csv to this
    "epl_2026_27.csv",   # rename E0__1_.csv to this
]


def load_and_standardize(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.rename(columns={
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "Date": "date",
    })
    df = df.dropna(subset=["home_team", "away_team", "home_goals", "away_goals"])
    df["home_goals"] = df["home_goals"].astype(int)
    df["away_goals"] = df["away_goals"].astype(int)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce")
    df["is_home"] = True
    return df[["date", "home_team", "away_team", "home_goals", "away_goals", "is_home"]]


def main():
    frames = []
    for fname in SOURCE_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if not os.path.exists(fpath):
            print(f"⚠️  Skipping missing file: {fpath}")
            continue
        frames.append(load_and_standardize(fpath))

    if not frames:
        raise FileNotFoundError(
            "No source CSVs found in data/. Rename your uploaded files to match "
            "SOURCE_FILES and place them in the data/ folder."
        )

    all_matches = pd.concat(frames, ignore_index=True)
    all_matches = all_matches.sort_values("date").reset_index(drop=True)

    print(f"Loaded {len(all_matches)} historical matches spanning "
          f"{all_matches['date'].min().date()} to {all_matches['date'].max().date()}")

    model = EloDixonColesModel()

    print("Fitting Elo ratings...")
    model.fit_elo_ratings(all_matches)

    print("Fitting Dixon-Coles MLE parameters (this can take a minute)...")
    model.fit_mle_parameters(all_matches)

    print("\nFinal model parameters:")
    print(model.params)
    print("\nSample Elo ratings:")
    for team in sorted(model.elos, key=lambda t: -model.elos[t])[:5]:
        print(f"  {team}: {model.elos[team]:.1f}")

    out_path = os.path.join(DATA_DIR, "trained_model.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(model, f)

    print(f"\n✅ Saved trained model to {out_path}")
    print("Commit this file to GitHub so the deployed app uses it.")


if __name__ == "__main__":
    main()

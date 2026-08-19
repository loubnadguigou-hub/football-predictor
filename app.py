import numpy as np
import pandas as pd
import streamlit as st

# 1. Page Configuration
st.set_page_config(
    page_title="Global Football Analytics & Goal Engine",
    layout="wide"
)

# 2. Custom CSS Styling
st.markdown("""
    <style>
    .stApp {
        background-color: #0b0f19;
        color: #e2e8f0;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    h1 {
        color: #f8fafc !important;
        font-size: 2.1rem !important;
        font-weight: 700 !important;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 0.75rem;
    }
    h2, h3 {
        color: #cbd5e1 !important;
        font-size: 1.25rem !important;
        font-weight: 600 !important;
    }
    div[data-testid="stMetric"] {
        background-color: #111827;
        border: 1px solid #1f2937;
        border-radius: 6px;
        padding: 18px;
    }
    div[data-testid="stMetricLabel"] {
        color: #9ca3af !important;
        font-size: 0.8rem !important;
        font-weight: 500 !important;
        text-transform: uppercase;
    }
    div[data-testid="stMetricValue"] {
        color: #38bdf8 !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    button[data-baseweb="tab"] {
        font-size: 0.95rem !important;
        font-weight: 500 !important;
        color: #94a3b8 !important;
    }
    button[aria-selected="true"] {
        color: #38bdf8 !important;
        border-bottom-color: #38bdf8 !important;
    }
    .stDataFrame {
        border: 1px solid #1f2937;
        border-radius: 6px;
    }
    </style>
""", unsafe_allow_html=True)


# 3. Global Teams & Data Engine (~380 Teams Dataset)
@st.cache_data(ttl=86400)
def load_all_global_teams():
    """Loads match data from major European leagues to generate ~380 teams dynamically."""
    league_urls = [
        # Premier League, Championship, League 1, League 2
        "https://www.football-data.co.uk/mmz4281/2526/E0.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E2.csv",
        "https://www.football-data.co.uk/mmz4281/2526/E3.csv",
        # La Liga, Segunda
        "https://www.football-data.co.uk/mmz4281/2526/SP1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/SP2.csv",
        # Serie A, Serie B
        "https://www.football-data.co.uk/mmz4281/2526/I1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/I2.csv",
        # Bundesliga, Bundesliga 2
        "https://www.football-data.co.uk/mmz4281/2526/D1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/D2.csv",
        # Ligue 1, Ligue 2
        "https://www.football-data.co.uk/mmz4281/2526/F1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/F2.csv",
        # Netherlands, Portugal, Belgium, Scotland
        "https://www.football-data.co.uk/mmz4281/2526/N1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/P1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/B1.csv",
        "https://www.football-data.co.uk/mmz4281/2526/SC0.csv",
    ]

    all_teams_stats = {}

    for url in league_urls:
        try:
            df = pd.read_csv(url)
            if "HomeTeam" in df.columns and "FTHG" in df.columns:
                teams = df["HomeTeam"].dropna().unique()
                for team in teams:
                    home_m = df[df["HomeTeam"] == team]
                    away_m = df[df["AwayTeam"] == team]
                    total_games = len(home_m) + len(away_m)
                    
                    if total_games > 0:
                        goals = home_m["FTHG"].sum() + away_m["FTAG"].sum()
                        shots = (home_m["HST"].sum() if "HST" in df.columns else total_games * 4) + \
                                (away_m["AST"].sum() if "AST" in df.columns else total_games * 4)
                        
                        all_teams_stats[team] = {
                            "avg_goals": goals / total_games,
                            "avg_sot": shots / total_games
                        }
        except Exception:
            continue

    if not all_teams_stats:
        # Fallback generated team list if external data source is unreachable
        for i in range(1, 381):
            all_teams_stats[f"Team_{i}"] = {"avg_goals": 1.35, "avg_sot": 4.2}

    return all_teams_stats

teams_database = load_all_global_teams()
all_team_names = sorted(list(teams_database.keys()))


def get_squad_player_predictions(team_name):
    """Generates player-level Poisson probabilities for any given team."""
    team_info = teams_database.get(team_name, {"avg_goals": 1.35, "avg_sot": 4.2})
    team_xg = team_info["avg_goals"]
    team_sot = team_info["avg_sot"]

    # Squad positional distribution based on team attacking power
    player_slots = [
        {"name": f"{team_name} Main Striker", "pos": "FW", "xg_share": 0.38, "sot_share": 0.35},
        {"name": f"{team_name} Left Winger", "pos": "FW", "xg_share": 0.24, "sot_share": 0.25},
        {"name": f"{team_name} Right Winger", "pos": "FW", "xg_share": 0.20, "sot_share": 0.20},
        {"name": f"{team_name} Attacking Midfielder", "pos": "MF", "xg_share": 0.10, "sot_share": 0.12},
        {"name": f"{team_name} Central Midfielder", "pos": "MF", "xg_share": 0.05, "sot_share": 0.05},
        {"name": f"{team_name} Center Back", "pos": "DF", "xg_share": 0.03, "sot_share": 0.03},
    ]

    players = []
    for p in player_slots:
        xg90 = round(team_xg * p["xg_share"], 2)
        xsot90 = round(team_sot * p["sot_share"], 2)

        goal_prob = round((1 - np.exp(-xg90)) * 100, 1)
        no_goal_prob = round((np.exp(-xg90)) * 100, 1)
        zero_shots_prob = round((np.exp(-xsot90)) * 100, 1)

        players.append({
            "Player Name": p["name"],
            "Position": p["pos"],
            "xG / 90": xg90,
            "xSoT / 90": xsot90,
            "Goal Prob (%)": goal_prob,
            "No Goal Prob (%)": no_goal_prob,
            "0 Shots Target Prob (%)": zero_shots_prob
        })

    return pd.DataFrame(players)


# 4. Streamlit Application UI
st.title("Global Football Predictive Engine")
st.caption(f"Loaded Database: {len(all_team_names)} Teams Supported")

tab1, tab2 = st.tabs(["Match Prediction & Goalscorers", "Squad Goal Model"])

# TAB 1: Match Level Predictions (Any Matchup)
with tab1:
    st.header("Match Predictions & Expected Goalscorers")

    col_home, col_away = st.columns(2)
    with col_home:
        home_team = st.selectbox("Select Home Team", all_team_names, index=0)
    with col_away:
        away_team = st.selectbox("Select Away Team", all_team_names, index=1 if len(all_team_names) > 1 else 0)

    st.markdown("---")
    st.subheader("Top Predicted Goalscorers for Fixture")

    col_h_scorers, col_a_scorers = st.columns(2)

    # Home Team Goal Candidates
    with col_h_scorers:
        st.markdown(f"### {home_team}")
        df_home = get_squad_player_predictions(home_team)
        top_h = df_home.iloc[0]
        st.metric(
            label=f"Top Goal Candidate ({top_h['Player Name']})",
            value=f"{top_h['Goal Prob (%)']}%"
        )
        st.dataframe(
            df_home[["Player Name", "Position", "Goal Prob (%)"]].head(4),
            use_container_width=True,
            hide_index=True
        )

    # Away Team Goal Candidates
    with col_a_scorers:
        st.markdown(f"### {away_team}")
        df_away = get_squad_player_predictions(away_team)
        top_a = df_away.iloc[0]
        st.metric(
            label=f"Top Goal Candidate ({top_a['Player Name']})",
            value=f"{top_a['Goal Prob (%)']}%"
        )
        st.dataframe(
            df_away[["Player Name", "Position", "Goal Prob (%)"]].head(4),
            use_container_width=True,
            hide_index=True
        )

# TAB 2: Individual Squad Analysis
with tab2:
    st.header("Squad Goal Probability Analysis")

    selected_team = st.selectbox("Select Team for Squad Breakdown", all_team_names)
    df_squad = get_squad_player_predictions(selected_team)

    top_player = df_squad.iloc[0]
    st.metric(
        label=f"Highest Goal Probability ({selected_team})",
        value=f"{top_player['Player Name']} — {top_player['Goal Prob (%)']}%"
    )

    st.dataframe(
        df_squad[["Player Name", "Position", "xG / 90", "Goal Prob (%)", "No Goal Prob (%)", "0 Shots Target Prob (%)"]],
        use_container_width=True,
        hide_index=True
    )
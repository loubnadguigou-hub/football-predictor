import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from src.data_loader import load_data, load_schedule, fetch_sportradar_player_profile
from src.elo_dixon_coles import EloDixonColesModel
from src.player_model import PlayerGoalModel

# 1. Page Configuration
st.set_page_config(
    page_title="EPL Predictive Analytics Hub",
    page_icon="⚽",
    layout="wide"
)

# Custom Styling
st.markdown("""
    <style>
    .main-title {
        text-align: center;
        font-size: 34px;
        font-weight: 900;
        color: #111827;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-top: -10px;
        margin-bottom: 20px;
    }
    .metric-box {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-title {
        font-size: 11px;
        font-weight: 700;
        color: #4a5568;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-bottom: 6px;
    }
    .metric-val {
        font-size: 28px;
        font-weight: 800;
        color: #1a202c;
    }
    </style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("<div class='main-title'>⚽ Premier League Predictive Analytics Hub</div>", unsafe_allow_html=True)

# Load Models & Schedule Data
dixon_coles_engine = EloDixonColesModel()
player_engine = PlayerGoalModel()
teams = dixon_coles_engine.get_teams()
df_schedule = load_schedule()

# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------
st.sidebar.markdown("### ⚙️ 1. Match Data Source")
data_source = st.sidebar.radio("Match Data:", ["Premier League Direct (URL)", "Sample Dataset"], index=0)
st.sidebar.success("✓ Loaded 380 matches")
st.sidebar.markdown("---")

st.sidebar.markdown("### 🧑‍💻 2. Player Data Source")
st.sidebar.caption("Upload Real Player Stats CSV (FBref/Understat)")
uploaded_file = st.sidebar.file_uploader("Upload", type=["csv"], help="200MB per file • CSV")

# ---------------------------------------------------------
# MAIN NAVIGATION TABS
# ---------------------------------------------------------
tab1, tab2 = st.tabs(["📊 Match Outcome Engine", "🎯 Player Goal Scoring Model"])

# ---------------------------------------------------------
# TAB 1: MATCH OUTCOME ENGINE
# ---------------------------------------------------------
with tab1:
    st.markdown("### 🗓️ 2026/27 Full Season Schedule & Match Selector")

    if not df_schedule.empty:
        # Standardize column names
        df_sched = df_schedule.copy()
        df_sched.columns = df_sched.columns.str.lower().str.strip()

        rename_map = {
            "wk": "matchweek", "week": "matchweek",
            "home": "home_team", "away": "away_team",
            "time": "kickoff_time_uk"
        }
        df_sched = df_sched.rename(columns=rename_map)

        # Matchweek Filter
        available_mws = sorted(df_sched["matchweek"].unique()) if "matchweek" in df_sched.columns else [1]
        selected_mw = st.selectbox("📅 Select Matchweek:", available_mws, index=0)

        # Filter schedule by Matchweek
        mw_df = df_sched[df_sched["matchweek"] == selected_mw].copy().reset_index(drop=True)

        # ── Clickable match list ──
        if "selected_match" not in st.session_state:
            st.session_state.selected_match = None

        st.markdown("#### 🖱️ Click a fixture to select it:")

        for idx, row in mw_df.iterrows():
            home = str(row.get("home_team", ""))
            away = str(row.get("away_team", ""))
            date_str = str(row.get("date", ""))
            day_str = str(row.get("day", ""))[:3]
            time_str = str(row.get("kickoff_time_uk", ""))

            is_selected = (
                st.session_state.selected_match is not None
                and st.session_state.selected_match.get("matchweek") == selected_mw
                and st.session_state.selected_match.get("home") == home
                and st.session_state.selected_match.get("away") == away
            )

            with st.container(border=True):
                col_info, col_btn = st.columns([4, 1])
                with col_info:
                    st.markdown(f"**{home} vs {away}**")
                    st.caption(f"{day_str} {date_str} · {time_str} UK")
                with col_btn:
                    btn_label = "✅ Selected" if is_selected else "Select"
                    if st.button(
                        btn_label,
                        key=f"btn_{selected_mw}_{idx}",
                        type="primary" if is_selected else "secondary",
                        use_container_width=True
                    ):
                        st.session_state.selected_match = {
                            "matchweek": selected_mw, "home": home, "away": away
                        }
                        st.rerun()

        # Default to first match in the list if nothing has been picked yet
        if st.session_state.selected_match is None and len(mw_df) > 0:
            first_row = mw_df.iloc[0]
            st.session_state.selected_match = {
                "matchweek": selected_mw,
                "home": str(first_row.get("home_team", "")),
                "away": str(first_row.get("away_team", "")),
            }

        sel = st.session_state.selected_match
        home_team = sel["home"]
        away_team = sel["away"]

        # Look up full fixture details (works even if selection is from a different matchweek)
        match_lookup = df_sched[
            (df_sched["matchweek"] == sel["matchweek"]) &
            (df_sched["home_team"] == home_team) &
            (df_sched["away_team"] == away_team)
        ]
        if not match_lookup.empty:
            match_row = match_lookup.iloc[0]
            match_date = match_row.get("date", "")
            match_time = match_row.get("kickoff_time_uk", "")
            match_day = match_row.get("day", "")
            st.info(f"📌 **Selected Fixture (MW{sel['matchweek']}):** {home_team} vs {away_team} — **Date:** {match_day}, {match_date} at **{match_time} UK**")

    else:
        col_home, col_away = st.columns(2)
        with col_home:
            home_team = st.selectbox("Home Team", teams, index=0)
        with col_away:
            away_team = st.selectbox("Away Team", teams, index=1)

    st.markdown("<br>", unsafe_allow_html=True)

    if home_team == away_team:
        st.warning("Please select two different teams.")
    else:
        # Compute match forecasts
        res = dixon_coles_engine.predict(home_team, away_team)

        # Top Banner & Projected Scorelines
        col_banner, col_top_scores = st.columns([2, 1])

        with col_banner:
            st.markdown(f"""
            <div style="background-color: #00a86b; padding: 22px; border-radius: 12px; color: white; margin-bottom: 15px;">
                <div style="font-size: 13px; font-weight: 800; letter-spacing: 1px; margin-bottom: 5px;">🎯 MODEL FORECASTED EXACT SCORE</div>
                <div style="font-size: 36px; font-weight: 800; margin-bottom: 5px;">{home_team} {res['best_home_goals']} - {res['best_away_goals']} {away_team}</div>
                <div style="font-size: 16px; opacity: 0.95;">Highest probability scoreline at <b>{res['best_prob']:.1%}</b> chance</div>
            </div>
            """, unsafe_allow_html=True)

        with col_top_scores:
            st.markdown("##### 📈 Top Projected Scorelines")
            score_probs = []
            matrix_norm = res['matrix'] / np.sum(res['matrix'])
            for h_g in range(4):
                for a_g in range(5):
                    score_probs.append((f"{home_team} {h_g} - {a_g} {away_team}", matrix_norm[h_g, a_g]))
            score_probs.sort(key=lambda x: x[1], reverse=True)

            top_df = pd.DataFrame(score_probs[:4], columns=["Scoreline", "Probability"])
            top_df["Probability"] = top_df["Probability"].apply(lambda x: f"{x:.1%}")
            st.dataframe(top_df, use_container_width=True, hide_index=True)

        # Metric Cards
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">{home_team.upper()} ELO</div>
                <div class="metric-val">{res['home_elo']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">{away_team.upper()} ELO</div>
                <div class="metric-val">{res['away_elo']}</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">{home_team.upper()} EXP. GOALS (λ)</div>
                <div class="metric-val">{res['lambda_home']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        with col4:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-title">{away_team.upper()} EXP. GOALS (λ)</div>
                <div class="metric-val">{res['lambda_away']:.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Outcome Probabilities
        p1, p2, p3 = st.columns(3)
        with p1:
            st.markdown(f"<div style='font-size: 16px; font-weight: 600; color: #2d3748;'>🏠 {home_team} Win</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 42px; font-weight: 800; color: #1a202c; margin-top: -5px;'>{res['home_win_p']:.1%}</div>", unsafe_allow_html=True)

        with p2:
            st.markdown("<div style='font-size: 16px; font-weight: 600; color: #2d3748;'>🤝 Draw</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 42px; font-weight: 800; color: #1a202c; margin-top: -5px;'>{res['draw_p']:.1%}</div>", unsafe_allow_html=True)

        with p3:
            st.markdown(f"<div style='font-size: 16px; font-weight: 600; color: #2d3748;'>🚀 {away_team} Win</div>", unsafe_allow_html=True)
            st.markdown(f"<div style='font-size: 42px; font-weight: 800; color: #1a202c; margin-top: -5px;'>{res['away_win_p']:.1%}</div>", unsafe_allow_html=True)

        st.markdown("---")

        # Probability Matrix
        st.markdown("### 🧮 Scoreline Probability Matrix")

        max_home, max_away = 4, 7
        matrix_normalized = res['matrix'] / np.sum(res['matrix'])
        matrix_normalized = matrix_normalized[:max_home, :max_away]
        text_matrix = [[f"{matrix_normalized[i, j]:.1%}" for j in range(max_away)] for i in range(max_home)]

        fig = px.imshow(
            matrix_normalized,
            labels=dict(x="", y="", color="Probability"),
            x=[f"{away_team} {j}" for j in range(max_away)],
            y=[f"{home_team} {i}" for i in range(max_home)],
            color_continuous_scale="Purples",
            aspect="auto"
        )

        fig.update_traces(text=text_matrix, texttemplate="%{text}", textfont=dict(size=12, color="white"))
        fig.update_layout(coloraxis_showscale=False, margin=dict(l=10, r=10, t=10, b=10), height=260, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')

        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------
# TAB 2: PLAYER GOAL SCORING MODEL
# ---------------------------------------------------------
with tab2:
    st.header(f"🎯 Player Goalscorer Predictions: {home_team} vs. {away_team}")
    st.write("Identifies specific players from both competing squads who are most likely to score based on individual xG/90 and match expectations.")

    df_home_players = player_engine.predict_player_probabilities(home_team)
    df_away_players = player_engine.predict_player_probabilities(away_team)

    # Standardize column headers
    column_mapping = {
        "Player": "Player Name",
        "Pos": "Position",
        "xG": "xG/90",
        "Goal_Prob": "Anytime Goal Prob (%)",
        "Prob": "Anytime Goal Prob (%)"
    }
    df_home_players = df_home_players.rename(columns=column_mapping)
    df_away_players = df_away_players.rename(columns=column_mapping)

    target_cols = ["Player Name", "Position", "xG/90", "Anytime Goal Prob (%)"]
    display_home_cols = [col for col in target_cols if col in df_home_players.columns] or df_home_players.columns.tolist()
    display_away_cols = [col for col in target_cols if col in df_away_players.columns] or df_away_players.columns.tolist()

    col_h_players, col_a_players = st.columns(2)

    with col_h_players:
        st.subheader(f"🏠 {home_team} Top Goal Scorers")
        st.dataframe(df_home_players[display_home_cols], use_container_width=True, hide_index=True)

        if "Player Name" in df_home_players.columns and "Anytime Goal Prob (%)" in df_home_players.columns:
            fig_h = px.bar(
                df_home_players,
                x="Player Name",
                y="Anytime Goal Prob (%)",
                color="Position" if "Position" in df_home_players.columns else None,
                title=f"Goal Probability (%) by Player - {home_team}",
                color_discrete_sequence=['#00a86b', '#3182ce', '#dd6b20']
            )
            fig_h.update_layout(template="plotly_white", height=320)
            st.plotly_chart(fig_h, use_container_width=True)

    with col_a_players:
        st.subheader(f"🚀 {away_team} Top Goal Scorers")
        st.dataframe(df_away_players[display_away_cols], use_container_width=True, hide_index=True)

        if "Player Name" in df_away_players.columns and "Anytime Goal Prob (%)" in df_away_players.columns:
            fig_a = px.bar(
                df_away_players,
                x="Player Name",
                y="Anytime Goal Prob (%)",
                color="Position" if "Position" in df_away_players.columns else None,
                title=f"Goal Probability (%) by Player - {away_team}",
                color_discrete_sequence=['#805ad5', '#e53e3e', '#319795']
            )
            fig_a.update_layout(template="plotly_white", height=320)
            st.plotly_chart(fig_a, use_container_width=True)

    st.markdown("---")
    st.subheader("🔍 Sportradar Marketplace Live Player Data")
    player_id_input = st.text_input("Enter Sportradar Player ID:", value="sr:player:159665")

    if st.button("Fetch Live Player Data"):
        with st.spinner("Fetching data from Sportradar Marketplace..."):
            player_info = fetch_sportradar_player_profile(player_id_input)
            if player_info:
                st.success("Data successfully loaded!")
                col_a, col_b, col_c = st.columns(3)
                col_a.metric("Player Name", player_info.get("Player Name", "N/A"))
                col_b.metric("Position", player_info.get("Position", "N/A"))
                col_c.metric("Current Club", player_info.get("Current Club", "N/A"))
                st.table(pd.DataFrame([player_info]))
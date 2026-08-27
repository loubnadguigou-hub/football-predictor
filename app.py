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
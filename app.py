import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import numpy as np
import json
import yfinance as yf
from streamlit_autorefresh import st_autorefresh
import time
import os
import mlflow
from mlflow.tracking import MlflowClient

# =============================================================================
# 1. CONFIGURATION & STYLE
# =============================================================================
st.set_page_config(page_title="AlphaEdge Dashboard", layout="wide", initial_sidebar_state="expanded")
st_autorefresh(interval=900000, key="datarefresh")

BASE_DIR = Path(__file__).resolve().parent
HF_REPO_ID = os.getenv("HF_REPO_ID", "soradata/alphaedge-data")

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .kpi-container { background-color: #151922; padding: 15px; border-radius: 8px; border: 1px solid #262730; }
    .kpi-value { font-size: 24px; font-weight: 700; color: #ffffff; }
    .kpi-delta-pos { color: #00cc96; font-size: 20px; font-weight: 600; }
    .kpi-delta-neg { color: #ef553b; font-size: 20px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# 2. CHARGEMENT DYNAMIQUE
# =============================================================================
@st.cache_data(ttl=600, show_spinner=False)
def load_all_data(market: str):
    base_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/data/{market}"
    df_hist, df_signals, df_rebalance = pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    
    files = {
        "portfolio_history.parquet": "df_hist",
        "latest_signals.parquet": "df_signals",
        "rebalance_history.parquet": "df_rebalance"
    }
    
    for filename, _ in files.items():
        try:
            df = pd.read_parquet(f"{base_url}/{filename}")
            if filename == "portfolio_history.parquet":
                df.index = pd.to_datetime(df.index, errors="coerce")
                df_hist = df[df.index.notna()].sort_index(ascending=True)
            elif filename == "latest_signals.parquet":
                df_signals = df
            elif filename == "rebalance_history.parquet":
                df.index = pd.to_datetime(df.index, errors="coerce")
                df_rebalance = df.sort_index(ascending=False)
        except Exception:
            continue
    return df_hist, df_signals, df_rebalance

# =============================================================================
# 3. SIDEBAR
# =============================================================================
st.sidebar.title("AlphaEdge")
market_options = ["CAC40", "BRVM"] 
selected_market = st.sidebar.selectbox("Sélectionner le marché", market_options)

df_hist, df_signals, df_rebalance = load_all_data(selected_market)

if st.sidebar.button("🔄 Force Sync Pipeline"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GITHUB-Source_Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SORADATA/CAC40-Quantitative-Analysis-Predictive-Asset-Allocation)"
)
st.sidebar.markdown("---")

page = st.sidebar.radio("Navigation", ["Dashboard", "Daily Signals", "Model Details"])

# =============================================================================
# 4. PAGES
# =============================================================================
if page == "Dashboard":
    st.title(f"Portfolio Overview - {selected_market}")
    # [Logique de ton Dashboard existant]

elif page == "Daily Signals":
    st.title(f"📡 Daily Trading Signals - {selected_market}")
    # [Logique de ton Daily Signals existant]

elif page == "Model Details":
    st.title("⚙️ Model Configuration")
    st.subheader(f"Performance Live - {selected_market} (MLflow)")
    
    client = MlflowClient()
    try:
        model_name = f"AlphaEdge_Ensemble_{selected_market}"
        model_version = client.get_latest_versions(model_name, stages=["Champion"])[0]
        metrics = client.get_run(model_version.run_id).data.metrics
        
        cols = st.columns(4)
        for i, (name, val) in enumerate(metrics.items()):
            cols[i % 4].metric(name.upper(), f"{val:.4f}")
    except Exception:
        st.info("⏳ Aucune métrique disponible : vérifiez le stage 'Champion' sur MLflow.")

st.sidebar.markdown("---")
st.sidebar.caption("⚠️ **Disclaimer:** Not financial advice.")

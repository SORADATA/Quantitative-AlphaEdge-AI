import os
import json
import time
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh
import yfinance as yf

import mlflow
from mlflow.tracking import MlflowClient
from mlflow.exceptions import MlflowException

# =============================================================================
# CONFIGURATION & STYLE
# =============================================================================
st.set_page_config(
    page_title="AlphaEdge Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st_autorefresh(interval=900000, key="datarefresh")

BASE_DIR = Path(__file__).resolve().parent
HF_REPO_ID = os.getenv("HF_REPO_ID", "soradata/alphaedge-data")
MLFLOW_TRACKING_URI = "https://soradata-alphaedge-registry.hf.space"
HF_TOKEN = os.getenv("HF_TOKEN")
MLFLOW_ENABLED = bool(HF_TOKEN)

if MLFLOW_ENABLED:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "SORADATA"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = HF_TOKEN
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

MODEL_DIR = BASE_DIR / "models"

@st.cache_data(ttl=1800, show_spinner=False)
def _discover_markets():
    """
    Decouvre dynamiquement les marches disponibles en interrogeant le repo
    Hugging Face distant (dataset HF_REPO_ID), sous le prefixe data/<MARKET>/.
    Fallback sur un scan local (data/processed) si l'API HF echoue, puis sur
    une liste par defaut en dernier recours.
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        files = api.list_repo_files(repo_id=HF_REPO_ID, repo_type="dataset", token=HF_TOKEN)
        markets = sorted({
            f.split("/")[1] for f in files
            if f.startswith("data/") and len(f.split("/")) > 2
        })
        if markets:
            return markets
    except Exception:
        pass

    local_dir = BASE_DIR / "data" / "processed"
    if local_dir.exists():
        found = sorted([p.name for p in local_dir.iterdir() if p.is_dir()])
        if found:
            return found

    return ["CAC40", "BRVM"]

MARKET_OPTIONS = _discover_markets()

st.markdown("""
<style>
    .main { background-color: #0E1117; }
    .kpi-container {
        background-color: #151922;
        padding: 15px;
        border-radius: 8px;
        border: 1px solid #262730;
        text-align: left;
    }
    .kpi-minimal { text-align: left; padding: 10px 0; }
    .kpi-label { font-size: 12px; color: #8b92a5; margin-bottom: 4px; }
    .kpi-value { font-size: 24px; font-weight: 700; color: #ffffff; }
    .kpi-delta-pos { color: #00cc96; font-size: 20px; font-weight: 600; }
    .kpi-delta-neg { color: #ef553b; font-size: 20px; font-weight: 600; }
    .disclaimer-box {
        background-color: #1E1E1E;
        color: #888888;
        padding: 20px;
        border-radius: 5px;
        font-size: 12px;
        border-top: 1px solid #333;
        margin-top: 50px;
        text-align: center;
    }
    .disclaimer-title {
        color: #EF553B;
        font-weight: bold;
        margin-bottom: 10px;
        font-size: 14px;
        text-transform: uppercase;
    }
    .mlflow-badge {
        display: inline-block;
        padding: 3px 10px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-bottom: 10px;
    }
    .badge-champion { background-color: rgba(0, 204, 150, 0.15); color: #00CC96; }
    .badge-fallback { background-color: rgba(255, 193, 7, 0.15); color: #FFC107; }
</style>
""", unsafe_allow_html=True)


# =============================================================================
# CHARGEMENT DES DONNEES DEPUIS HUGGING FACE
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def load_all_data(market: str):
    """
    Charge portfolio_history / latest_signals / rebalance_history pour un marche.
    Version durcie : normalisation du nom de marche, chargement defensif
    (safe_load) qui isole chaque fichier pour qu'une erreur sur l'un ne
    bloque pas les autres, tout en conservant les controles metier
    (colonnes manquantes, fraicheur des donnees) de la version precedente.
    """
    clean_market = str(market).strip()
    base_url = f"https://huggingface.co/datasets/{HF_REPO_ID}/resolve/main/data/{clean_market}"

    errors = []

    def safe_load(url, key):
        try:
            df = pd.read_parquet(url)
            if key in ["hist", "rebal"]:
                df.index = pd.to_datetime(df.index, errors="coerce")
                df = df[df.index.notna()].sort_index(ascending=(key == "hist"))
            return df
        except Exception as e:
            errors.append(f"Error loading {key}: {e}")
            return pd.DataFrame()

    df_hist = safe_load(f"{base_url}/portfolio_history.parquet", "hist")
    df_signals = safe_load(f"{base_url}/latest_signals.parquet", "signals")
    df_rebalance = safe_load(f"{base_url}/rebalance_history.parquet", "rebal")

    if not df_hist.empty:
        days_old = (datetime.now() - df_hist.index[-1]).days
        if days_old > 7:
            errors.append(f"Portfolio data is {days_old} days old")

    if not df_signals.empty:
        missing = [c for c in ["Ticker", "Signal"] if c not in df_signals.columns]
        if missing:
            errors.append(f"Missing columns in signals: {missing}")

    return df_hist, df_signals, df_rebalance, errors


# =============================================================================
# FONCTIONS UTILITAIRES
# =============================================================================

def display_kpi_card(label, value, is_percent=True, color_code=False, prefix="", suffix="", minimal=False):
    if pd.isna(value) or np.isinf(value):
        html_val = '<span class="kpi-value">N/A</span>'
    else:
        if is_percent:
            formatted_val = f"{prefix}{value:.1%}{suffix}"
        elif isinstance(value, (int, np.integer)) or suffix:
            formatted_val = f"{prefix}{int(value)}{suffix}"
        else:
            formatted_val = f"{prefix}{value:.2f}{suffix}"
        if color_code:
            color_class = "kpi-delta-pos" if value >= 0 else "kpi-delta-neg"
            arrow = "▲" if value >= 0 else "▼"
            html_val = f'<span class="{color_class}">{arrow} {formatted_val}</span>'
        else:
            html_val = f'<span class="kpi-value">{formatted_val}</span>'
    css_class = "kpi-minimal" if minimal else "kpi-container"
    st.markdown(f"""
    <div class="{css_class}">
        <div class="kpi-label">{label}</div>
        {html_val}
    </div>
    """, unsafe_allow_html=True)


def calculate_metrics(df):
    if df.empty or len(df) < 2:
        return 0, 0, 0, 0, 0
    try:
        total_ret = (df["Strategy"].iloc[-1] / df["Strategy"].iloc[0]) - 1
        bench_ret = (df["Benchmark"].iloc[-1] / df["Benchmark"].iloc[0]) - 1
        alpha = total_ret - bench_ret
        strategy_returns = df["Strategy"].pct_change().dropna()
        sharpe = (strategy_returns.mean() / strategy_returns.std()) * np.sqrt(252) if strategy_returns.std() != 0 else 0
        cum_ret = (1 + strategy_returns).cumprod()
        running_max = cum_ret.cummax()
        dd_series = (cum_ret - running_max) / running_max
        max_dd = dd_series.min()
        recovery_time = _compute_recovery_time(dd_series)
        return total_ret, alpha, sharpe, max_dd, recovery_time
    except Exception:
        return 0, 0, 0, 0, 0


def _compute_recovery_time(dd_series: pd.Series) -> int:
    """
    Calcule le nombre de jours écoulés depuis le point bas du dernier
    drawdown significatif jusqu'au retour au plus haut (0). Si la
    stratégie n'a pas encore récupéré, retourne le nombre de jours
    depuis le point bas jusqu'à aujourd'hui (recovery en cours).
    """
    if dd_series.empty:
        return 0
    trough_idx = dd_series.idxmin()
    post_trough = dd_series.loc[trough_idx:]
    recovered = post_trough[post_trough >= -0.0001]
    if len(recovered) > 1:
        recovery_date = recovered.index[1]
        return (recovery_date - trough_idx).days
    return (dd_series.index[-1] - trough_idx).days


def calculate_period_return(df, days=None, ytd=False, daily=False):
    if df.empty or "Strategy" not in df.columns or len(df) < 2:
        return 0.0
    try:
        if daily:
            return (df["Strategy"].iloc[-1] / df["Strategy"].iloc[-2]) - 1
        last_price, last_date = df["Strategy"].iloc[-1], df.index[-1]
        if ytd:
            target_date = datetime(last_date.year, 1, 1)
        elif days:
            target_date = last_date - timedelta(days=days)
        else:
            target_date = df.index[0]
        if target_date < df.index[0]:
            start_price = df["Strategy"].iloc[0]
        else:
            start_price = df["Strategy"].iloc[df.index.get_indexer([target_date], method="nearest")[0]]
        return ((last_price / start_price) - 1) if start_price != 0 else 0.0
    except Exception:
        return 0.0


def _trim_flat_start(df: pd.DataFrame, tol: float = 1e-6) -> pd.DataFrame:
    """
    Supprime la periode initiale "plate" (valeurs constantes, generalement un
    placeholder egal a la valeur de base) presente au debut de certains
    historiques, avant le premier rebalancement reel de la strategie.
    On garde tout l'historique si aucune periode plate n'est detectee.
    """
    if df.empty or "Strategy" not in df.columns or len(df) < 3:
        return df
    changes = df["Strategy"].diff().abs() > tol
    if "Benchmark" in df.columns:
        changes = changes | (df["Benchmark"].diff().abs() > tol)
    first_move = changes[changes].index
    if len(first_move) == 0:
        return df
    start_idx = df.index.get_loc(first_move[0])
    start_idx = max(0, start_idx - 1)
    return df.iloc[start_idx:]


@st.cache_data(ttl=3600)
def get_live_ticker_data(ticker, period="1y"):
    for _ in range(3):
        try:
            df = yf.download(ticker, period=period, progress=False, timeout=10)
            if not df.empty:
                df.columns = df.columns.get_level_values(0) if isinstance(df.columns, pd.MultiIndex) else df.columns
                df.columns = df.columns.str.lower()
                if "adj close" not in df.columns and "close" in df.columns:
                    df["adj close"] = df["close"]
                return df
            time.sleep(2)
        except Exception:
            time.sleep(2)
    return pd.DataFrame()


# =============================================================================
# FONCTIONS UTILITAIRES - MLFLOW (aligne sur train.py : alias "champion")
# =============================================================================

@st.cache_data(ttl=600, show_spinner=False)
def get_champion_metrics(market: str):
    """
    Recupere les metriques du modele 'champion' pour un marche donne.
    1) Essaie MLflow via l'alias 'champion'.
    2) Si MLflow est indisponible ou qu'aucun alias 'champion' n'existe encore,
       on retombe sur le model_card.json sauvegarde localement par train.py.
    """
    result = {
        "source": None,
        "metrics": {},
        "version": None,
        "run_id": None,
        "promoted": None,
        "error": None,
    }
    registered_model_name = f"AlphaEdge_Ensemble_{market}"

    if MLFLOW_ENABLED:
        try:
            client = MlflowClient()
            mv = client.get_model_version_by_alias(registered_model_name, "champion")
            run = client.get_run(mv.run_id)
            result["source"] = "mlflow"
            result["metrics"] = run.data.metrics
            result["version"] = mv.version
            result["run_id"] = mv.run_id
            result["promoted"] = True
            return result
        except MlflowException as e:
            result["error"] = f"MLflow: {e}"
        except Exception as e:
            result["error"] = f"MLflow: {e}"

    card_path = MODEL_DIR / market / "model_card.json"
    if card_path.exists():
        try:
            with open(card_path, "r") as f:
                card = json.load(f)
            result["source"] = "local"
            metrics = {}
            for k, v in card.get("metrics_ml", {}).items():
                metrics[f"ml_{k}"] = v
            for k, v in card.get("metrics_fin", {}).items():
                metrics[f"fin_{k}"] = v
            result["metrics"] = metrics
            result["promoted"] = card.get("mlflow", {}).get("promoted", False)
            result["run_id"] = card.get("mlflow", {}).get("run_id")
        except Exception as e:
            result["error"] = (result["error"] + " | " if result["error"] else "") + f"model_card.json: {e}"

    return result


# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.title("AlphaEdge")
st.sidebar.caption("Quantitative Asset Allocation")

selected_market = st.sidebar.selectbox("Marché", MARKET_OPTIONS, index=0)

with st.spinner(f"Loading {selected_market} data..."):
    df_hist, df_signals, df_rebalance, load_errors = load_all_data(selected_market)

if st.sidebar.button("Force Sync Pipeline"):
    st.cache_data.clear()
    st.rerun()

st.sidebar.markdown(
    "[![GitHub](https://img.shields.io/badge/GITHUB-Source_Code-181717?style=for-the-badge&logo=github&logoColor=white)](https://github.com/SORADATA/CAC40-Quantitative-Analysis-Predictive-Asset-Allocation)"
)

st.sidebar.markdown("---")
page = st.sidebar.radio("Navigation", [
    "Dashboard",
    "Daily Signals",
    "Data Explorer",
    "Model Details",
    "Rebalance History"
])
st.sidebar.markdown("---")

if not df_hist.empty:
    last_dt = df_hist.index[-1]
    days_old = (datetime.now() - last_dt).days

    if days_old <= 1:
        status_icon, status_text = "🟢", "System Online"
    elif days_old <= 3:
        status_icon, status_text = "🟡", "Data Slightly Old"
    else:
        status_icon, status_text = "🔴", "Data Outdated"
    st.sidebar.info(f"Last Update: {last_dt.date()}")
    st.sidebar.markdown(f"{status_icon} {status_text}")
else:
    st.sidebar.error("No Data Available")

st.sidebar.markdown("---")

ticker_val_path = BASE_DIR / "config" / selected_market / "ticker_validation.json"
if not ticker_val_path.exists():
    ticker_val_path = BASE_DIR / "ticker_validation.json"
if ticker_val_path.exists():
    with st.sidebar.expander("Ticker Health", expanded=False):
        try:
            with open(ticker_val_path, "r") as f:
                validation = json.load(f)
            alerts = validation.get("alerts", {})
            if alerts.get("delisted"):
                st.error(f"{len(alerts['delisted'])} delistés")
            if alerts.get("stale"):
                st.warning(f"{len(alerts['stale'])} obsolètes")
            if alerts.get("warnings"):
                st.info(f"{len(alerts['warnings'])} warnings")
            st.metric("Valid Tickers", validation.get("valid_tickers", 0))
        except Exception:
            st.caption("Validation data unavailable")

st.sidebar.markdown("---")

if load_errors:
    with st.sidebar.expander("Data Issues", expanded=False):
        for err in load_errors:
            st.warning(err, icon="⚠️")

if not MLFLOW_ENABLED:
    st.sidebar.caption("HF_TOKEN absent : métriques modèle en mode local uniquement.")

st.sidebar.caption("Disclaimer: Not financial advice.")


# =============================================================================
# PAGE 1 : DASHBOARD
# =============================================================================

if page == "Dashboard":
    st.title(f"Portfolio Overview - {selected_market}")
    if not df_hist.empty:
        tot_ret, alpha, sharpe, max_dd, recovery_days = calculate_metrics(df_hist)
        c1, c2, c3, c4, c5 = st.columns(5)
        with c1:
            display_kpi_card("Total Return", tot_ret, color_code=True)
        with c2:
            display_kpi_card("Alpha vs Bench", alpha, color_code=True)
        with c3:
            display_kpi_card("Sharpe Ratio", sharpe, is_percent=False)
        with c4:
            display_kpi_card("Max Drawdown", max_dd, color_code=True)
        with c5:
            display_kpi_card("Recovery Time", recovery_days, is_percent=False, suffix=" j", minimal=False)

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("Period Performance")
        k1, k2, k3, k4, k5 = st.columns(5)
        with k1:
            display_kpi_card("YTD", calculate_period_return(df_hist, ytd=True), color_code=True, minimal=True)
        with k2:
            display_kpi_card("6 Months", calculate_period_return(df_hist, days=180), color_code=True, minimal=True)
        with k3:
            display_kpi_card("3 Months", calculate_period_return(df_hist, days=90), color_code=True, minimal=True)
        with k4:
            display_kpi_card("1 Month", calculate_period_return(df_hist, days=30), color_code=True, minimal=True)
        with k5:
            display_kpi_card("Daily Return", calculate_period_return(df_hist, daily=True), color_code=True, minimal=True)

        st.markdown("---")
        col_title, col_filter = st.columns([2, 1])
        with col_title:
            st.subheader("Strategy vs Benchmark")
        with col_filter:
            p_sel = st.radio("Zoom:", ["1M", "3M", "6M", "YTD", "1Y", "ALL"], index=5, horizontal=True, label_visibility="collapsed")

        df_c = _trim_flat_start(df_hist)
        end = df_c.index[-1]
        if p_sel == "1M":
            start = end - timedelta(days=30)
        elif p_sel == "3M":
            start = end - timedelta(days=90)
        elif p_sel == "6M":
            start = end - timedelta(days=180)
        elif p_sel == "YTD":
            start = datetime(end.year, 1, 1)
        elif p_sel == "1Y":
            start = end - timedelta(days=365)
        else:
            start = df_c.index[0]
        if start < df_c.index[0]:
            start = df_c.index[0]
        df_c = df_c[df_c.index >= pd.Timestamp(start)]
        df_base = df_c.apply(lambda x: x / x.iloc[0] * 100)

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_base.index, y=df_base["Benchmark"],
            mode="lines", name="Benchmark",
            line=dict(color="#8b92a5", width=1.3, dash="dot"),
            hovertemplate="Benchmark: %{y:.1f}<extra></extra>"
        ))
        fig.add_trace(go.Scatter(
            x=df_base.index, y=df_base["Strategy"],
            mode="lines", name="Strategy",
            line=dict(color="#2ED9A0", width=2),
            hovertemplate="Strategy: %{y:.1f}<extra></extra>"
        ))
        fig.update_layout(
            template="plotly_white",
            plot_bgcolor="#11151c",
            paper_bgcolor="#11151c",
            font=dict(color="#c9ced6", size=12),
            margin=dict(l=0, r=0, t=30, b=0),
            height=400,
            hovermode="x unified",
            legend=dict(
                orientation="h", y=1.12, x=1, xanchor="right",
                bgcolor="rgba(0,0,0,0)", title=None,
                font=dict(size=12)
            ),
            xaxis=dict(showgrid=False, showline=True, linecolor="#2a2f3a", ticks="outside", tickcolor="#2a2f3a"),
            yaxis=dict(
                title="Indexed Value (Base 100)", title_font=dict(size=11, color="#8b92a5"),
                showgrid=True, gridcolor="rgba(255,255,255,0.06)", zeroline=False,
                showline=False
            )
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("---")
        c_risk, c_pie = st.columns([3, 2])
        with c_risk:
            st.subheader("Risk Analysis")
            s_ret = df_c["Strategy"].pct_change().dropna()
            cum = (1 + s_ret).cumprod()
            dd = (cum - cum.cummax()) / cum.cummax()
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(
                x=dd.index, y=dd, fill="tozeroy", mode="lines",
                line=dict(color="#EF553B", width=1.5),
                name="Drawdown", fillcolor="rgba(239, 85, 59, 0.3)"
            ))
            fig_dd.update_layout(
                template="plotly_dark", margin=dict(l=0, r=0, t=10, b=0),
                height=320, yaxis_tickformat=".1%", yaxis_title="Drawdown"
            )
            st.plotly_chart(fig_dd, use_container_width=True)
        with c_pie:
            st.subheader("Current Allocation")
            if not df_signals.empty and "Allocation" in df_signals.columns:
                active = df_signals[df_signals["Allocation"] > 0.001].copy()
                cash = max(0, 1.0 - active["Allocation"].sum())
                if cash > 0.001:
                    final = pd.concat([active, pd.DataFrame([{"Ticker": "CASH", "Allocation": cash}])], ignore_index=True)
                else:
                    final = active
                fig_p = px.pie(final, values="Allocation", names="Ticker", hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism)
                fig_p.update_traces(textposition="outside", textinfo="percent+label")
                fig_p.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=0, b=0), showlegend=False, height=370)
                st.plotly_chart(fig_p, use_container_width=True)
            else:
                st.info("Waiting for signals...")
    else:
        st.warning(f"No data available for {selected_market} from Hugging Face.")


# =============================================================================
# PAGE 2 : DAILY SIGNALS
# =============================================================================

elif page == "Daily Signals":
    st.title(f"Daily Trading Signals - {selected_market}")
    if not df_signals.empty:
        d = df_signals.copy()
        if "Allocation" in d.columns:
            d = d.sort_values("Allocation", ascending=False)
        col_filter1, col_filter2 = st.columns([1, 3])
        with col_filter1:
            filter_opt = st.selectbox("Filter", ["All Signals", "BUY Only", "NEUTRAL Only"])
        if filter_opt == "BUY Only":
            d = d[d["Signal"] == "BUY"]
        elif filter_opt == "NEUTRAL Only":
            d = d[d["Signal"] == "NEUTRAL"]
        st.dataframe(
            d, use_container_width=True, height=600, hide_index=True,
            column_config={
                "Allocation": st.column_config.ProgressColumn("Weight", format="%.2f", min_value=0, max_value=1),
                "Proba_Hausse": st.column_config.NumberColumn("Probability Up", format="%.1f%%")
            }
        )
        st.markdown("---")
        col_s1, col_s2, col_s3 = st.columns(3)
        with col_s1:
            st.metric("Total Tickers", len(df_signals))
        with col_s2:
            n_buy = len(df_signals[df_signals["Signal"] == "BUY"]) if "Signal" in df_signals.columns else 0
            st.metric("BUY Signals", n_buy)
        with col_s3:
            alloc_total = df_signals["Allocation"].sum() if "Allocation" in df_signals.columns else 0
            st.metric("Total Allocated", f"{alloc_total:.1%}")
    else:
        st.info(f"No signals available for {selected_market}.")


# =============================================================================
# PAGE 3 : DATA EXPLORER
# =============================================================================

elif page == "Data Explorer":
    st.title("Market Data Explorer")
    default_tickers = ["AI.PA", "AIR.PA", "BNP.PA", "MC.PA", "OR.PA", "TTE.PA"]
    tickers = df_signals["Ticker"].unique().tolist() if not df_signals.empty and "Ticker" in df_signals.columns else default_tickers
    col_sel1, col_sel2 = st.columns([1, 3])
    with col_sel1:
        selected_ticker = st.selectbox("Select Asset", tickers, index=0)
    with col_sel2:
        period_exp = st.selectbox("Timeframe", ["1 Month", "3 Months", "6 Months", "1 Year", "5 Years"], index=2)
    yf_period_map = {"1 Month": "1mo", "3 Months": "3mo", "6 Months": "6mo", "1 Year": "1y", "5 Years": "5y"}
    with st.spinner(f"Downloading {selected_ticker}..."):
        df_asset = get_live_ticker_data(selected_ticker, period=yf_period_map[period_exp])
    if not df_asset.empty and len(df_asset) > 1:
        try:
            last_close = df_asset["adj close"].iloc[-1]
            prev_close = df_asset["adj close"].iloc[-2]
            daily_var = (last_close / prev_close) - 1
            total_ret_period = (last_close / df_asset["adj close"].iloc[0]) - 1
            volatility = df_asset["adj close"].pct_change().dropna().std() * np.sqrt(252)
        except Exception:
            last_close = daily_var = total_ret_period = volatility = 0
        m1, m2, m3, m4 = st.columns(4)
        with m1:
            display_kpi_card("Last Price", last_close, is_percent=False, prefix="€ ")
        with m2:
            display_kpi_card("Daily Change", daily_var, color_code=True)
        with m3:
            display_kpi_card(f"Return ({period_exp})", total_ret_period, color_code=True)
        with m4:
            display_kpi_card("Ann. Volatility", volatility, is_percent=True)
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
        fig.add_trace(go.Candlestick(
            x=df_asset.index, open=df_asset["open"], high=df_asset["high"],
            low=df_asset["low"], close=df_asset["close"], name="OHLC"
        ), row=1, col=1)
        colors = ["#00CC96" if r >= 0 else "#EF553B" for r in df_asset["adj close"].pct_change().fillna(0)]
        fig.add_trace(go.Bar(x=df_asset.index, y=df_asset["volume"], name="Volume", marker_color=colors), row=2, col=1)
        fig.update_layout(template="plotly_dark", xaxis_rangeslider_visible=False, height=550, margin=dict(l=0, r=0, t=30, b=0), showlegend=False)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data for {selected_ticker}")


# =============================================================================
# PAGE 4 : MODEL DETAILS (MLflow - alias "champion", fallback model_card.json)
# =============================================================================

elif page == "Model Details":
    st.title(f"Model Configuration - {selected_market}")
    tab1, tab2 = st.tabs(["Performance", "Clusters"])

    with tab1:
        st.markdown("""
        ### Hybrid Strategy Components
        **1. XGBoost + LightGBM + Ridge -> LogisticRegression**: Predicts 1-month upside probability
        **2. K-Means**: Market regime detection (RSI-based)
        **3. Markowitz**: Portfolio optimization (Max Sharpe)
        """)
        st.markdown("---")

        champ = get_champion_metrics(selected_market)

        if champ["source"] == "mlflow":
            st.markdown(
                '<span class="mlflow-badge badge-champion">MLflow - Champion v'
                f'{champ["version"]}</span>', unsafe_allow_html=True
            )
        elif champ["source"] == "local":
            status = "promu champion" if champ.get("promoted") else "dernier run (non promu - seuils non atteints)"
            st.markdown(
                f'<span class="mlflow-badge badge-fallback">Fallback local - {status}</span>',
                unsafe_allow_html=True
            )
        else:
            st.info("Aucune métrique disponible pour ce marché (ni MLflow, ni model_card.json local).")

        if champ["metrics"]:
            st.markdown("### Model Performance")
            metric_items = list(champ["metrics"].items())
            cols = st.columns(4)
            for i, (name, val) in enumerate(metric_items):
                try:
                    formatted = f"{val:.4f}"
                except (TypeError, ValueError):
                    formatted = str(val)
                cols[i % 4].metric(name.replace("_", " ").title(), formatted)

        if champ["error"]:
            with st.expander("Détails techniques (debug)"):
                st.caption(champ["error"])

        st.markdown("---")
        st.markdown("### Feature Importance")
        st.caption("Explicabilité du modèle XGBoost — variables les plus influentes sur la prédiction.")
        feat_imp = None
        if isinstance(champ.get("metrics"), dict):
            feat_imp = champ["metrics"].get("feature_importance")
        if feat_imp:
            fi_df = pd.DataFrame(list(feat_imp.items()), columns=["Feature", "Importance"]).sort_values("Importance", ascending=True)
        else:
            demo_features = ["momentum_3m", "rsi_14", "volume_lag1", "pe_ratio", "book_to_market",
                              "volatility_60d", "ff_smb", "ff_hml", "earnings_yield", "beta_1y"]
            rng = np.random.default_rng(42)
            demo_vals = np.sort(rng.uniform(0.02, 0.22, size=len(demo_features)))
            fi_df = pd.DataFrame({"Feature": demo_features, "Importance": demo_vals})
            st.caption("Données factices affichées à titre d'exemple — en attente de l'extraction réelle depuis le modèle XGBoost.")
        fig_fi = px.bar(
            fi_df, x="Importance", y="Feature", orientation="h",
            color="Importance", color_continuous_scale=px.colors.sequential.Tealgrn
        )
        fig_fi.update_layout(template="plotly_dark", height=400, margin=dict(l=0, r=0, t=10, b=0), showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig_fi, use_container_width=True)

    with tab2:
        st.subheader("Cluster Analysis")
        st.markdown("Segmentation based on RSI to identify momentum vs reversal regimes.")
        if not df_signals.empty and "RSI" in df_signals.columns and "Return_3M" in df_signals.columns:
            fig = px.scatter(
                df_signals, x="RSI", y="Return_3M", color="Cluster", hover_name="Ticker",
                color_continuous_scale=px.colors.sequential.Viridis,
                labels={"RSI": "RSI (20)", "Return_3M": "3-Month Momentum"}
            )
            fig.update_layout(template="plotly_dark", height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No cluster data available")


# =============================================================================
# PAGE 5 : REBALANCE HISTORY
# =============================================================================

elif page == "Rebalance History":
    st.title(f"Monthly Rebalancing History - {selected_market}")
    if not df_rebalance.empty:
        st.markdown("""
        This page shows the **monthly rebalancing decisions** made by the strategy.
        Each month, the model selects new assets and recalculates optimal weights.
        """)
        col_r1, col_r2, col_r3 = st.columns(3)
        with col_r1:
            st.metric("Total Rebalances", len(df_rebalance))
        with col_r2:
            avg_stocks = df_rebalance["N_Stocks"].mean() if "N_Stocks" in df_rebalance.columns else 0
            st.metric("Avg. Stocks/Month", f"{avg_stocks:.1f}")
        with col_r3:
            last_rebal = df_rebalance.index[0].date() if len(df_rebalance) > 0 else "N/A"
            st.metric("Last Rebalance", str(last_rebal))
        st.markdown("---")
        if "N_Stocks" in df_rebalance.columns:
            st.subheader("Portfolio Size Evolution")
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=df_rebalance.index, y=df_rebalance["N_Stocks"],
                mode="lines+markers", name="N Stocks",
                line=dict(color="#00CC96", width=2), marker=dict(size=6)
            ))
            fig.update_layout(template="plotly_dark", height=350, yaxis_title="Number of Stocks", xaxis_title="Date")
            st.plotly_chart(fig, use_container_width=True)
        st.subheader("Detailed Rebalancing Log")
        st.dataframe(df_rebalance, use_container_width=True, height=400)
    else:
        st.info(f"No rebalance history available for {selected_market}.")


# =============================================================================
# DISCLAIMER FOOTER
# =============================================================================

st.markdown("---")
st.markdown("""
<div class="disclaimer-box">
    <div class="disclaimer-title">AVIS DE NON-RESPONSABILITÉ</div>
    <p>Les informations présentées sur ce tableau de bord sont fournies <strong>à titre informatif et éducatif uniquement</strong>. Elles ne constituent en aucun cas un conseil en investissement.</p>
    <p><strong>Risques :</strong> Tout investissement comporte des risques. Les performances passées ne garantissent pas les résultats futurs.</p>
    <p><strong>Responsabilité :</strong> Consultez un conseiller financier agréé avant toute décision d'investissement.</p>
</div>
""", unsafe_allow_html=True)

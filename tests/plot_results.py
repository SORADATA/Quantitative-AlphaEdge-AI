import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path

def plot_performance(market_name="CAC40"):
    print(f"📊 Génération du graphique pour {market_name}...")
    
    # 1. Chargement des données
    base_dir = Path(f"data/processed/{market_name}")
    hist_path = base_dir / 'portfolio_history.parquet'
    
    if not hist_path.exists():
        print(f"❌ Erreur : Fichier introuvable ({hist_path}). Lance d'abord le daily_run.py.")
        return

    df = pd.read_parquet(hist_path)
    
    # 2. Rebase à 100 pour une comparaison parfaite
    df['Strategy'] = (df['Strategy'] / df['Strategy'].iloc[0]) * 100
    df['Benchmark'] = (df['Benchmark'] / df['Benchmark'].iloc[0]) * 100
    
    # 3. Calcul des Drawdowns (pertes maximales depuis le sommet)
    df['Strat_Peak'] = df['Strategy'].cummax()
    df['Strat_DD'] = (df['Strategy'] - df['Strat_Peak']) / df['Strat_Peak'] * 100
    
    df['Bench_Peak'] = df['Benchmark'].cummax()
    df['Bench_DD'] = (df['Benchmark'] - df['Bench_Peak']) / df['Bench_Peak'] * 100

    # 4. Création de la figure (2 sous-graphiques)
    fig = make_subplots(
        rows=2, cols=1, 
        shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.7, 0.3],
        subplot_titles=("Croissance du Capital (Base 100)", "Drawdown (%)")
    )

    # --- LIGNE 1 : PERFORMANCE ---
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Strategy'], name='AlphaEdge (Stratégie)', 
                   line=dict(color='#00d2ff', width=2)),
        row=1, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Benchmark'], name='Benchmark (^FCHI)', 
                   line=dict(color='#888888', width=2)),
        row=1, col=1
    )

    # --- LIGNE 2 : DRAWDOWN ---
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Strat_DD'], name='DD Stratégie', 
                   fill='tozeroy', line=dict(color='#ff4b4b', width=1)),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=df.index, y=df['Bench_DD'], name='DD Benchmark', 
                   fill='tozeroy', line=dict(color='#888888', width=1, dash='dot')),
        row=2, col=1
    )

    # 5. Esthétique et Layout
    final_strat = round(df['Strategy'].iloc[-1], 2)
    final_bench = round(df['Benchmark'].iloc[-1], 2)
    
    fig.update_layout(
        title=f"<b>AlphaEdge vs {market_name}</b> | Final: Stratégie {final_strat}€ vs Bench {final_bench}€",
        template="plotly_dark",
        hovermode="x unified",
        height=800,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
        margin=dict(l=40, r=40, t=60, b=40)
    )
    
    fig.update_yaxes(title_text="Valeur (€)", row=1, col=1)
    fig.update_yaxes(title_text="Drawdown (%)", row=2, col=1)

    # AFFICHER LE GRAPHIQUE DIRECTEMENT
    fig.show()

if __name__ == "__main__":
    plot_performance("CAC40")
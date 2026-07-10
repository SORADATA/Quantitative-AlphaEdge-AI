"""
UI Utils
=========
Composants d'affichage réutilisables pour le dashboard Streamlit.

Fonctions :
  - display_kpi_card() : carte KPI stylisée (valeur, delta coloré, prefix/suffix)
"""

import numpy as np
import pandas as pd
import streamlit as st


def display_kpi_card(
    label: str,
    value,
    is_percent: bool = True,
    color_code: bool = False,
    prefix: str = "",
    suffix: str = "",
    minimal: bool = False,
):
    """
    Affiche une carte KPI stylisée.

    Parameters
    ----------
    label : str — libellé affiché au-dessus de la valeur
    value : float | int — valeur à afficher (N/A si NaN/inf)
    is_percent : bool — formate en pourcentage (%.1%)
    color_code : bool — colore + ajoute une flèche selon le signe
    prefix / suffix : str — texte ajouté avant/après la valeur
    minimal : bool — variante sans le cadre "container" (fond + bordure)
    """
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


def load_css():
    """
    Injecte le style CSS custom du dashboard (cartes KPI, disclaimer,
    badges MLflow). À appeler une seule fois, juste après les imports.
    """
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
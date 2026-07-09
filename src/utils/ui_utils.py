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
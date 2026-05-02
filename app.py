"""
Application principale – Interface Fouille de Données
Lancer avec : streamlit run app.py
"""

import html

import numpy as np
import pandas as pd
import streamlit as st

from modules.preprocessing import (
    load_dataset,
    get_basic_info,
    descriptive_stats,
    handle_missing,
    remove_duplicates,
    normalize,
    plot_boxplots,
    plot_scatter,
    plot_correlation_heatmap,
    plot_histograms,
)
from modules.clustering import (
    run_kmeans,
    run_kmedoids,
    run_dbscan,
    run_agnes,
    run_diana,
    compute_silhouette,
    elbow_curve,
    plot_elbow,
    plot_clusters_2d,
    plot_dendrogram,
)
from modules.classification import (
    MODELS,
    prepare_classification_data,
    train_model,
    evaluate_model,
    compare_models,
    plot_confusion_matrix,
    plot_metrics_bar,
    plot_feature_importance,
    plot_decision_tree,
    plot_model_comparison,
)

# ---------------------------------------------------------------------------
# Configuration de la page
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="InzLab",
    page_icon="⬡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# CSS — Glassmorphism dark theme
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, .stApp, [data-testid="stAppViewContainer"], [data-testid="stSidebar"],
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] *,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
button, input, textarea, select {
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stIconMaterial"] {
    font-family: "Material Symbols Rounded", "Material Symbols Outlined", "Material Icons" !important;
    font-weight: 400 !important;
    font-style: normal !important;
    font-size: 1.1rem !important;
    line-height: 1 !important;
    letter-spacing: normal !important;
    text-transform: none !important;
    white-space: nowrap !important;
    direction: ltr !important;
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    -webkit-font-smoothing: antialiased !important;
}

/* ── Background ── */
.stApp {
    background: linear-gradient(135deg, #0f0c29 0%, #302b63 55%, #24243e 100%) !important;
    min-height: 100vh;
}
[data-testid="stMain"], .block-container, section.main { background: transparent !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background:
        radial-gradient(circle at top left, rgba(92, 117, 255, 0.20), transparent 32%),
        linear-gradient(180deg, rgba(8, 10, 28, 0.98), rgba(13, 16, 39, 0.94)) !important;
    backdrop-filter: blur(24px) !important;
    -webkit-backdrop-filter: blur(24px) !important;
    border-right: 1px solid rgba(255,255,255,0.06) !important;
    box-shadow: inset -1px 0 0 rgba(255,255,255,0.04) !important;
}
[data-testid="stSidebar"] * { color: rgba(255,255,255,0.85) !important; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
    padding-top: 0.4rem !important;
}

/* ── Top bar ── */
header[data-testid="stHeader"] {
    background: rgba(8,6,25,0.90) !important;
    backdrop-filter: blur(20px) !important;
    border-bottom: 1px solid rgba(255,255,255,0.05) !important;
}

/* ── Glass cards (expanders) ── */
[data-testid="stExpander"] {
    background: linear-gradient(180deg, rgba(41,45,98,0.72), rgba(35,38,86,0.56)) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    border: 1px solid rgba(167,139,250,0.18) !important;
    border-radius: 18px !important;
    box-shadow: 0 18px 38px rgba(7,8,24,0.32), inset 0 1px 0 rgba(255,255,255,0.04) !important;
    margin-bottom: 1.1rem !important;
    overflow: hidden !important;
}
[data-testid="stExpander"] details {
    border: none !important;
}
[data-testid="stExpander"] details > summary {
    padding: 0.9rem 1.1rem !important;
    background: linear-gradient(180deg, rgba(12,14,36,0.88), rgba(24,27,61,0.72)) !important;
    border-bottom: 1px solid rgba(167,139,250,0.14) !important;
}
[data-testid="stExpander"] details > summary > span {
    display: flex !important;
    align-items: center !important;
    gap: 0.65rem !important;
}
[data-testid="stExpander"] details > summary:hover {
    background: linear-gradient(180deg, rgba(18,20,48,0.94), rgba(30,33,72,0.78)) !important;
}
[data-testid="stExpander"] details > summary p {
    color: rgba(255,255,255,0.92) !important;
    font-weight: 600 !important;
    font-size: 0.97rem !important;
    margin: 0 !important;
}
[data-testid="stExpander"] details > summary [data-testid="stIconMaterial"] {
    color: rgba(255,255,255,0.78) !important;
}
[data-testid="stExpanderDetails"] {
    background: transparent !important;
}

/* ── Text ── */
p, li { color: rgba(255,255,255,0.80) !important; }
h1,h2,h3,h4 { color: #ffffff !important; }
label { color: rgba(255,255,255,0.80) !important; }
small, [data-testid="stCaptionContainer"] p { color: rgba(255,255,255,0.45) !important; }

/* ── Metrics ── */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.06) !important;
    border: 1px solid rgba(255,255,255,0.10) !important;
    border-radius: 14px !important;
    padding: 1rem 1.1rem !important;
    box-shadow: 0 4px 16px rgba(0,0,0,0.30) !important;
}
[data-testid="stMetricValue"] {
    color: #c4b5fd !important;
    font-weight: 700 !important;
    font-size: 1.75rem !important;
}
[data-testid="stMetricLabel"] {
    color: rgba(255,255,255,0.50) !important;
    font-size: 0.80rem !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.45rem 1.4rem !important;
    font-weight: 600 !important;
    box-shadow: 0 4px 14px rgba(102,126,234,0.40) !important;
    transition: transform 0.20s, box-shadow 0.20s !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 7px 22px rgba(102,126,234,0.58) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:disabled {
    background: rgba(255,255,255,0.08) !important;
    color: rgba(255,255,255,0.28) !important;
    box-shadow: none !important;
    transform: none !important;
}

/* ── Inputs ── */
[data-testid="stSelectbox"] > div > div,
[data-testid="stMultiSelect"] > div > div {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
}
[data-testid="stTextInput"] input,
[data-testid="stNumberInput"] input {
    background: rgba(255,255,255,0.07) !important;
    border: 1px solid rgba(255,255,255,0.14) !important;
    border-radius: 10px !important;
    color: rgba(255,255,255,0.88) !important;
}

/* ── Tabs ── */
[data-testid="stTabs"] [role="tablist"] {
    background: rgba(255,255,255,0.04) !important;
    border-radius: 12px !important;
    padding: 5px !important;
    gap: 8px !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    align-items: center !important;
}
[data-testid="stTabs"] button[role="tab"] {
    color: rgba(255,255,255,0.55) !important;
    border-radius: 9px !important;
    font-weight: 500 !important;
    background: transparent !important;
    border: none !important;
    padding: 0.7rem 1rem !important;
    min-height: 44px !important;
}
[data-testid="stTabs"] button[role="tab"] p {
    margin: 0 !important;
    white-space: nowrap !important;
}
[data-testid="stTabs"] button[role="tab"][aria-selected="true"] {
    background: rgba(118,75,162,0.30) !important;
    color: #fff !important;
    font-weight: 600 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] section {
    background: rgba(255,255,255,0.025) !important;
    border: 2px dashed rgba(167,139,250,0.40) !important;
    border-radius: 14px !important;
    transition: border-color 0.2s !important;
}
[data-testid="stFileUploader"] section:hover {
    border-color: rgba(167,139,250,0.72) !important;
}
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] span,
[data-testid="stFileUploader"] [data-testid="stFileUploaderDropzoneInstructions"] div {
    color: rgba(255,255,255,0.55) !important;
}
[data-testid="stFileUploader"] [data-testid="stWidgetLabel"] p {
    color: rgba(255,255,255,0.90) !important;
    font-weight: 600 !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] > div {
    border-radius: 12px !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    overflow: hidden !important;
}

/* ── Comparison table ── */
.comparison-table-wrap {
    margin: 0.55rem 0 1rem 0;
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    overflow-x: auto;
    background: rgba(12, 14, 36, 0.38);
}
.comparison-table {
    width: 100%;
    border-collapse: separate;
    border-spacing: 0;
}
.comparison-table thead th {
    padding: 0.9rem 1rem;
    text-align: left;
    font-size: 0.82rem;
    font-weight: 600;
    letter-spacing: 0.02em;
    color: rgba(255,255,255,0.62);
    background: rgba(255,255,255,0.05);
    border-bottom: 1px solid rgba(255,255,255,0.08);
}
.comparison-table tbody td {
    padding: 0.9rem 1rem;
    color: rgba(255,255,255,0.84);
    border-bottom: 1px solid rgba(255,255,255,0.06);
}
.comparison-table tbody tr:last-child td {
    border-bottom: none;
}
.comparison-table .model-cell {
    font-weight: 600;
    color: rgba(255,255,255,0.96);
}
.comparison-table .metric-cell {
    font-variant-numeric: tabular-nums;
}
.comparison-table .is-best {
    background: linear-gradient(135deg, rgba(167,139,250,0.26), rgba(102,126,234,0.18));
    color: #ffffff;
    font-weight: 700;
    box-shadow: inset 0 0 0 1px rgba(167,139,250,0.22);
}

/* ── Alerts ── */
div[data-testid="stAlert"] { border-radius: 12px !important; }

/* ── File uploader browse button (prevent double text) ── */
[data-testid="stFileUploaderDropzone"] button,
[data-testid="stFileUploader"] button {
    background: rgba(167,139,250,0.12) !important;
    color: rgba(255,255,255,0.85) !important;
    border: 1px solid rgba(167,139,250,0.35) !important;
    border-radius: 8px !important;
    box-shadow: none !important;
    transform: none !important;
    font-size: 0.85rem !important;
}
[data-testid="stFileUploaderDropzone"] button:hover,
[data-testid="stFileUploader"] button:hover {
    border-color: rgba(167,139,250,0.65) !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background: rgba(167,139,250,0.35); border-radius: 99px; }
::-webkit-scrollbar-thumb:hover { background: rgba(167,139,250,0.62); }

/* ── Page headers ── */
.page-header { padding: 0.25rem 0 1.25rem 0; }
.page-title-row { display:flex; align-items:center; gap:0.55rem; margin-bottom:0.25rem; }
h1.page-title {
    font-size: 2.1rem !important;
    font-weight: 700 !important;
    margin: 0 !important;
    background: linear-gradient(130deg, #ffffff 20%, #a78bfa 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.15 !important;
}
.page-subtitle {
    color: rgba(255,255,255,0.42) !important;
    font-size: 0.875rem !important;
    margin: 0 0 0.4rem 0 !important;
    font-weight: 400 !important;
    -webkit-text-fill-color: rgba(255,255,255,0.42) !important;
}
.title-bar {
    height: 2px; width: 52px;
    background: linear-gradient(90deg, #a78bfa, #667eea);
    border-radius: 99px;
}

/* ── Sidebar brand & navigation ── */
.sidebar-hero {
    padding: 0.45rem 0 1rem 0;
}
.brand-eyebrow {
    font-size: 0.68rem !important;
    text-transform: uppercase;
    letter-spacing: 0.16em;
    color: rgba(161, 174, 209, 0.60) !important;
    -webkit-text-fill-color: rgba(161, 174, 209, 0.60) !important;
    margin-bottom: 0.45rem;
}
.brand-wordmark {
    font-size: 1.7rem !important;
    font-weight: 700 !important;
    letter-spacing: -0.04em;
    line-height: 1 !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}
.brand-tag {
    margin-top: 0.55rem;
    max-width: 15rem;
    font-size: 0.76rem !important;
    line-height: 1.5;
    color: rgba(203, 213, 225, 0.56) !important;
    -webkit-text-fill-color: rgba(203, 213, 225, 0.56) !important;
}
.brand-divider {
    height: 1px;
    margin: 0.95rem 0 0.95rem 0;
    background: linear-gradient(90deg, rgba(96, 165, 250, 0.55), rgba(167, 139, 250, 0.22), transparent);
}
.sidebar-section-title {
    font-size: 0.72rem !important;
    text-transform: uppercase;
    letter-spacing: 0.14em;
    color: rgba(161, 174, 209, 0.58) !important;
    -webkit-text-fill-color: rgba(161, 174, 209, 0.58) !important;
}
.sidebar-section-subtitle {
    margin-top: 0.25rem;
    font-size: 0.78rem !important;
    color: rgba(255,255,255,0.42) !important;
    -webkit-text-fill-color: rgba(255,255,255,0.42) !important;
}
.sidebar-chip-row {
    display: flex;
    gap: 0.45rem;
    flex-wrap: wrap;
    margin-top: 0.85rem;
}
.sidebar-chip {
    display: inline-flex;
    align-items: center;
    padding: 0.34rem 0.58rem;
    border-radius: 999px;
    font-size: 0.70rem !important;
    color: rgba(226,232,240,0.78) !important;
    -webkit-text-fill-color: rgba(226,232,240,0.78) !important;
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(148,163,184,0.14);
}
[data-testid="stSidebar"] [role="radiogroup"] {
    display: grid !important;
    gap: 0.55rem !important;
    margin-top: 0.45rem !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] {
    display: block !important;
    margin: 0 !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] input {
    position: absolute !important;
    opacity: 0 !important;
    pointer-events: none !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] input + div {
    width: 100% !important;
    padding: 0.88rem 0.95rem !important;
    border-radius: 16px !important;
    border: 1px solid rgba(148,163,184,0.12) !important;
    background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.02)) !important;
    transition: transform 0.18s ease, border-color 0.18s ease, background 0.18s ease, box-shadow 0.18s ease !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"]:hover input + div {
    transform: translateX(2px) !important;
    border-color: rgba(129, 140, 248, 0.26) !important;
    background: linear-gradient(180deg, rgba(255,255,255,0.07), rgba(255,255,255,0.03)) !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] input:checked + div {
    border-color: rgba(125, 211, 252, 0.28) !important;
    background: linear-gradient(135deg, rgba(70, 92, 214, 0.44), rgba(84, 160, 255, 0.18)) !important;
    box-shadow: 0 10px 24px rgba(30, 41, 89, 0.34), inset 0 0 0 1px rgba(191, 219, 254, 0.08) !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] p {
    margin: 0 !important;
    font-size: 0.95rem !important;
    font-weight: 600 !important;
    color: rgba(226,232,240,0.80) !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] input:checked + div p {
    color: #ffffff !important;
}
[data-testid="stSidebar"] label[data-baseweb="radio"] input:focus + div {
    box-shadow: 0 0 0 1px rgba(191, 219, 254, 0.18), 0 0 0 4px rgba(99, 102, 241, 0.16) !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# État global (session_state)
# ---------------------------------------------------------------------------

def init_state():
    defaults = {
        "df_raw": None,         # Données brutes chargées
        "df_clean": None,       # Données après nettoyage/normalisation
        "feature_cols": [],     # Features sélectionnées
        "target_col": None,     # Variable cible (classification)
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_comparison_table(summary_df: pd.DataFrame, display_cols: list[str]):
    metric_cols = [col for col in display_cols if col != "Modèle"]
    display_df = summary_df[display_cols].copy()
    best_values = {
        col: display_df[col].max(skipna=True) for col in metric_cols
    }

    header_html = "".join(
        f"<th>{html.escape(col)}</th>" for col in display_cols
    )

    row_html = []
    for _, row in display_df.iterrows():
        cells = [
            f'<td class="model-cell">{html.escape(str(row["Modèle"]))}</td>'
        ]
        for col in metric_cols:
            value = row[col]
            best_value = best_values[col]
            formatted_value = "-" if pd.isna(value) else f"{float(value):.4f}"
            classes = "metric-cell"
            if pd.notna(value) and pd.notna(best_value) and np.isclose(float(value), float(best_value)):
                classes += " is-best"
            cells.append(f'<td class="{classes}">{formatted_value}</td>')
        row_html.append(f"<tr>{''.join(cells)}</tr>")

    table_html = (
        '<div class="comparison-table-wrap">'
        '<table class="comparison-table">'
        f'<thead><tr>{header_html}</tr></thead>'
        f'<tbody>{"".join(row_html)}</tbody>'
        '</table>'
        '</div>'
    )
    st.markdown(table_html, unsafe_allow_html=True)

init_state()

# ---------------------------------------------------------------------------
# Barre latérale – Navigation
# ---------------------------------------------------------------------------

st.sidebar.markdown(
    '<div class="sidebar-hero">'
    '<div class="brand-eyebrow">Data Mining Workspace</div>'
    '<div class="brand-wordmark">InzLab</div>'
    '<div class="brand-divider"></div>'
    '</div>',
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigation",
    ["Prétraitement", "Clustering", "Classification"],
    index=0,
    label_visibility="collapsed",
)


# ===========================================================================
# VOLET 1 – PRÉTRAITEMENT
# ===========================================================================

if page == "Prétraitement":
    st.markdown('<div class="page-header"><h1 class="page-title">Prétraitement</h1><p class="page-subtitle">Chargement · Exploration · Nettoyage · Normalisation</p><div class="title-bar"></div></div>', unsafe_allow_html=True)

    # -----------------------------------------------------------------------
    # 1.1 Importation
    # -----------------------------------------------------------------------
    with st.expander("Importation du dataset", expanded=True):
        uploaded = st.file_uploader(
            "Charger un fichier CSV ou Excel",
            type=["csv", "xlsx", "xls"],
            key="uploader",
        )
        if uploaded:
            try:
                df = load_dataset(uploaded)
                st.session_state.df_raw = df
                st.session_state.df_clean = df.copy()
                st.session_state.feature_cols = list(df.select_dtypes(include=[np.number]).columns)
                st.session_state.target_col = None
                st.success(f"Fichier chargé : {df.shape[0]} lignes, {df.shape[1]} colonnes")
            except Exception as e:
                st.error(f"Erreur lors du chargement : {e}")

    if st.session_state.df_raw is None:
        st.info("Chargez un fichier pour commencer.")
        st.stop()

    df_raw = st.session_state.df_raw
    df_work = st.session_state.df_clean

    # -----------------------------------------------------------------------
    # 1.2 Exploration
    # -----------------------------------------------------------------------
    with st.expander("Exploration des données", expanded=True):
        tab_apercu, tab_stats, tab_types = st.tabs(["Aperçu", "Statistiques descriptives", "Types & Valeurs manquantes"])

        with tab_apercu:
            n_rows = st.slider("Nombre de lignes à afficher", 5, 50, 10, key="apercu_rows")
            st.dataframe(df_work.head(n_rows), use_container_width=True)

        with tab_stats:
            st.dataframe(descriptive_stats(df_work), use_container_width=True)

        with tab_types:
            info = get_basic_info(df_work)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Lignes", info["shape"][0])
            col2.metric("Colonnes", info["shape"][1])
            col3.metric("Valeurs manquantes", sum(info["missing"].values()))
            col4.metric("Doublons", info["duplicates"])

            miss_df = pd.DataFrame({
                "Colonne": list(info["missing"].keys()),
                "Valeurs manquantes": list(info["missing"].values()),
                "Pourcentage (%)": list(info["missing_pct"].values()),
                "Type": [info["dtypes"][c] for c in info["missing"].keys()],
            })
            st.dataframe(miss_df, use_container_width=True)

    # -----------------------------------------------------------------------
    # 1.3 Nettoyage
    # -----------------------------------------------------------------------
    with st.expander("Nettoyage des données"):
        col_a, col_b = st.columns(2)

        with col_a:
            st.markdown("**Gestion des valeurs manquantes**")
            miss_strategy = st.selectbox(
                "Stratégie",
                ["mean", "median", "mode", "constant", "drop_rows", "drop_cols"],
                format_func=lambda x: {
                    "mean": "Remplacement par la moyenne",
                    "median": "Remplacement par la médiane",
                    "mode": "Remplacement par le mode",
                    "constant": "Valeur constante",
                    "drop_rows": "Supprimer les lignes avec NaN",
                    "drop_cols": "Supprimer les colonnes avec NaN",
                }[x],
                key="miss_strat",
            )
            fill_val = None
            if miss_strategy == "constant":
                fill_val = st.text_input("Valeur de remplacement", value="0", key="fill_val")

            if st.button("Appliquer le nettoyage des NaN", key="btn_clean_miss"):
                try:
                    df_work = handle_missing(df_work, miss_strategy, fill_val)
                    st.session_state.df_clean = df_work
                    st.success(f"Nettoyage appliqué. Dataset : {df_work.shape}")
                except Exception as e:
                    st.error(str(e))

        with col_b:
            st.markdown("**Suppression des doublons**")
            n_dup = df_work.duplicated().sum()
            st.info(f"Doublons détectés : **{n_dup}**")
            if st.button("Supprimer les doublons", key="btn_rm_dup", disabled=(n_dup == 0)):
                df_work = remove_duplicates(df_work)
                st.session_state.df_clean = df_work
                st.success(f"Doublons supprimés. Dataset : {df_work.shape}")

    # -----------------------------------------------------------------------
    # 1.4 Normalisation
    # -----------------------------------------------------------------------
    with st.expander("Normalisation"):
        num_cols = list(df_work.select_dtypes(include=[np.number]).columns)
        if not num_cols:
            st.warning("Aucune colonne numérique disponible.")
        else:
            norm_method = st.radio(
                "Méthode",
                ["minmax", "zscore"],
                format_func=lambda x: {"minmax": "Min-Max Scaling [0,1]", "zscore": "Standardisation (Z-score)"}[x],
                horizontal=True,
                key="norm_method",
            )
            # La cible ne doit jamais être normalisée — on l'exclut des options par défaut
            target_for_norm = st.session_state.target_col
            normable_cols = [c for c in num_cols if c != target_for_norm]
            if target_for_norm and target_for_norm in num_cols:
                st.info(f"La variable cible **{target_for_norm}** est exclue des colonnes normalisables.")
            cols_to_norm = st.multiselect(
                "Colonnes à normaliser",
                normable_cols,
                default=normable_cols,
                key="cols_to_norm",
            )
            if st.button("Appliquer la normalisation", key="btn_norm"):
                if not cols_to_norm:
                    st.warning("Sélectionnez au moins une colonne.")
                else:
                    df_work = normalize(df_work, norm_method, cols_to_norm)
                    st.session_state.df_clean = df_work
                    st.success(f"Normalisation appliquée ({norm_method}) sur {len(cols_to_norm)} colonnes.")
                    st.dataframe(df_work[cols_to_norm].head(5), use_container_width=True)

    # -----------------------------------------------------------------------
    # 1.5 Visualisation
    # -----------------------------------------------------------------------
    with st.expander("Visualisation"):
        num_cols_viz = list(df_work.select_dtypes(include=[np.number]).columns)
        all_cols = list(df_work.columns)

        tab_box, tab_hist, tab_scatter, tab_corr = st.tabs([
            "Boxplots", "Histogrammes", "Scatter Plot", "Corrélation"
        ])

        with tab_box:
            cols_box = st.multiselect("Colonnes pour les boxplots", num_cols_viz,
                                       default=num_cols_viz[:min(6, len(num_cols_viz))],
                                       key="box_cols")
            if cols_box:
                fig = plot_boxplots(df_work, cols_box)
                if fig:
                    st.pyplot(fig)

        with tab_hist:
            cols_hist = st.multiselect("Colonnes pour les histogrammes", num_cols_viz,
                                        default=num_cols_viz[:min(6, len(num_cols_viz))],
                                        key="hist_cols")
            if cols_hist:
                fig = plot_histograms(df_work, cols_hist)
                if fig:
                    st.pyplot(fig)

        with tab_scatter:
            if len(num_cols_viz) < 2:
                st.warning("Au moins 2 colonnes numériques requises.")
            else:
                col1, col2, col3 = st.columns(3)
                x_col = col1.selectbox("Axe X", num_cols_viz, key="sc_x")
                y_col = col2.selectbox("Axe Y", num_cols_viz,
                                        index=min(1, len(num_cols_viz) - 1), key="sc_y")
                hue_col = col3.selectbox("Couleur (optionnel)", ["Aucun"] + all_cols, key="sc_hue")
                fig = plot_scatter(df_work, x_col, y_col,
                                   hue_col=None if hue_col == "Aucun" else hue_col)
                st.pyplot(fig)

        with tab_corr:
            fig = plot_correlation_heatmap(df_work)
            if fig:
                st.pyplot(fig)
            else:
                st.info("Pas assez de colonnes numériques pour une heatmap.")

    # -----------------------------------------------------------------------
    # 1.6 Sélection des features & cible (persistée pour les autres volets)
    # -----------------------------------------------------------------------
    with st.expander("Sélection des features et de la variable cible", expanded=True):
        all_cols_clean = list(df_work.columns)
        num_cols_clean = list(df_work.select_dtypes(include=[np.number]).columns)

        feature_cols = st.multiselect(
            "Features (colonnes utilisées pour le clustering & la classification)",
            num_cols_clean,
            default=st.session_state.feature_cols
                    if st.session_state.feature_cols else num_cols_clean,
            key="feat_select",
        )
        target_col = st.selectbox(
            "Variable cible (classification uniquement)",
            ["Aucune"] + all_cols_clean,
            index=(all_cols_clean.index(st.session_state.target_col) + 1
                   if st.session_state.target_col in all_cols_clean else 0),
            key="target_select",
        )

        if st.button("Enregistrer la sélection", key="btn_save_cols"):
            st.session_state.feature_cols = feature_cols
            chosen_target = None if target_col == "Aucune" else target_col
            st.session_state.target_col = chosen_target
            st.success("Sélection enregistrée.")
            # Avertir si la cible choisie est numérique et pourrait avoir été normalisée
            if chosen_target and chosen_target in df_work.select_dtypes(include=[np.number]).columns:
                st.warning(
                    f"**{chosen_target}** est une colonne numérique. "
                    "Si vous avez appliqué une normalisation avant de définir la cible, "
                    "rechargez le dataset et refaites le prétraitement dans l'ordre recommandé : "
                    "nettoyage → sélection de la cible → normalisation des features."
                )


# ===========================================================================
# VOLET 2 – CLUSTERING
# ===========================================================================

elif page == "Clustering":
    st.markdown('<div class="page-header"><h1 class="page-title">Clustering</h1><p class="page-subtitle">K-Means · K-Medoids · DBSCAN · AGNES · DIANA</p><div class="title-bar"></div></div>', unsafe_allow_html=True)

    df_work = st.session_state.df_clean
    feature_cols = st.session_state.feature_cols

    if df_work is None:
        st.warning("Chargez et préparez un dataset dans le volet Prétraitement.")
        st.stop()

    if not feature_cols:
        st.warning("Sélectionnez des features dans le volet Prétraitement (section Sélection).")
        st.stop()

    # Préparer X — exclure la variable cible pour ne pas biaiser le clustering
    target_col_clust = st.session_state.target_col
    valid_feat = [c for c in feature_cols
                  if c in df_work.columns
                  and pd.api.types.is_numeric_dtype(df_work[c])
                  and c != target_col_clust]
    if not valid_feat:
        st.error("Aucune feature numérique valide (après exclusion de la cible).")
        st.stop()
    if target_col_clust and target_col_clust in feature_cols:
        st.info(f"La variable cible **{target_col_clust}** est automatiquement exclue des features de clustering.")

    X_all = df_work[valid_feat].dropna().values
    if len(X_all) < 3:
        st.error("Pas assez de données pour le clustering (minimum 3 lignes).")
        st.stop()

    st.info(f"Dataset utilisé pour le clustering : **{X_all.shape[0]}** observations × **{X_all.shape[1]}** features")

    # -----------------------------------------------------------------------
    # 2.1 Courbe d'Elbow
    # -----------------------------------------------------------------------
    with st.expander("Courbe d'Elbow (K-Means / K-Medoids)", expanded=True):
        col1, col2, col3 = st.columns(3)
        k_min = col1.number_input("k minimum", min_value=2, max_value=10, value=2, key="elbow_kmin")
        k_max = col2.number_input("k maximum", min_value=3, max_value=20, value=10, key="elbow_kmax")
        elbow_algo = col3.selectbox("Algorithme", ["kmeans", "kmedoids"],
                                     format_func=lambda x: {"kmeans": "K-Means", "kmedoids": "K-Medoids"}[x],
                                     key="elbow_algo")

        if st.button("Calculer la courbe d'Elbow", key="btn_elbow"):
            with st.spinner("Calcul en cours…"):
                ks, inertias = elbow_curve(X_all, range(int(k_min), int(k_max) + 1), elbow_algo)
            if ks:
                fig = plot_elbow(ks, inertias, "K-Means" if elbow_algo == "kmeans" else "K-Medoids")
                st.pyplot(fig)
                st.caption("Choisissez k au point de « coude » de la courbe.")
            else:
                st.error("Impossible de calculer la courbe d'Elbow.")

    # -----------------------------------------------------------------------
    # 2.2 Lancer un algorithme de clustering
    # -----------------------------------------------------------------------
    with st.expander("Paramétrage & Exécution", expanded=True):
        algo = st.selectbox(
            "Algorithme de clustering",
            ["K-Means", "K-Medoids", "DBSCAN", "AGNES", "DIANA"],
            key="cluster_algo",
        )

        labels = None

        # --- Paramètres spécifiques ---
        if algo in ("K-Means", "K-Medoids", "AGNES", "DIANA"):
            k = st.slider("Nombre de clusters k", 2, min(20, len(X_all) - 1), 3, key="k_val")

        if algo == "AGNES":
            linkage_method = st.selectbox(
                "Méthode de liaison",
                ["ward", "complete", "average", "single"],
                key="agnes_linkage",
            )

        if algo == "DBSCAN":
            col_e, col_m = st.columns(2)
            eps_val = col_e.slider("eps (rayon de voisinage)", 0.01, 5.0, 0.5, 0.01, key="dbscan_eps")
            min_s = col_m.slider("min_samples", 1, 20, 5, key="dbscan_min")

        if algo in ("K-Medoids", "DIANA") and len(X_all) > 800:
            st.warning(
                f"**{algo}** a une complexité O(n²). "
                f"Avec {len(X_all)} observations, le calcul peut prendre plusieurs minutes."
            )

        if st.button("Lancer le clustering", key="btn_run_cluster"):
            with st.spinner(f"Exécution de {algo}…"):
                try:
                    if algo == "K-Means":
                        labels, model = run_kmeans(X_all, k)
                        inertia = model.inertia_
                        st.metric("Inertie (WCSS)", f"{inertia:.4f}")

                    elif algo == "K-Medoids":
                        labels, inertia, _ = run_kmedoids(X_all, k)
                        st.metric("Inertie (somme distances médoïdes)", f"{inertia:.4f}")

                    elif algo == "DBSCAN":
                        labels, _ = run_dbscan(X_all, eps=eps_val, min_samples=int(min_s))
                        n_clusters = len(set(labels) - {-1})
                        n_noise = int((labels == -1).sum())
                        st.metric("Clusters trouvés", n_clusters)
                        st.metric("Points bruit", n_noise)

                    elif algo == "AGNES":
                        labels, _ = run_agnes(X_all, k, linkage=linkage_method)

                    elif algo == "DIANA":
                        labels = run_diana(X_all, k)

                    # Score de Silhouette
                    sil = compute_silhouette(X_all, labels)
                    if sil is not None:
                        st.metric("Score de Silhouette", f"{sil:.4f}",
                                  help="Entre -1 et 1. Plus proche de 1 = meilleur clustering.")
                    else:
                        st.info("Score de Silhouette non calculable (cluster unique ou bruit total).")

                    # Visualisation 2D
                    st.markdown("##### Visualisation des clusters (2D)")
                    fig_clust = plot_clusters_2d(X_all, labels, title=f"Clustering – {algo}")
                    st.pyplot(fig_clust)

                    # Tableau récap des clusters
                    st.markdown("##### Composition des clusters")
                    unique, counts = np.unique(labels, return_counts=True)
                    recap = pd.DataFrame({"Cluster": unique, "Nombre d'observations": counts})
                    recap["Cluster"] = recap["Cluster"].apply(
                        lambda x: "Bruit (-1)" if x == -1 else f"Cluster {x}"
                    )
                    st.dataframe(recap, use_container_width=True)

                    # Dendrogramme pour méthodes hiérarchiques
                    if algo in ("AGNES", "DIANA"):
                        st.markdown("##### Dendrogramme")
                        m = linkage_method if algo == "AGNES" else "ward"
                        fig_dend = plot_dendrogram(X_all, method=m)
                        st.pyplot(fig_dend)

                except Exception as e:
                    st.error(f"Erreur : {e}")


# ===========================================================================
# VOLET 3 – CLASSIFICATION
# ===========================================================================

elif page == "Classification":
    st.markdown('<div class="page-header"><h1 class="page-title">Classification</h1><p class="page-subtitle">KNN · Arbre · Random Forest · SVM · Régression · Naive Bayes</p><div class="title-bar"></div></div>', unsafe_allow_html=True)

    df_work = st.session_state.df_clean
    feature_cols = st.session_state.feature_cols
    target_col = st.session_state.target_col

    if df_work is None:
        st.warning("Chargez et préparez un dataset dans le volet Prétraitement.")
        st.stop()

    all_cols_class = list(df_work.columns)
    num_cols_class = list(df_work.select_dtypes(include=[np.number]).columns)

    needs_class_setup = not feature_cols or not target_col
    with st.expander("Configuration des données pour la classification", expanded=needs_class_setup):
        st.caption("Choisissez ici les colonnes explicatives et la variable cible si elles ne sont pas encore définies.")

        default_target = target_col if target_col in all_cols_class else "Aucune"
        target_choice = st.selectbox(
            "Variable cible",
            ["Aucune"] + all_cols_class,
            index=(["Aucune"] + all_cols_class).index(default_target),
            key="class_target_select",
        )

        effective_target = None if target_choice == "Aucune" else target_choice
        default_features = [
            col for col in feature_cols
            if col in num_cols_class and col != effective_target
        ] if feature_cols else [
            col for col in num_cols_class if col != effective_target
        ]

        feature_choices = st.multiselect(
            "Features numériques",
            [col for col in num_cols_class if col != effective_target],
            default=default_features,
            key="class_feat_select",
        )

        if st.button("Enregistrer la configuration", key="btn_save_class_setup"):
            st.session_state.target_col = effective_target
            st.session_state.feature_cols = feature_choices
            st.success("Configuration de classification enregistrée.")
            st.rerun()

    if not feature_cols:
        st.warning("Sélectionnez des features dans la configuration ci-dessus.")
        st.stop()

    if not target_col:
        st.warning("Sélectionnez une variable cible dans la configuration ci-dessus.")
        st.stop()

    st.info(f"Features : **{len(feature_cols)}** colonnes  |  Cible : **{target_col}**")

    # -----------------------------------------------------------------------
    # 3.1 Partitionnement
    # -----------------------------------------------------------------------
    with st.expander("Partitionnement Train / Test", expanded=True):
        test_size = st.slider("Proportion des données de test (%)", 10, 40, 20, key="test_size") / 100
        try:
            X_train, X_test, y_train, y_test, le, valid_features = prepare_classification_data(
                df_work, feature_cols, target_col, test_size=test_size
            )
            class_names = list(le.classes_)
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Total", len(X_train) + len(X_test))
            col2.metric("Entraînement", len(X_train))
            col3.metric("Test", len(X_test))
            col4.metric("Classes", len(class_names))
            st.success(f"Classes détectées : {class_names}")
        except Exception as e:
            st.error(str(e))
            st.stop()

    # -----------------------------------------------------------------------
    # 3.2 Un seul modèle en détail
    # -----------------------------------------------------------------------
    with st.expander("Entraîner & Évaluer un modèle", expanded=True):
        model_name = st.selectbox("Modèle", list(MODELS.keys()), key="single_model")

        # Paramètres ajustables selon le modèle
        custom_params = {}
        if model_name == "K-Nearest Neighbors":
            custom_params["n_neighbors"] = st.slider("k voisins", 1, 20, 5, key="knn_k")
        elif model_name == "Decision Tree":
            depth = st.slider("Profondeur maximale (0 = illimité)", 0, 20, 0, key="dt_depth")
            custom_params["max_depth"] = None if depth == 0 else depth
        elif model_name == "Random Forest":
            custom_params["n_estimators"] = st.slider("Nombre d'arbres", 10, 300, 100, key="rf_n")
        elif model_name == "SVM":
            custom_params["kernel"] = st.selectbox("Noyau", ["rbf", "linear", "poly"], key="svm_ker")

        if st.button("Entraîner le modèle", key="btn_train_single"):
            with st.spinner(f"Entraînement de {model_name}…"):
                try:
                    model = train_model(model_name, X_train, y_train, custom_params)
                    metrics = evaluate_model(model, X_test, y_test, class_names)

                    # Métriques principales
                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Accuracy", f"{metrics['accuracy']:.4f}")
                    m2.metric("Precision", f"{metrics['precision']:.4f}")
                    m3.metric("Recall", f"{metrics['recall']:.4f}")
                    m4.metric("F1-Score", f"{metrics['f1']:.4f}")

                    col_cm, col_bar = st.columns(2)
                    with col_cm:
                        st.markdown("**Matrice de Confusion**")
                        fig_cm = plot_confusion_matrix(metrics["cm"], class_names)
                        st.pyplot(fig_cm)
                    with col_bar:
                        st.markdown("**Métriques**")
                        fig_bar = plot_metrics_bar(metrics, model_name)
                        st.pyplot(fig_bar)

                    # Rapport détaillé
                    with st.expander("Rapport de classification complet"):
                        st.text(metrics["report"])

                    # Visualisations spécifiques
                    if model_name == "Decision Tree":
                        st.markdown("**Visualisation de l'arbre de décision**")
                        fig_tree = plot_decision_tree(model, valid_features, class_names)
                        st.pyplot(fig_tree)

                    fig_imp = plot_feature_importance(model, valid_features, model_name)
                    if fig_imp:
                        st.markdown("**Importance des features**")
                        st.pyplot(fig_imp)

                except Exception as e:
                    st.error(f"Erreur : {e}")

    # -----------------------------------------------------------------------
    # 3.3 Comparaison de plusieurs modèles
    # -----------------------------------------------------------------------
    with st.expander("Comparaison des modèles"):
        models_to_compare = st.multiselect(
            "Modèles à comparer",
            list(MODELS.keys()),
            default=list(MODELS.keys()),
            key="compare_models",
        )
        if st.button("Lancer la comparaison", key="btn_compare"):
            if not models_to_compare:
                st.warning("Sélectionnez au moins un modèle.")
            else:
                with st.spinner("Entraînement de tous les modèles…"):
                    summary_df, results = compare_models(
                        models_to_compare, X_train, X_test, y_train, y_test, class_names
                    )
                display_cols = ["Modèle", "Accuracy", "Precision", "Recall", "F1-Score"]
                st.caption("Les meilleures valeurs de chaque métrique sont surlignées.")
                render_comparison_table(summary_df, display_cols)
                # Afficher les erreurs éventuelles
                if "Erreur" in summary_df.columns:
                    errors = summary_df[summary_df["Erreur"].notna()][["Modèle", "Erreur"]]
                    if not errors.empty:
                        st.warning("Certains modèles ont échoué :")
                        for _, row in errors.iterrows():
                            st.error(f"**{row['Modèle']}** : {row['Erreur']}")

                fig_comp = plot_model_comparison(summary_df)
                if fig_comp:
                    st.pyplot(fig_comp)

                # Meilleur modèle selon F1
                _metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
                valid_summary = summary_df.dropna(subset=_metric_cols)
                if not valid_summary.empty:
                    best_row = valid_summary.sort_values("F1-Score", ascending=False).iloc[0]
                    st.success(f"Meilleur modèle (F1-Score) : **{best_row['Modèle']}** — F1 = {best_row['F1-Score']:.4f}")
                else:
                    st.error("Aucun modèle n'a pu être évalué sur ce dataset.")

                # Matrices de confusion pour chaque modèle
                if results:
                    st.markdown("##### Matrices de Confusion")
                    n_models = len(results)
                    cols = st.columns(min(3, n_models))
                    for i, (name, res) in enumerate(results.items()):
                        with cols[i % len(cols)]:
                            st.markdown(f"**{name}**")
                            fig_cm = plot_confusion_matrix(res["metrics"]["cm"], class_names, title=name)
                            st.pyplot(fig_cm)

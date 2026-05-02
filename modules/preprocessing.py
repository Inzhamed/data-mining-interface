"""
Volet 1 – Prétraitement (Preprocessing)
Toutes les fonctions utilitaires de chargement, exploration, nettoyage,
normalisation et visualisation des données.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'savefig.facecolor': '#1a1a2e',
    'grid.color': '#2d2d5e',
    'axes.edgecolor': '#3a3a6e',
})


# ---------------------------------------------------------------------------
# 1. Importation
# ---------------------------------------------------------------------------

def load_dataset(uploaded_file) -> pd.DataFrame:
    """Charge un fichier CSV ou Excel uploadé via Streamlit."""
    name = uploaded_file.name.lower()
    if name.endswith(".csv"):
        # Essaye plusieurs séparateurs courants
        raw = uploaded_file.read().decode("utf-8", errors="replace")
        uploaded_file.seek(0)
        for sep in [",", ";", "\t"]:
            try:
                df = pd.read_csv(StringIO(raw), sep=sep)
                if df.shape[1] > 1:
                    return df
            except Exception:
                pass
        return pd.read_csv(StringIO(raw))
    elif name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded_file)
    else:
        raise ValueError(f"Format non supporté : {uploaded_file.name}")


# ---------------------------------------------------------------------------
# 2. Exploration
# ---------------------------------------------------------------------------

def get_basic_info(df: pd.DataFrame) -> dict:
    """Retourne un dictionnaire de méta-informations sur le dataframe."""
    buf = StringIO()
    df.info(buf=buf)
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
        "info_str": buf.getvalue(),
    }


def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Statistiques descriptives étendues (numériques + catégorielles)."""
    # Variables numériques
    num = df.describe(include=[np.number]).T
    num["missing"] = df[num.index].isnull().sum()
    num["missing_%"] = (df[num.index].isnull().mean() * 100).round(2)

    # Variables catégorielles
    cat_cols = df.select_dtypes(exclude=[np.number]).columns
    if len(cat_cols) > 0:
        cat = df[cat_cols].describe().T
        cat["missing"] = df[cat_cols].isnull().sum()
        cat["missing_%"] = (df[cat_cols].isnull().mean() * 100).round(2)
        # Aligner les colonnes pour concaténation propre
        return pd.concat([num, cat], axis=0, sort=False)
    return num


# ---------------------------------------------------------------------------
# 3. Nettoyage
# ---------------------------------------------------------------------------

def handle_missing(df: pd.DataFrame, strategy: str, fill_value=None) -> pd.DataFrame:
    """
    Gère les valeurs manquantes.
    strategy : "drop_rows" | "drop_cols" | "mean" | "median" | "mode" | "constant"
    """
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns
    cat_cols = df.select_dtypes(exclude=[np.number]).columns

    if strategy == "drop_rows":
        df = df.dropna()
    elif strategy == "drop_cols":
        df = df.dropna(axis=1)
    elif strategy == "mean":
        df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0] if not df[cat_cols].empty else "unknown")
    elif strategy == "median":
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())
        df[cat_cols] = df[cat_cols].fillna(df[cat_cols].mode().iloc[0] if not df[cat_cols].empty else "unknown")
    elif strategy == "mode":
        for col in df.columns:
            if df[col].isnull().any():
                mode_val = df[col].mode()
                df[col] = df[col].fillna(mode_val.iloc[0] if not mode_val.empty else 0)
    elif strategy == "constant":
        df = df.fillna(fill_value if fill_value is not None else 0)
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Supprime les doublons."""
    return df.drop_duplicates().reset_index(drop=True)


# ---------------------------------------------------------------------------
# 4. Normalisation
# ---------------------------------------------------------------------------

def normalize(df: pd.DataFrame, method: str, columns: list) -> pd.DataFrame:
    """
    Normalise les colonnes numériques sélectionnées.
    method : "minmax" | "zscore"
    """
    df = df.copy()
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return df
    if method == "minmax":
        for c in cols:
            mn, mx = df[c].min(), df[c].max()
            if mx - mn > 0:
                df[c] = (df[c] - mn) / (mx - mn)
            else:
                df[c] = 0.0
    elif method == "zscore":
        for c in cols:
            std = df[c].std()
            if std > 0:
                df[c] = (df[c] - df[c].mean()) / std
            else:
                df[c] = 0.0
    return df


# ---------------------------------------------------------------------------
# 5. Visualisation
# ---------------------------------------------------------------------------

def plot_boxplots(df: pd.DataFrame, columns: list, max_cols: int = 4):
    """Retourne une figure matplotlib avec les boxplots des colonnes choisies."""
    cols = [c for c in columns if c in df.select_dtypes(include=[np.number]).columns]
    if not cols:
        return None
    n = len(cols)
    ncols = min(n, max_cols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].boxplot(df[col].dropna(), patch_artist=True,
                        boxprops=dict(facecolor="#4C72B0", color="#2d3a4a"),
                        medianprops=dict(color="#ff7f0e", linewidth=2))
        axes[i].set_title(col, fontsize=11)
        axes[i].set_xlabel("")
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    return fig


def plot_scatter(df: pd.DataFrame, x_col: str, y_col: str, hue_col: str = None):
    """Retourne un scatter plot entre deux colonnes, avec hue optionnel."""
    fig, ax = plt.subplots(figsize=(7, 5))
    if hue_col and hue_col in df.columns:
        categories = df[hue_col].astype(str).unique()
        palette = sns.color_palette("tab10", len(categories))
        for color, cat in zip(palette, categories):
            mask = df[hue_col].astype(str) == cat
            ax.scatter(df.loc[mask, x_col], df.loc[mask, y_col],
                       label=cat, color=color, alpha=0.7, s=40)
        ax.legend(title=hue_col, bbox_to_anchor=(1.05, 1), loc="upper left")
    else:
        ax.scatter(df[x_col], df[y_col], alpha=0.7, s=40, color="#4C72B0")
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.set_title(f"{x_col} vs {y_col}")
    fig.tight_layout()
    return fig


def plot_correlation_heatmap(df: pd.DataFrame):
    """Retourne une heatmap de corrélation des variables numériques."""
    num_df = df.select_dtypes(include=[np.number])
    if num_df.shape[1] < 2:
        return None
    fig, ax = plt.subplots(figsize=(max(6, num_df.shape[1]), max(5, num_df.shape[1] - 1)))
    sns.heatmap(num_df.corr(), annot=True, fmt=".2f", cmap="coolwarm",
                linewidths=0.5, ax=ax, annot_kws={"size": 8})
    ax.set_title("Matrice de corrélation")
    fig.tight_layout()
    return fig


def plot_histograms(df: pd.DataFrame, columns: list, max_cols: int = 4):
    """Histogrammes des colonnes numériques choisies."""
    cols = [c for c in columns if c in df.select_dtypes(include=[np.number]).columns]
    if not cols:
        return None
    n = len(cols)
    ncols = min(n, max_cols)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False)
    axes = axes.flatten()
    for i, col in enumerate(cols):
        axes[i].hist(df[col].dropna(), bins=20, color="#4C72B0", edgecolor="white")
        axes[i].set_title(col, fontsize=11)
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)
    fig.tight_layout()
    return fig

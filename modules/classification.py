"""
Volet 3 – Classification (Apprentissage Supervisé)
Partitionnement, modèles, évaluation (matrice de confusion, métriques).
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'savefig.facecolor': '#1a1a2e',
    'grid.color': '#2d2d5e',
    'axes.edgecolor': '#3a3a6e',
})
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB


# ---------------------------------------------------------------------------
# Modèles disponibles
# ---------------------------------------------------------------------------

MODELS = {
    "K-Nearest Neighbors": KNeighborsClassifier,
    "Decision Tree": DecisionTreeClassifier,
    "Random Forest": RandomForestClassifier,
    "SVM": SVC,
    "Logistic Regression": LogisticRegression,
    "Naive Bayes": GaussianNB,
}

MODEL_PARAMS = {
    "K-Nearest Neighbors": {"n_neighbors": 5},
    "Decision Tree": {"max_depth": None, "random_state": 42},
    "Random Forest": {"n_estimators": 100, "random_state": 42},
    "SVM": {"kernel": "rbf", "probability": True, "random_state": 42},
    "Logistic Regression": {"max_iter": 1000, "random_state": 42},
    "Naive Bayes": {},
}


# ---------------------------------------------------------------------------
# Préparation des données
# ---------------------------------------------------------------------------

def prepare_classification_data(
    df: pd.DataFrame,
    feature_cols: list,
    target_col: str,
    test_size: float = 0.2,
    random_state: int = 42,
):
    """
    Encode la cible si nécessaire, filtre les features numériques,
    supprime les NaN et retourne X_train, X_test, y_train, y_test + encodeur.
    """
    valid_features = [c for c in feature_cols
                      if c in df.columns and c != target_col
                      and pd.api.types.is_numeric_dtype(df[c])]
    if not valid_features:
        raise ValueError("Aucune colonne numérique sélectionnée comme feature.")

    sub = df[valid_features + [target_col]].dropna()
    if len(sub) < 10:
        raise ValueError("Pas assez de données après suppression des NaN (min 10 lignes).")

    X = sub[valid_features].values
    y_raw = sub[target_col].values

    le = LabelEncoder()
    y = le.fit_transform(y_raw)

    if len(np.unique(y)) < 2:
        raise ValueError("La variable cible doit contenir au moins 2 classes.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test, le, valid_features


# ---------------------------------------------------------------------------
# Entraînement & évaluation
# ---------------------------------------------------------------------------

def train_model(model_name: str, X_train, y_train, custom_params: dict = None):
    """Instancie et entraîne un modèle. Retourne le modèle entraîné."""
    params = dict(MODEL_PARAMS.get(model_name, {}))
    if custom_params:
        params.update(custom_params)
    model = MODELS[model_name](**params)
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, class_names=None):
    """
    Évalue le modèle sur les données de test.
    Retourne un dictionnaire de métriques.
    """
    y_pred = model.predict(X_test)
    avg = "binary" if len(np.unique(y_test)) == 2 else "weighted"

    safe_class_names = [str(c) for c in class_names] if class_names is not None else None
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        "cm": confusion_matrix(y_test, y_pred),
        "report": classification_report(
            y_test, y_pred,
            target_names=safe_class_names,
            zero_division=0
        ),
        "y_pred": y_pred,
    }
    return metrics


# ---------------------------------------------------------------------------
# Visualisations
# ---------------------------------------------------------------------------

def plot_confusion_matrix(cm: np.ndarray, class_names=None, title: str = "Matrice de Confusion"):
    """Affiche la matrice de confusion sous forme de heatmap."""
    if class_names is None:
        class_names = [str(i) for i in range(len(cm))]
    fig, ax = plt.subplots(figsize=(max(5, len(class_names)), max(4, len(class_names) - 1)))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=class_names, yticklabels=class_names,
        linewidths=0.5, ax=ax
    )
    ax.set_xlabel("Prédit", fontsize=12)
    ax.set_ylabel("Réel", fontsize=12)
    ax.set_title(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_metrics_bar(metrics: dict, model_name: str):
    """Diagramme en barres des métriques clés."""
    keys = ["accuracy", "precision", "recall", "f1"]
    values = [metrics[k] for k in keys]
    labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(labels, values, color=colors, edgecolor="white", width=0.5)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Score")
    ax.set_title(f"Métriques – {model_name}")
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                f"{val:.3f}", ha="center", va="bottom", fontsize=11)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_decision_tree(model, feature_names, class_names, max_depth: int = 3):
    """Visualise un arbre de décision."""
    fig, ax = plt.subplots(figsize=(16, 8))
    plot_tree(
        model,
        feature_names=feature_names,
        class_names=[str(c) for c in class_names],
        filled=True,
        max_depth=max_depth,
        fontsize=9,
        ax=ax,
    )
    fig.tight_layout()
    return fig


def plot_feature_importance(model, feature_names: list, model_name: str):
    """Importance des features pour les modèles qui l'exposent."""
    if not hasattr(model, "feature_importances_"):
        return None
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    sorted_names = [feature_names[i] for i in idx]
    sorted_imp = importances[idx]

    fig, ax = plt.subplots(figsize=(8, max(4, len(feature_names) * 0.4)))
    ax.barh(sorted_names[::-1], sorted_imp[::-1], color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Importance")
    ax.set_title(f"Importance des features – {model_name}")
    ax.grid(axis="x", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def compare_models(
    model_names: list,
    X_train, X_test, y_train, y_test,
    class_names=None,
):
    """
    Entraîne et évalue plusieurs modèles.
    Retourne un DataFrame récapitulatif et le dictionnaire des résultats individuels.
    """
    rows = []
    results = {}
    for name in model_names:
        try:
            model = train_model(name, X_train, y_train)
            m = evaluate_model(model, X_test, y_test, class_names)
            results[name] = {"model": model, "metrics": m}
            rows.append({
                "Modèle": name,
                "Accuracy": round(m["accuracy"], 4),
                "Precision": round(m["precision"], 4),
                "Recall": round(m["recall"], 4),
                "F1-Score": round(m["f1"], 4),
            })
        except Exception as e:
            rows.append({
                "Modèle": name,
                "Accuracy": None,
                "Precision": None,
                "Recall": None,
                "F1-Score": None,
                "Erreur": str(e),
            })
    return pd.DataFrame(rows), results


def plot_model_comparison(summary_df: pd.DataFrame):
    """Graphique de comparaison des modèles."""
    metric_cols = ["Accuracy", "Precision", "Recall", "F1-Score"]
    df = summary_df.dropna(subset=metric_cols)
    if df.empty:
        return None
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score"]
    x = np.arange(len(df))
    width = 0.2
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 2), 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        ax.bar(x + i * width, df[metric], width, label=metric, color=color, edgecolor="white")
    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(df["Modèle"], rotation=15, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Score")
    ax.set_title("Comparaison des modèles")
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig

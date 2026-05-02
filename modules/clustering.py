"""
Volet 2 – Clustering
Algorithmes : K-Means, K-Medoids, DBSCAN, AGNES (agglomératif), DIANA (divisif).
Évaluation  : Courbe d'Elbow (inertie), Score de Silhouette.
Visualisation : Projection 2D via PCA.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.decomposition import PCA

plt.style.use('dark_background')
plt.rcParams.update({
    'figure.facecolor': '#1a1a2e',
    'axes.facecolor': '#16213e',
    'savefig.facecolor': '#1a1a2e',
    'grid.color': '#2d2d5e',
    'axes.edgecolor': '#3a3a6e',
})
from sklearn.metrics import silhouette_score


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pca_2d(X: np.ndarray):
    """Projette X en 2D via PCA si nécessaire."""
    if X.shape[1] == 1:
        return np.column_stack([X, np.zeros(len(X))])
    if X.shape[1] == 2:
        return X
    pca = PCA(n_components=2, random_state=42)
    return pca.fit_transform(X)


# ---------------------------------------------------------------------------
# 1. K-Means
# ---------------------------------------------------------------------------

def run_kmeans(X: np.ndarray, k: int, random_state: int = 42):
    """Applique K-Means et retourne les labels et le modèle."""
    model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
    labels = model.fit_predict(X)
    return labels, model


# ---------------------------------------------------------------------------
# 2. K-Medoids (implémentation manuelle – pas de dépendance externe imposée)
# ---------------------------------------------------------------------------

def run_kmedoids(X: np.ndarray, k: int, max_iter: int = 300, random_state: int = 42):
    """
    K-Medoids (PAM simplifié).
    Retourne : labels (np.ndarray), inertia (float), medoid_indices (list).
    """
    rng = np.random.default_rng(random_state)
    n = len(X)
    medoid_idx = rng.choice(n, k, replace=False).tolist()

    for _ in range(max_iter):
        # Assignation
        dists = np.array([[np.linalg.norm(X[i] - X[m]) for m in medoid_idx] for i in range(n)])
        labels = np.argmin(dists, axis=1)

        # Mise à jour des médoïdes
        new_medoids = []
        for c in range(k):
            cluster_pts = np.where(labels == c)[0]
            if len(cluster_pts) == 0:
                new_medoids.append(medoid_idx[c])
                continue
            sub = X[cluster_pts]
            intra = np.sum(np.linalg.norm(sub[:, None] - sub[None, :], axis=2), axis=1)
            new_medoids.append(cluster_pts[np.argmin(intra)])

        if new_medoids == medoid_idx:
            break
        medoid_idx = new_medoids

    # Recalcul final des labels avec les médoïdes définitifs (cohérence garantie)
    dists = np.array([[np.linalg.norm(X[i] - X[m]) for m in medoid_idx] for i in range(n)])
    labels = np.argmin(dists, axis=1)
    inertia = sum(np.linalg.norm(X[i] - X[medoid_idx[labels[i]]]) for i in range(n))
    return labels, inertia, medoid_idx


# ---------------------------------------------------------------------------
# 3. DBSCAN
# ---------------------------------------------------------------------------

def run_dbscan(X: np.ndarray, eps: float = 0.5, min_samples: int = 5):
    """
    DBSCAN. Retourne les labels (-1 = bruit).
    """
    model = DBSCAN(eps=eps, min_samples=min_samples)
    labels = model.fit_predict(X)
    return labels, model


# ---------------------------------------------------------------------------
# 4. AGNES – Agglomératif (Agglomerative Nesting)
# ---------------------------------------------------------------------------

def run_agnes(X: np.ndarray, k: int, linkage: str = "ward"):
    """
    AGNES (clustering hiérarchique agglomératif).
    linkage : "ward" | "complete" | "average" | "single"
    """
    model = AgglomerativeClustering(n_clusters=k, linkage=linkage)
    labels = model.fit_predict(X)
    return labels, model


# ---------------------------------------------------------------------------
# 5. DIANA – Divisif (DIvisive ANAlysis) – implémentation simplifiée
# ---------------------------------------------------------------------------

def _diameter(X: np.ndarray) -> float:
    """Diamètre d'un cluster = distance maximale intra-cluster."""
    if len(X) <= 1:
        return 0.0
    dists = np.linalg.norm(X[:, None] - X[None, :], axis=2)
    return float(np.max(dists))


def _split_cluster(X: np.ndarray, indices: np.ndarray):
    """
    Divise un cluster en deux via une heuristique de dissimilarité moyenne.
    Travaille sur les sous-points locaux (X[indices]) et retourne des indices originaux.
    """
    n = len(indices)
    if n <= 1:
        return indices, np.array([], dtype=int)

    sub = X[indices]  # sous-ensemble local
    dists = np.linalg.norm(sub[:, None] - sub[None, :], axis=2)  # matrice n×n locale

    # L'objet le plus éloigné en moyenne des autres démarre le splinter group
    avg_dists = dists.mean(axis=1)
    splinter_start = int(np.argmax(avg_dists))
    # splinter et main stockent des positions LOCALES (0..n-1)
    splinter = [splinter_start]
    main = [i for i in range(n) if i != splinter_start]

    changed = True
    while changed:
        changed = False
        new_main = []
        new_splinter = list(splinter)
        for i in list(main):
            d_main = np.mean([dists[i, j] for j in main if j != i]) if len(main) > 1 else np.inf
            d_splinter = np.mean([dists[i, j] for j in splinter])
            if d_splinter < d_main:
                new_splinter.append(i)
                changed = True
            else:
                new_main.append(i)
        main = new_main
        splinter = new_splinter

    # Convertir les positions locales en indices originaux
    return indices[np.array(main, dtype=int)], indices[np.array(splinter, dtype=int)]


def run_diana(X: np.ndarray, k: int):
    """
    DIANA – clustering hiérarchique divisif.
    Divise récursivement le cluster de plus grand diamètre jusqu'à obtenir k clusters.
    Retourne les labels.
    """
    clusters = [np.arange(len(X))]

    while len(clusters) < k:
        # Choisir le cluster avec le plus grand diamètre
        diameters = [_diameter(X[c]) for c in clusters]
        target = int(np.argmax(diameters))
        to_split = clusters.pop(target)
        if len(to_split) <= 1:
            clusters.append(to_split)
            break
        part_a, part_b = _split_cluster(X, to_split)
        clusters.append(part_a)
        if len(part_b) > 0:
            clusters.append(part_b)

    labels = np.zeros(len(X), dtype=int)
    for cluster_id, indices in enumerate(clusters):
        labels[indices] = cluster_id
    return labels


# ---------------------------------------------------------------------------
# Évaluation
# ---------------------------------------------------------------------------

def compute_silhouette(X: np.ndarray, labels: np.ndarray) -> float | None:
    """
    Calcule le score de Silhouette.
    Retourne None si le calcul est impossible (< 2 clusters valides, etc.).
    """
    unique = np.unique(labels[labels != -1])
    if len(unique) < 2:
        return None
    # Filtrer le bruit DBSCAN
    mask = labels != -1
    if mask.sum() < 2:
        return None
    try:
        return float(silhouette_score(X[mask], labels[mask]))
    except Exception:
        return None


def elbow_curve(X: np.ndarray, k_range: range, algorithm: str = "kmeans"):
    """
    Calcule les inerties pour chaque k (K-Means ou K-Medoids).
    Retourne deux listes : ks, inertias.
    """
    ks, inertias = [], []
    for k in k_range:
        if k >= len(X):
            break
        try:
            if algorithm == "kmeans":
                _, model = run_kmeans(X, k)
                inertia = model.inertia_
            else:  # kmedoids
                _, inertia, _ = run_kmedoids(X, k)
            ks.append(k)
            inertias.append(inertia)
        except Exception:
            pass
    return ks, inertias


def recommend_elbow_k(ks: list, inertias: list) -> int | None:
    """
    Propose un k via une heuristique simple de "coude".
    Le point retenu est celui dont la distance à la droite reliant
    le premier et le dernier point de la courbe est maximale.
    """
    if len(ks) < 3 or len(ks) != len(inertias):
        return None

    x = np.asarray(ks, dtype=float)
    y = np.asarray(inertias, dtype=float)

    x_span = x.max() - x.min()
    y_span = y.max() - y.min()
    if np.isclose(x_span, 0.0) or np.isclose(y_span, 0.0):
        return None

    x_norm = (x - x.min()) / x_span
    y_norm = (y - y.min()) / y_span

    start = np.array([x_norm[0], y_norm[0]])
    end = np.array([x_norm[-1], y_norm[-1]])
    line_vec = end - start
    line_norm = np.linalg.norm(line_vec)
    if np.isclose(line_norm, 0.0):
        return None

    points = np.column_stack([x_norm, y_norm])
    distances = np.abs(
        (points[:, 0] - start[0]) * line_vec[1]
        - (points[:, 1] - start[1]) * line_vec[0]
    ) / line_norm

    elbow_idx = int(np.argmax(distances[1:-1])) + 1
    return int(ks[elbow_idx])


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_elbow(ks: list, inertias: list, algorithm: str = "K-Means"):
    """Trace la courbe d'Elbow."""
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(ks, inertias, "bo-", linewidth=2, markersize=7)
    ax.set_xlabel("Nombre de clusters k")
    ax.set_ylabel("Inertie (WCSS)")
    ax.set_title(f"Courbe d'Elbow – {algorithm}")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    return fig


def plot_clusters_2d(X: np.ndarray, labels: np.ndarray, title: str = "Clusters"):
    """Projette les clusters en 2D (PCA si dim > 2)."""
    X2d = _pca_2d(X)
    unique_labels = np.unique(labels)
    palette = plt.colormaps["tab10"]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, lbl in enumerate(unique_labels):
        mask = labels == lbl
        color = "gray" if lbl == -1 else palette(i)
        label_name = "Bruit" if lbl == -1 else f"Cluster {lbl}"
        ax.scatter(X2d[mask, 0], X2d[mask, 1],
                   color=color, label=label_name, alpha=0.7, s=50, edgecolors="k", linewidths=0.3)

    ax.set_title(title + (" (projection PCA)" if X.shape[1] > 2 else ""))
    ax.set_xlabel("Composante 1")
    ax.set_ylabel("Composante 2")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    return fig


def plot_dendrogram(X: np.ndarray, method: str = "ward", max_leaves: int = 50):
    """
    Dendrogramme pour la visualisation hiérarchique (AGNES/DIANA).
    Utilise scipy linkage.
    """
    from scipy.cluster.hierarchy import dendrogram, linkage as scipy_linkage

    fig, ax = plt.subplots(figsize=(12, 5))
    n = min(len(X), max_leaves)
    sample = X if len(X) <= max_leaves else X[np.random.choice(len(X), max_leaves, replace=False)]
    Z = scipy_linkage(sample, method=method)
    dendrogram(Z, ax=ax, leaf_rotation=90, leaf_font_size=8,
               color_threshold=0.7 * max(Z[:, 2]))
    ax.set_title(f"Dendrogramme ({method})")
    ax.set_xlabel("Échantillons")
    ax.set_ylabel("Distance")
    fig.tight_layout()
    return fig

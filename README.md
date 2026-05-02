# Interface Fouille de Données

Mini-Projet – Module Fouille de Données · 2025-2026

## Lancer l'application

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Structure

```
projet-fd/
├── app.py                  # Interface Streamlit principale
├── requirements.txt
└── modules/
    ├── __init__.py
    ├── preprocessing.py    # Volet 1 – Prétraitement
    ├── clustering.py       # Volet 2 – Clustering
    └── classification.py   # Volet 3 – Classification
```

## Fonctionnalités

### Volet 1 – Prétraitement
- Chargement CSV / Excel (auto-détection du séparateur)
- Statistiques descriptives, types, valeurs manquantes, doublons
- Nettoyage : suppression ou imputation (moyenne, médiane, mode, constante)
- Normalisation : Min-Max ou Z-score
- Visualisations : Boxplots, Histogrammes, Scatter Plot, Heatmap de corrélation
- Sélection des features et de la variable cible pour les volets suivants

### Volet 2 – Clustering
- **K-Means** (scikit-learn)
- **K-Medoids** (implémentation PAM)
- **DBSCAN** (paramètres eps et min_samples)
- **AGNES** – hiérarchique agglomératif (ward / complete / average / single)
- **DIANA** – hiérarchique divisif (implémentation manuelle)
- Courbe d'Elbow pour choisir k
- Score de Silhouette
- Visualisation 2D des clusters (PCA si dim > 2)
- Dendrogramme pour les méthodes hiérarchiques

### Volet 3 – Classification
- Partitionnement Train/Test configurable
- Modèles : KNN, Decision Tree, Random Forest, SVM, Logistic Regression, Naive Bayes
- Évaluation : Matrice de Confusion, Accuracy, Precision, Recall, F1-Score
- Rapport de classification détaillé
- Visualisation de l'arbre (Decision Tree) et importance des features
- Comparaison automatique de tous les modèles avec mise en évidence du meilleur

## Formats de fichiers supportés
- CSV (virgule, point-virgule, tabulation)
- Excel (.xlsx, .xls)

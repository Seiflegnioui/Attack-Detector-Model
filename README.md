# 🛡️ Détection d'Intrusions Réseau (IDS) Temps Réel par Machine Learning

## Inspiré de NVIDIA Morpheus

Système de détection d'intrusions réseau (IDS) de bout en bout, utilisant des algorithmes d'apprentissage supervisé avancés (XGBoost, Random Forest) couplés à des techniques de traitement de données (SMOTE) pour détecter le trafic malveillant en temps réel.

---

## 📋 Problématique et Approche

Les cyberattaques évoluent plus rapidement que les méthodes de détection classiques basées sur des signatures statiques. Ce projet construit un **IDS intelligent** en appliquant du Machine Learning pour détecter automatiquement les intrusions réseau (DDoS, Probe, R2L, U2R).

Cette approche s'inspire directement du framework **NVIDIA Morpheus**, qui utilise l'IA pour analyser les comportements anormaux en temps réel dans les flux de données massifs.

Pour y parvenir, ce projet est construit sur trois piliers :
1. **Modélisation Avancée** : Hyperparameter Tuning et Stratified K-Fold Cross Validation.
2. **Gestion du Déséquilibre Extrême** : Utilisation de **SMOTE** (Synthetic Minority Over-sampling Technique) dans un pipeline strict pour détecter les attaques très rares.
3. **Produit Logiciel "Temps Réel"** : Une API backend (FastAPI) ultra-rapide couplée à un Dashboard interactif moderne.

---

## 📊 Dataset : NSL-KDD

Le dataset [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd) est un benchmark public de référence pour la cybersécurité. Le jeu de test contient de nombreux types d'attaques qui n'existent pas dans le jeu d'entraînement, ce qui représente un défi de généralisation énorme.

📁 Le projet suit l'architecture standard MLOps (Cookiecutter Data Science) :
```text
Attack-Detector-Model/
├── api/
│   └── api.py                  # Backend FastAPI
├── data/
│   └── KDD*.txt, KDD*.arff     # Datasets bruts NSL-KDD
├── frontend/
│   └── index.html              # Dashboard UI
├── models/
│   └── *.pkl                   # Modèles entraînés, scalers, colonnes
├── notebooks/
│   ├── model_training.ipynb    # Recherche & expérimentation
│   └── preparedata.ipynb       # Préparation des données
├── reports/
│   └── *.png                   # Graphiques de performance
├── src/
│   ├── run_advanced_models.py  # Code d'entraînement optimisé (XGBoost + SMOTE)
│   └── run_models.py           # Ancienne version (Random Forest)
├── Dockerfile                  # Configuration de l'image de l'API
├── docker-compose.yml          # Orchestration des microservices
├── requirements.txt            # Dépendances Python
└── README.md
```


| Classe | Description | Train Samples | Déséquilibre |
|--------|-------------|:-------------:|:------------:|
| **Normal** | Trafic légitime | 67,343 (53.5%) | Majoritaire |
| **DoS** | Déni de service | 45,927 (36.5%) | Majoritaire |
| **Probe** | Scan/Reconnaissance | 11,656 (9.3%) | Modéré |
| **R2L** | Remote to Local | 995 (0.8%) | Critique |
| **U2R** | User to Root | 52 (0.04%) | **Extrême** |

---

## 🔬 Méthodologie et Pipeline ML

### 1. Preprocessing
- Mapping des 39 types d'attaques vers 5 grandes catégories.
- **One-Hot Encoding** des colonnes catégorielles → 122 features finales.
- **StandardScaling** pour la normalisation (sauvegardé dans `scaler.pkl`).

### 2. Le défi du Déséquilibre (SMOTE Pipeline)
Pour contrer le déséquilibre massif (seulement 0.04% d'attaques U2R !), nous utilisons la librairie `imbalanced-learn`.
> ⚠️ **Zero Data Leakage** : Le SMOTE est appliqué rigoureusement *à l'intérieur* de la validation croisée (`imblearn.pipeline.Pipeline`). Ainsi, les données synthétiques ne débordent jamais sur le jeu de validation.

### 3. Hyperparameter Tuning (RandomizedSearchCV)
Les modèles ne sont pas configurés au hasard. Nous utilisons une validation croisée stratifiée à 3 plis (3-Fold Stratified CV) couplée à une recherche aléatoire (`RandomizedSearchCV`) pour trouver la profondeur d'arbre et le taux d'apprentissage optimaux.

---

## 🏆 Résultats des Modèles Optimisés

Le modèle final sélectionné pour la production est **XGBoost**.

### Comparaison Globale (Test Set)

| Métrique | 🌲 Random Forest (SMOTE) | 🚀 XGBoost (SMOTE) |
|----------|:-:|:-:|
| **Accuracy (Exactitude)** | 75.00% | **78.95%** |
| **F1-Score Macro** | 0.5342 | **0.6314** |
| **F1-Score Weighted** | 0.7087 | **0.7582** |

### Détection des Attaques Rares (Le véritable exploit)

Grâce au SMOTE, le modèle XGBoost parvient enfin à déceler les intrusions les plus furtives :
- **Attaques U2R** : F1-Score passé de 0.17 à **0.46** ! (Précision : 71.88%).
- **Attaques R2L** : Précision exceptionnelle de **97.11%**.
- **Attaques DoS** : Détection quasi-parfaite avec une Précision de **96.57%**.

---

## ⚡ L'Inspiration Morpheus : Preuve de Concept Temps Réel

Un bon IDS ne sert à rien s'il ralentit le trafic réseau. Nous avons benchmarké l'inférence du modèle final (XGBoost) :

| Taille du Batch réseau | Temps d'inférence mesuré |
|------------------------|-------------------------|
| 1 requête unitaire     | **~ 7 ms**             |
| 1 000 requêtes         | **~ 28 ms**            |
| 10 000 requêtes        | **~ 218 ms**           |

> **Conclusion** : Le système est capable de classer plus de **45 000 connexions par seconde** sur un processeur standard. L'exigence du "temps réel" est pleinement validée.

---

## 🚀 Déploiement & Lancement

Ce projet n'est plus qu'un simple script, c'est un produit complet. Vous avez deux options pour le lancer :

## ⚙️ Génération des modèles (étape obligatoire)

Avant de lancer l'API ou Docker, vous devez entraîner et générer les modèles en local. Les modèles (`.pkl`) ne sont pas versionnés sur GitHub pour des raisons de bonnes pratiques MLOps.

```bash
# Depuis la racine du projet
python src/run_advanced_models.py
```

Cette commande génère `models/best_model_cv.pkl`, `models/scaler.pkl`, et `models/model_columns.pkl` qui sont requis par l'API.

### Option A : Déploiement Docker (Recommandé / MLOps)
Le projet est entièrement conteneurisé. L'API et le Dashboard tournent dans des conteneurs séparés.
```bash
docker-compose up -d --build
```
- Le Dashboard sera accessible sur `http://localhost:8080`
- L'API sera accessible sur `http://localhost:8000/docs`

### Option B : Déploiement Local (Python)
1. **Installer les dépendances**
```bash
pip install -r requirements.txt
```

2. **Démarrer le backend (FastAPI)**
L'API charge le modèle et simule le flux réseau.
```bash
uvicorn api.api:app --reload
```
- 📘 Testez l'API manuellement sur **http://127.0.0.1:8000/docs** (Interface Swagger).

3. **Ouvrir le Dashboard**
Double-cliquez simplement sur le fichier **`frontend/index.html`** dans votre navigateur web et cliquez sur "Démarrer la Simulation" !

---

*Projet de Fin de Module*

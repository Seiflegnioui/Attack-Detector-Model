# 🛡️ Détection d'Intrusions Réseau par Apprentissage Automatique

## Inspiré de NVIDIA Morpheus

Système de détection d'intrusions réseau (IDS) utilisant des algorithmes de classification supervisée pour distinguer le trafic normal du trafic malveillant.

---

## 📋 Problématique

Les cyberattaques évoluent plus rapidement que les méthodes de détection classiques basées sur des signatures statiques. Ce projet construit un **IDS intelligent** en appliquant des algorithmes de classification supervisée (**Random Forest**, **XGBoost**) pour détecter automatiquement les intrusions réseau : DDoS, port scan, brute force, etc.

Cette approche s'inspire du framework **NVIDIA Morpheus**, qui utilise le machine learning pour détecter les comportements anormaux en temps réel dans les infrastructures réseau.

---

## 📊 Dataset : NSL-KDD

Le dataset [NSL-KDD](https://www.kaggle.com/datasets/hassan06/nslkdd) est un benchmark public de référence pour la détection d'intrusions réseau.

| Fichier | Lignes | Description |
|---------|--------|-------------|
| `KDDTrain+.txt` | 125,973 | Jeu d'entraînement complet |
| `KDDTest+.txt` | 22,544 | Jeu de test complet |
| `KDDTrain+_20Percent.txt` | 25,192 | 20% du train set |
| `KDDTest-21.txt` | 11,850 | Test filtré (difficulté < 21) |

### Classes d'attaques (5 catégories)

| Classe | Description | Exemples | Train Samples |
|--------|-------------|----------|:-------------:|
| **Normal** | Trafic légitime | — | 67,343 (53.5%) |
| **DoS** | Déni de service | neptune, smurf, back | 45,927 (36.5%) |
| **Probe** | Scan/Reconnaissance | satan, ipsweep, portsweep | 11,656 (9.3%) |
| **R2L** | Remote to Local | guess_passwd, warezmaster | 995 (0.8%) |
| **U2R** | User to Root | buffer_overflow, rootkit | 52 (0.04%) |

### Features (41 attributs)

Le dataset contient 41 features réseau réparties en 3 catégories :
- **Features de base** (9) : durée, protocole, service, flag, octets src/dst...
- **Features de contenu** (13) : tentatives de login, accès root, fichiers créés...
- **Features de trafic** (19) : taux d'erreur, connexions similaires, taux srv...

---

## 🏗️ Architecture du Projet

```
Attack-Detector-Model/
├── KDDTrain+.txt              # Dataset d'entraînement
├── KDDTest+.txt               # Dataset de test
├── KDDTrain+_20Percent.txt    # Sous-ensemble 20%
├── KDDTest-21.txt             # Test filtré
├── preparedata.ipynb          # Notebook d'exploration et preprocessing
├── model_training.ipynb       # Notebook d'entraînement et évaluation
├── run_models.py              # Script standalone (alternative au notebook)
├── rf_model.pkl               # Modèle Random Forest sauvegardé
├── xgb_model.pkl              # Modèle XGBoost sauvegardé
├── best_model.pkl             # Meilleur modèle (XGBoost)
├── scaler.pkl                 # StandardScaler sauvegardé
├── confusion_matrices.png     # Matrices de confusion
├── model_comparison.png       # Comparaison F1/Precision/Recall
├── roc_curves.png             # Courbes ROC multiclasse
├── feature_importance.png     # Top 20 features importantes
├── index.html                 # Page de description du dataset NSL-KDD
└── README.md
```

---

## 🔬 Méthodologie

### 1. Preprocessing
- Suppression de la colonne `difficulty_level`
- Mapping des 39 types d'attaques vers 5 catégories (Normal, DoS, Probe, R2L, U2R)
- **One-Hot Encoding** des colonnes catégorielles (protocol_type, service, flag) → 122 features
- **Normalisation** avec StandardScaler (mean=0, std=1)
- **Class Weights** calculés avec `compute_class_weight('balanced')` pour gérer le déséquilibre

### 2. Modèles

#### 🌲 Random Forest
- 200 arbres, profondeur max 25
- Class weights balancés
- Parallélisation sur tous les CPU

#### 🚀 XGBoost
- 200 arbres, profondeur max 10, learning rate 0.1
- Subsample 80%, colsample 80%
- Sample weights pour gérer le déséquilibre

### 3. Évaluation
- Matrice de confusion (brute + normalisée)
- Classification report (precision, recall, F1 par classe)
- Courbes ROC multiclasse (One-vs-Rest)
- Feature importance (Top 20)

---

## 📈 Résultats

### Comparaison globale

| Métrique | 🌲 Random Forest | 🚀 XGBoost |
|----------|:-:|:-:|
| **Accuracy** | 74.46% | **78.45%** |
| **F1-Score Macro** | 0.5225 | **0.5979** |
| **F1-Score Weighted** | 0.7052 | **0.7481** |
| **Temps d'entraînement** | **7.80s** | 44.93s |

### Performance par classe

| Classe | RF Precision | RF Recall | RF F1 | XGB Precision | XGB Recall | XGB F1 |
|--------|:-:|:-:|:-:|:-:|:-:|:-:|
| Normal | 0.643 | **0.972** | 0.774 | 0.692 | **0.972** | **0.808** |
| DoS | **0.961** | 0.765 | 0.852 | **0.964** | 0.834 | **0.895** |
| Probe | **0.844** | 0.607 | 0.706 | 0.811 | 0.727 | **0.767** |
| R2L | 0.964 | 0.056 | 0.106 | **0.973** | 0.088 | **0.161** |
| U2R | 0.539 | 0.104 | 0.175 | **0.727** | 0.239 | **0.360** |

### 🏆 Meilleur modèle : **XGBoost** (F1 Macro = 0.5979)

XGBoost surpasse Random Forest sur toutes les classes, en particulier sur les classes rares :
- **U2R** : +105% de F1-Score
- **R2L** : +52% de F1-Score
- **Probe** : +9% de F1-Score

> **Note** : Le recall faible sur R2L/U2R s'explique par le fait que le test set contient 17 types d'attaques absents du train set, représentant un vrai défi de généralisation.

---

## 🔗 Inspiration NVIDIA Morpheus

Notre approche s'inspire du framework [NVIDIA Morpheus](https://developer.nvidia.com/morpheus-cybersecurity) :

| Aspect | Notre Projet | NVIDIA Morpheus |
|--------|:--:|:--:|
| **Pipeline** | CSV → Preprocessing → ML → Classification | Streaming → GPU Preprocessing → DL/ML → Alerte |
| **Algorithmes** | Random Forest, XGBoost | Random Forest, XGBoost, Autoencoders, LSTM |
| **Détection** | 5 classes (Normal + 4 attaques) | Multi-catégories temps réel |
| **Déséquilibre** | Class weights balancés | Techniques avancées (SMOTE, etc.) |
| **Infrastructure** | CPU (scikit-learn) | GPU (RAPIDS/cuML) |
| **Données** | Batch (fichiers CSV) | Streaming temps réel (Kafka) |

### Points communs
1. **Pipeline structuré** : ingestion → preprocessing → feature engineering → classification
2. **Classification supervisée** avec des modèles ensemblistes
3. **Détection multi-classes** pour catégoriser les types d'attaques
4. **Gestion du déséquilibre** des classes dans les données de sécurité

### Différences
- Morpheus fonctionne en **temps réel sur GPU** ; notre projet est en batch sur CPU
- Morpheus intègre du **deep learning** (autoencoders, LSTM) en plus du ML classique
- Morpheus traite des **flux de données en streaming**, pas des fichiers statiques

---

## 🚀 Comment Reproduire

### Prérequis
```bash
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn xgboost joblib pandas matplotlib seaborn numpy
```

### Exécution
```bash
# Option 1 : Script standalone
python run_models.py

# Option 2 : Notebook Jupyter
pip install jupyter
jupyter notebook model_training.ipynb
```

### Charger un modèle sauvegardé
```python
import joblib

model = joblib.load('best_model.pkl')
scaler = joblib.load('scaler.pkl')

# Prédiction sur de nouvelles données
X_new_scaled = scaler.transform(X_new)
predictions = model.predict(X_new_scaled)
```

---

## 📚 Références

1. M. Tavallaee, E. Bagheri, W. Lu, and A. Ghorbani, *A Detailed Analysis of the KDD CUP 99 Data Set*, Second IEEE Symposium on Computational Intelligence for Security and Defense Applications (CISDA), 2009.
2. J. McHugh, *Testing intrusion detection systems: a critique of the 1998 and 1999 DARPA intrusion detection system evaluations as performed by Lincoln Laboratory*, ACM Transactions on Information and System Security, vol. 3, no. 4, pp. 262-294, 2000.
3. NVIDIA Morpheus — [https://developer.nvidia.com/morpheus-cybersecurity](https://developer.nvidia.com/morpheus-cybersecurity)

---

## 👤 Auteur

**Seif Legnioui** — Projet de fin de module : Détection d'intrusions réseau par apprentissage automatique

"""
Advanced script for ML pipeline with SMOTE and Real Time Benchmarking.
Includes Hyperparameter Tuning (RandomizedSearchCV) and Stratified K-Fold.
Run with: python -X utf8 run_advanced_models.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_curve, auc, precision_recall_fscore_support
)
import xgboost as xgb
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
CLASS_NAMES = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
CLASS_LABELS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
COLORS = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#e67e22']

print("="*70)
print("  DÉTECTION D'INTRUSIONS RÉSEAU — NSL-KDD (ADVANCED)")
print("  RF vs XGBoost + SMOTE Pipeline + Benchmark Temps Réel")
print("="*70)

# --- Data Loading ---
columns = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level'
]

train_df = pd.read_csv('KDDTrain+.txt', names=columns)
test_df = pd.read_csv('KDDTest+.txt', names=columns)
train_df = train_df.drop('difficulty_level', axis=1)
test_df = test_df.drop('difficulty_level', axis=1)
print(f"\n📊 Training set : {train_df.shape[0]:,} lignes")
print(f"📊 Test set     : {test_df.shape[0]:,} lignes")

# --- Preprocessing ---
attack_mapping = {
    'neptune': 'DoS', 'smurf': 'DoS', 'back': 'DoS', 'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'portsweep': 'Probe', 'nmap': 'Probe',
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L', 'imap': 'R2L',
    'ftp_write': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'buffer_overflow': 'U2R', 'rootkit': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'normal': 'Normal',
    'apache2': 'DoS', 'udpstorm': 'DoS', 'processtable': 'DoS', 'mailbomb': 'DoS', 'worm': 'DoS',
    'mscan': 'Probe', 'saint': 'Probe',
    'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
    'xlock': 'R2L', 'xsnoop': 'R2L', 'httptunnel': 'R2L',
    'ps': 'U2R', 'xterm': 'U2R', 'sqlattack': 'U2R'
}

train_df['attack_class'] = train_df['label'].map(attack_mapping)
test_df['attack_class'] = test_df['label'].map(attack_mapping)
train_df = train_df.drop('label', axis=1)
test_df = test_df.drop('label', axis=1)

class_mapping = {'Normal': 0, 'DoS': 1, 'Probe': 2, 'R2L': 3, 'U2R': 4}
train_df['attack_class'] = train_df['attack_class'].map(class_mapping)
test_df['attack_class'] = test_df['attack_class'].map(class_mapping)

y_train = train_df['attack_class']
y_test = test_df['attack_class']

X_train_raw = train_df.drop(['attack_class'], axis=1)
X_test_raw = test_df.drop(['attack_class'], axis=1)

categorical_cols = ['protocol_type', 'service', 'flag']
X_train = pd.get_dummies(X_train_raw, columns=categorical_cols)
X_test = pd.get_dummies(X_test_raw, columns=categorical_cols)
X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_train.columns)

print(f"\n✅ Preprocessing terminé — Features shape: {X_train_scaled.shape}")
print("INFO: Suppression des class_weights car SMOTE est utilisé dans le pipeline.")

# Stratified K-Fold setup
cv_strategy = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# --- Random Forest ---
print(f"\n{'='*70}")
print("🌲 RANDOM FOREST - SMOTE PIPELINE")
print(f"{'='*70}")

rf_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42, n_jobs=1))
])

rf_param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [15, 25, 35, None],
    'classifier__min_samples_split': [2, 5, 10],
    'classifier__min_samples_leaf': [1, 2, 4]
}

rf_search = RandomizedSearchCV(
    rf_pipeline, param_distributions=rf_param_grid, 
    n_iter=5, cv=cv_strategy, scoring='f1_macro', 
    n_jobs=-1, verbose=1, random_state=42
)

print("Lancement de la recherche avec SMOTE (cela peut prendre quelques minutes)...")
start_time = time.time()
rf_search.fit(X_train_scaled, y_train)
rf_train_time = time.time() - start_time

best_rf = rf_search.best_estimator_
print(f"Meilleurs hyperparamètres RF : {rf_search.best_params_}")
print(f"Meilleur score CV (F1 Macro) : {rf_search.best_score_:.4f}")

start_time = time.time()
rf_y_pred = best_rf.predict(X_test_scaled)
rf_pred_time = time.time() - start_time
rf_y_proba = best_rf.predict_proba(X_test_scaled)

print(f"\n⏱️  Temps total recherche RF : {rf_train_time:.2f}s | Prédiction : {rf_pred_time:.4f}s")
print(f"🎯 F1 Macro (Test)  : {f1_score(y_test, rf_y_pred, average='macro'):.4f}")
print(f"\n{classification_report(y_test, rf_y_pred, target_names=CLASS_LABELS, digits=4)}")

# --- XGBoost ---
print(f"\n{'='*70}")
print("🚀 XGBOOST - SMOTE PIPELINE")
print(f"{'='*70}")

xgb_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', xgb.XGBClassifier(
        objective='multi:softprob', num_class=5, eval_metric='mlogloss',
        random_state=42, verbosity=0, n_jobs=1
    ))
])

xgb_param_grid = {
    'classifier__n_estimators': [100, 200, 300],
    'classifier__max_depth': [6, 10, 15],
    'classifier__learning_rate': [0.01, 0.1, 0.2],
    'classifier__subsample': [0.8, 1.0],
    'classifier__colsample_bytree': [0.8, 1.0]
}

xgb_search = RandomizedSearchCV(
    xgb_pipeline, param_distributions=xgb_param_grid, 
    n_iter=5, cv=cv_strategy, scoring='f1_macro', 
    n_jobs=-1, verbose=1, random_state=42
)

print("Lancement de la recherche avec SMOTE (cela peut prendre quelques minutes)...")
start_time = time.time()
xgb_search.fit(X_train_scaled, y_train)
xgb_train_time = time.time() - start_time

best_xgb = xgb_search.best_estimator_
print(f"Meilleurs hyperparamètres XGB : {xgb_search.best_params_}")
print(f"Meilleur score CV (F1 Macro) : {xgb_search.best_score_:.4f}")

start_time = time.time()
xgb_y_pred = best_xgb.predict(X_test_scaled)
xgb_pred_time = time.time() - start_time
xgb_y_proba = best_xgb.predict_proba(X_test_scaled)

print(f"\n⏱️  Temps total recherche XGB : {xgb_train_time:.2f}s | Prédiction : {xgb_pred_time:.4f}s")
print(f"🎯 F1 Macro (Test)  : {f1_score(y_test, xgb_y_pred, average='macro'):.4f}")
print(f"\n{classification_report(y_test, xgb_y_pred, target_names=CLASS_LABELS, digits=4)}")

# --- Comparison ---
print(f"\n{'='*70}")
print("📊 COMPARAISON FINALE DES MODÈLES OPTIMISÉS (AVEC SMOTE)")
print(f"{'='*70}")

rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_test, rf_y_pred, average=None)
xgb_precision, xgb_recall, xgb_f1, _ = precision_recall_fscore_support(y_test, xgb_y_pred, average=None)

per_class = pd.DataFrame({
    'Classe': CLASS_LABELS,
    'RF_F1': rf_f1, 'XGB_F1': xgb_f1,
    'RF_Precision': rf_precision, 'XGB_Precision': xgb_precision,
    'RF_Recall': rf_recall, 'XGB_Recall': xgb_recall,
})
print(per_class.to_string(index=False, float_format='%.4f'))

# --- Benchmarking ---
print(f"\n{'='*70}")
print("⏱️  BENCHMARK DE LATENCE (INSPIRATION MORPHEUS) - XGBoost")
print(f"{'='*70}")
print("Évaluation de la capacité du modèle à tenir la charge d'un flux réseau temps réel.")

def benchmark_latency(model, X_test, num_samples):
    if num_samples <= len(X_test):
        X_sample = X_test.sample(n=num_samples, random_state=42)
    else:
        X_sample = X_test.sample(n=num_samples, replace=True, random_state=42)
        
    start_time = time.perf_counter()
    model.predict(X_sample)
    end_time = time.perf_counter()
    
    latency_ms = (end_time - start_time) * 1000
    print(f"Batch de {num_samples:5d} connexion(s) : {latency_ms:6.2f} ms")

benchmark_latency(best_xgb, X_test_scaled, 1)
benchmark_latency(best_xgb, X_test_scaled, 1000)
benchmark_latency(best_xgb, X_test_scaled, 10000)

# --- Plots ---
# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
rf_cm = confusion_matrix(y_test, rf_y_pred)
xgb_cm = confusion_matrix(y_test, xgb_y_pred)

sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS, ax=axes[0], linewidths=0.5)
axes[0].set_title('SMOTE Random Forest — Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Vrai label'); axes[0].set_xlabel('Label prédit')

sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS, ax=axes[1], linewidths=0.5)
axes[1].set_title('SMOTE XGBoost — Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Vrai label'); axes[1].set_xlabel('Label prédit')

plt.tight_layout()
plt.savefig('confusion_matrices_optimized.png', dpi=150, bbox_inches='tight')
print("\n💾 confusion_matrices_optimized.png")

# Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
x = np.arange(len(CLASS_LABELS)); width = 0.35

for ax, metric_rf, metric_xgb, title in [
    (axes[0], rf_f1, xgb_f1, 'F1-Score'),
    (axes[1], rf_precision, xgb_precision, 'Precision'),
    (axes[2], rf_recall, xgb_recall, 'Recall')
]:
    ax.bar(x - width/2, metric_rf, width, label='SMOTE RF', color='#3498db', alpha=0.85)
    ax.bar(x + width/2, metric_xgb, width, label='SMOTE XGB', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=45)
    ax.set_title(f'{title} par Classe', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(axis='y', alpha=0.3)

plt.suptitle('Comparaison Random Forest vs XGBoost (SMOTE + CV)', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model_comparison_optimized.png', dpi=150, bbox_inches='tight')
print("💾 model_comparison_optimized.png")

# ROC curves
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, y_proba, title in [(axes[0], rf_y_proba, 'SMOTE Random Forest'), (axes[1], xgb_y_proba, 'SMOTE XGBoost')]:
    for i in range(5):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], lw=2, label=f'{CLASS_LABELS[i]} (AUC={roc_auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_title(f'{title} — Courbes ROC', fontsize=14, fontweight='bold')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves_optimized.png', dpi=150, bbox_inches='tight')
print("💾 roc_curves_optimized.png")

# Feature importance
fig, axes = plt.subplots(1, 2, figsize=(20, 8))

# best_rf is a Pipeline. To get feature importances, access the 'classifier' step.
rf_classifier = best_rf.named_steps['classifier']
xgb_classifier = best_xgb.named_steps['classifier']

rf_imp = pd.Series(rf_classifier.feature_importances_, index=X_train.columns).nlargest(20)
xgb_imp = pd.Series(xgb_classifier.feature_importances_, index=X_train.columns).nlargest(20)

rf_imp.sort_values().plot(kind='barh', ax=axes[0], color='#3498db', alpha=0.85)
axes[0].set_title('SMOTE Random Forest — Top 20 Features', fontsize=14, fontweight='bold')
xgb_imp.sort_values().plot(kind='barh', ax=axes[1], color='#e74c3c', alpha=0.85)
axes[1].set_title('SMOTE XGBoost — Top 20 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance_optimized.png', dpi=150, bbox_inches='tight')
print("💾 feature_importance_optimized.png")

# --- Save models ---
joblib.dump(best_rf, 'rf_model_cv.pkl')
joblib.dump(best_xgb, 'xgb_model_cv.pkl')
joblib.dump(scaler, 'scaler.pkl')

rf_f1_macro = f1_score(y_test, rf_y_pred, average='macro')
xgb_f1_macro = f1_score(y_test, xgb_y_pred, average='macro')

if xgb_f1_macro > rf_f1_macro:
    joblib.dump(best_xgb, 'best_model_cv.pkl')
    print(f"\n🏆 Meilleur modèle final : XGBoost (F1 Macro: {xgb_f1_macro:.4f})")
else:
    joblib.dump(best_rf, 'best_model_cv.pkl')
    print(f"\n🏆 Meilleur modèle final : Random Forest (F1 Macro: {rf_f1_macro:.4f})")

print("\n✅ Terminé ! Modèles optimisés sauvegardés (rf_model_cv.pkl, xgb_model_cv.pkl, best_model_cv.pkl)")

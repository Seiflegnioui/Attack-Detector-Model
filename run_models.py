"""
Quick test script to verify model training pipeline works.
Run with: python run_models.py
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    f1_score, roc_curve, auc, roc_auc_score, precision_recall_fscore_support
)
import xgboost as xgb
import joblib
import time
import warnings
warnings.filterwarnings('ignore')

plt.style.use('seaborn-v0_8-darkgrid')
CLASS_NAMES = {0: 'Normal', 1: 'DoS', 2: 'Probe', 3: 'R2L', 4: 'U2R'}
CLASS_LABELS = ['Normal', 'DoS', 'Probe', 'R2L', 'U2R']
COLORS = ['#2ecc71', '#e74c3c', '#f39c12', '#9b59b6', '#e67e22']

print("="*60)
print("  DÉTECTION D'INTRUSIONS RÉSEAU — NSL-KDD")
print("  Random Forest vs XGBoost")
print("="*60)

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
    # Train set attacks
    'neptune': 'DoS', 'smurf': 'DoS', 'back': 'DoS', 'teardrop': 'DoS', 'pod': 'DoS', 'land': 'DoS',
    'satan': 'Probe', 'ipsweep': 'Probe', 'portsweep': 'Probe', 'nmap': 'Probe',
    'warezclient': 'R2L', 'guess_passwd': 'R2L', 'warezmaster': 'R2L', 'imap': 'R2L',
    'ftp_write': 'R2L', 'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L',
    'buffer_overflow': 'U2R', 'rootkit': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
    'normal': 'Normal',
    # Test set additional attacks
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

class_weights_array = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weights_dict = dict(enumerate(class_weights_array))

print(f"\n✅ Preprocessing terminé — Features shape: {X_train_scaled.shape}")

# --- Random Forest ---
print(f"\n{'='*60}")
print("🌲 RANDOM FOREST")
print(f"{'='*60}")

rf_model = RandomForestClassifier(
    n_estimators=200, max_depth=25, min_samples_split=5, min_samples_leaf=2,
    class_weight=class_weights_dict, random_state=42, n_jobs=-1
)

start_time = time.time()
rf_model.fit(X_train_scaled, y_train)
rf_train_time = time.time() - start_time

start_time = time.time()
rf_y_pred = rf_model.predict(X_test_scaled)
rf_pred_time = time.time() - start_time
rf_y_proba = rf_model.predict_proba(X_test_scaled)

print(f"⏱️  Entraînement : {rf_train_time:.2f}s | Prédiction : {rf_pred_time:.4f}s")
print(f"🎯 Accuracy  : {accuracy_score(y_test, rf_y_pred):.4f}")
print(f"🎯 F1 Macro  : {f1_score(y_test, rf_y_pred, average='macro'):.4f}")
print(f"🎯 F1 Weight : {f1_score(y_test, rf_y_pred, average='weighted'):.4f}")
print(f"\n{classification_report(y_test, rf_y_pred, target_names=CLASS_LABELS, digits=4)}")

# --- XGBoost ---
print(f"\n{'='*60}")
print("🚀 XGBOOST")
print(f"{'='*60}")

sample_weights = y_train.map(class_weights_dict).values

xgb_model = xgb.XGBClassifier(
    n_estimators=200, max_depth=10, learning_rate=0.1,
    subsample=0.8, colsample_bytree=0.8,
    objective='multi:softprob', num_class=5, eval_metric='mlogloss',
    random_state=42, n_jobs=-1, verbosity=0
)

start_time = time.time()
xgb_model.fit(X_train_scaled, y_train, sample_weight=sample_weights)
xgb_train_time = time.time() - start_time

start_time = time.time()
xgb_y_pred = xgb_model.predict(X_test_scaled)
xgb_pred_time = time.time() - start_time
xgb_y_proba = xgb_model.predict_proba(X_test_scaled)

print(f"⏱️  Entraînement : {xgb_train_time:.2f}s | Prédiction : {xgb_pred_time:.4f}s")
print(f"🎯 Accuracy  : {accuracy_score(y_test, xgb_y_pred):.4f}")
print(f"🎯 F1 Macro  : {f1_score(y_test, xgb_y_pred, average='macro'):.4f}")
print(f"🎯 F1 Weight : {f1_score(y_test, xgb_y_pred, average='weighted'):.4f}")
print(f"\n{classification_report(y_test, xgb_y_pred, target_names=CLASS_LABELS, digits=4)}")

# --- Comparison ---
print(f"\n{'='*60}")
print("📊 COMPARAISON FINALE")
print(f"{'='*60}")

rf_precision, rf_recall, rf_f1, _ = precision_recall_fscore_support(y_test, rf_y_pred, average=None)
xgb_precision, xgb_recall, xgb_f1, _ = precision_recall_fscore_support(y_test, xgb_y_pred, average=None)

per_class = pd.DataFrame({
    'Classe': CLASS_LABELS,
    'RF_F1': rf_f1, 'XGB_F1': xgb_f1,
    'RF_Precision': rf_precision, 'XGB_Precision': xgb_precision,
    'RF_Recall': rf_recall, 'XGB_Recall': xgb_recall,
})
print(per_class.to_string(index=False, float_format='%.4f'))

# --- Plots ---
# Confusion matrices
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
rf_cm = confusion_matrix(y_test, rf_y_pred)
xgb_cm = confusion_matrix(y_test, xgb_y_pred)

sns.heatmap(rf_cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS, ax=axes[0], linewidths=0.5)
axes[0].set_title('Random Forest — Confusion Matrix', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Vrai label'); axes[0].set_xlabel('Label prédit')

sns.heatmap(xgb_cm, annot=True, fmt='d', cmap='Oranges', xticklabels=CLASS_LABELS,
            yticklabels=CLASS_LABELS, ax=axes[1], linewidths=0.5)
axes[1].set_title('XGBoost — Confusion Matrix', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Vrai label'); axes[1].set_xlabel('Label prédit')

plt.tight_layout()
plt.savefig('confusion_matrices.png', dpi=150, bbox_inches='tight')
print("\n💾 confusion_matrices.png")

# Comparison bar chart
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
x = np.arange(len(CLASS_LABELS)); width = 0.35

for ax, metric_rf, metric_xgb, title in [
    (axes[0], rf_f1, xgb_f1, 'F1-Score'),
    (axes[1], rf_precision, xgb_precision, 'Precision'),
    (axes[2], rf_recall, xgb_recall, 'Recall')
]:
    ax.bar(x - width/2, metric_rf, width, label='Random Forest', color='#3498db', alpha=0.85)
    ax.bar(x + width/2, metric_xgb, width, label='XGBoost', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(CLASS_LABELS, rotation=45)
    ax.set_title(f'{title} par Classe', fontsize=14, fontweight='bold')
    ax.set_ylim(0, 1.1); ax.legend(); ax.grid(axis='y', alpha=0.3)

plt.suptitle('Comparaison Random Forest vs XGBoost', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=150, bbox_inches='tight')
print("💾 model_comparison.png")

# ROC curves
y_test_bin = label_binarize(y_test, classes=[0, 1, 2, 3, 4])
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

for ax, y_proba, title in [(axes[0], rf_y_proba, 'Random Forest'), (axes[1], xgb_y_proba, 'XGBoost')]:
    for i in range(5):
        fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_proba[:, i])
        roc_auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, color=COLORS[i], lw=2, label=f'{CLASS_LABELS[i]} (AUC={roc_auc_val:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5)
    ax.set_title(f'{title} — Courbes ROC', fontsize=14, fontweight='bold')
    ax.set_xlabel('FPR'); ax.set_ylabel('TPR')
    ax.legend(loc='lower right'); ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('roc_curves.png', dpi=150, bbox_inches='tight')
print("💾 roc_curves.png")

# Feature importance
fig, axes = plt.subplots(1, 2, figsize=(20, 8))
rf_imp = pd.Series(rf_model.feature_importances_, index=X_train.columns).nlargest(20)
xgb_imp = pd.Series(xgb_model.feature_importances_, index=X_train.columns).nlargest(20)

rf_imp.sort_values().plot(kind='barh', ax=axes[0], color='#3498db', alpha=0.85)
axes[0].set_title('Random Forest — Top 20 Features', fontsize=14, fontweight='bold')
xgb_imp.sort_values().plot(kind='barh', ax=axes[1], color='#e74c3c', alpha=0.85)
axes[1].set_title('XGBoost — Top 20 Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=150, bbox_inches='tight')
print("💾 feature_importance.png")

# --- Save models ---
joblib.dump(rf_model, 'rf_model.pkl')
joblib.dump(xgb_model, 'xgb_model.pkl')
joblib.dump(scaler, 'scaler.pkl')

rf_f1_macro = f1_score(y_test, rf_y_pred, average='macro')
xgb_f1_macro = f1_score(y_test, xgb_y_pred, average='macro')

if xgb_f1_macro > rf_f1_macro:
    joblib.dump(xgb_model, 'best_model.pkl')
    print(f"\n🏆 Meilleur modèle : XGBoost (F1 Macro: {xgb_f1_macro:.4f})")
else:
    joblib.dump(rf_model, 'best_model.pkl')
    print(f"\n🏆 Meilleur modèle : Random Forest (F1 Macro: {rf_f1_macro:.4f})")

print("\n✅ Terminé ! Modèles sauvegardés (rf_model.pkl, xgb_model.pkl, best_model.pkl)")

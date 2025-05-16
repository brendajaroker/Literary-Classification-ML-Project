import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

# Load data
data = pd.read_csv('text_features.csv')
X = data.drop(['Movement', 'Author', 'GutenbergID'], axis=1)  # Features
y = data['Movement']  # Labels

# Train/validation/test split (69.63% train, 15.19% val, 15.19% test)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3037, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Apply SMOTE to training data
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

# Pipeline for original SVM
scaler = StandardScaler()
selector = SelectKBest(score_func=f_classif, k=50)
X_train_scaled = scaler.fit_transform(X_train_smote)
X_train_selected = selector.fit_transform(X_train_scaled, y_train_smote)
X_val_scaled = scaler.transform(X_val)
X_val_selected = selector.transform(X_val_scaled)
X_test_scaled = scaler.transform(X_test)
X_test_selected = selector.transform(X_test_scaled)

# SVM with GridSearchCV
param_grid = {'C': [0.1, 0.3, 0.5], 'gamma': ['scale', 0.005], 'kernel': ['rbf'], 'class_weight': ['balanced']}
svm = SVC()
grid_search = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_selected, y_train_smote)

# Evaluate original model
best_svm = grid_search.best_estimator_
y_pred_test = best_svm.predict(X_test_selected)
original_accuracy = accuracy_score(y_test, y_pred_test)
original_f1_renaissance = f1_score(y_test, y_pred_test, labels=['Renaissance'], average=None)[0]
print(f"Original SVM - Test Accuracy: {original_accuracy:.4f}, Renaissance F1: {original_f1_renaissance:.4f}")

# Ablation study
features_to_remove = ['semicolons', 'present_tense_ratio']
X_train_ablated = X_train_smote.drop(features_to_remove, axis=1, errors='ignore')
X_val_ablated = X_val.drop(features_to_remove, axis=1, errors='ignore')
X_test_ablated = X_test.drop(features_to_remove, axis=1, errors='ignore')

# New pipeline for ablated features
X_train_ablated_scaled = scaler.fit_transform(X_train_ablated)
X_train_ablated_selected = selector.fit_transform(X_train_ablated_scaled, y_train_smote)
X_val_ablated_scaled = scaler.transform(X_val_ablated)
X_val_ablated_selected = selector.transform(X_val_ablated_scaled)
X_test_ablated_scaled = scaler.transform(X_test_ablated)
X_test_ablated_selected = selector.transform(X_test_ablated_scaled)

# Train and evaluate ablated model
grid_search_ablated = GridSearchCV(svm, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search_ablated.fit(X_train_ablated_selected, y_train_smote)
best_svm_ablated = grid_search_ablated.best_estimator_
y_pred_test_ablated = best_svm_ablated.predict(X_test_ablated_selected)
ablated_accuracy = accuracy_score(y_test, y_pred_test_ablated)
ablated_f1_renaissance = f1_score(y_test, y_pred_test_ablated, labels=['Renaissance'], average=None)[0]
print(f"Ablated SVM (without {features_to_remove}) - Test Accuracy: {ablated_accuracy:.4f}, Renaissance F1: {ablated_f1_renaissance:.4f}")